"""Google Cloud Logging sink as a structlog processor.

F-032 wired this into ``configure_logging``. F-039 makes the sink honest:

- ``__call__`` never talks to the SDK. It filters, allowlists, and
  ``put_nowait`` onto a bounded queue. That is what "fire-and-forget"
  means here — not a lie about a background thread.
- Drain is an asyncio task started in :meth:`start`. The SDK call runs
  in ``asyncio.to_thread`` so the event loop is not blocked. Tests that
  skip ``start()`` call :meth:`flush` after the processor to drain the
  queue on the caller thread.
- A test that proves the 30 Hz stall exists **only** when both
  ``LoggingConfig.level`` and ``GCPLoggingConfig.min_level`` are DEBUG
  lives in ``tests/unit/cloud/test_logging_sink_tick_level.py``. Default
  INFO overlays never see ``tick_complete``.

The allowlist is a module-level frozenset, not a YAML field: operators
must not be able to widen it to dump mission text into Cloud Logging.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Callable
from typing import Any, Final, cast

from mousedroid.config.schema import GCPConfig
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

_LEVEL_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_SCALAR_TYPES: tuple[type, ...] = (str, int, float, bool, type(None))

# Sentinel enqueued by :meth:`close` so a blocking ``queue.get`` wakes.
_QUEUE_STOP: Final[object] = object()

# Operational keys that may travel to Cloud Logging. Anything else is
# dropped. Keep this set small; a YAML-widenable list would be the
# footgun F-039 exists to close.
CLOUD_LOG_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "backend",
        "degraded",
        "elapsed_ms",
        "emergency",
        "error",
        "event",
        "level",
        "logger",
        "loop_time_ms",
        "ready",
        "robot_id",
        "status",
        "timestamp",
    }
)

# Mission / NL / prompt keys. Redaction wins even if a key is also
# listed in the allowlist.
CLOUD_LOG_REDACTED_KEYS: frozenset[str] = frozenset(
    {
        "command",
        "content",
        "mission",
        "nl",
        "nl_command",
        "prompt",
        "query",
        "raw_text",
        "text",
        "user_content",
    }
)


class CloudLoggingSink:
    """structlog processor that queues entries for Cloud Logging.

    Construction is cheap and never imports ``google.cloud.logging``.
    Call :meth:`start` from an async context to construct the client
    and spawn the drain task. Call :meth:`close` to stop the drain
    and drop the client.

    Tests that inject a mock logger and skip :meth:`start` MUST call
    :meth:`flush` after the processor if they assert on ``log_struct``.
    """

    def __init__(self, cfg: GCPConfig) -> None:
        self._cfg = cfg
        self._log_cfg = cfg.logging
        self._min_level = _LEVEL_MAP.get(self._log_cfg.min_level.lower(), logging.INFO)
        self._cloud_logger: Any | None = None
        self._started = False
        self._drain_task: asyncio.Task[None] | None = None
        self._stop = threading.Event()
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=cfg.logging.queue_maxsize)
        self._drop_count: int = 0
        self._forward_failures: int = 0

    @property
    def drop_count(self) -> int:
        """Entries dropped because the bounded queue was full."""
        return self._drop_count

    @property
    def forward_failure_count(self) -> int:
        """SDK ``log_struct`` failures observed on the drain path."""
        return self._forward_failures

    async def start(self) -> None:
        """Construct the GCP logging client and spawn the drain task.

        Raises:
            ImportError: ``google-cloud-logging`` is not installed.
            Exception: Client construction failed (credentials, network).
        """
        if self._started:
            return
        import google.cloud.logging as cloud_logging

        from mousedroid.cloud._auth import resolve_credentials

        creds, _project = resolve_credentials(self._cfg)
        client_factory = cast(Callable[..., Any], cloud_logging.Client)
        client = client_factory(
            credentials=creds,
            project=self._cfg.project_id,
        )
        logger_factory = cast(Callable[[str], Any], client.logger)
        self._cloud_logger = logger_factory(self._log_cfg.log_name)
        self._stop.clear()
        self._started = True
        loop = asyncio.get_running_loop()
        self._drain_task = loop.create_task(
            self._drain_loop(),
            name="cloud-logging-drain",
        )
        _log.info(
            "cloud_logging_sink_started",
            log_name=self._log_cfg.log_name,
            queue_maxsize=self._log_cfg.queue_maxsize,
        )

    async def close(self) -> None:
        """Stop the drain task, flush remaining entries, drop the client."""
        self._stop.set()
        try:
            self._queue.put_nowait(_QUEUE_STOP)
        except queue.Full:
            self._drop_count += 1
        drain = self._drain_task
        self._drain_task = None
        if drain is not None:
            await self._await_drain(drain)
        self.flush()
        self._cloud_logger = None
        self._started = False

    async def _await_drain(self, drain: asyncio.Task[None]) -> None:
        """Wait for the drain task; cancel it if it hangs."""
        try:
            await asyncio.wait_for(drain, timeout=5.0)
        except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError):
            drain.cancel()
            try:
                await drain
            except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError):
                return

    def flush(self) -> None:
        """Drain the queue on this thread. Tests and :meth:`close` use this.

        Production ``tick()`` must never call this. The drain task owns
        the live path.
        """
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is _QUEUE_STOP:
                continue
            if isinstance(item, dict):
                self._emit(item)

    async def _drain_loop(self) -> None:
        """Pull queued entries until :meth:`close` sets the stop event."""
        while not self._stop.is_set():
            try:
                item = await asyncio.to_thread(self._queue.get, True, 0.2)
            except queue.Empty:
                continue
            if item is _QUEUE_STOP:
                return
            if isinstance(item, dict):
                await asyncio.to_thread(self._emit, item)

    def _emit(self, payload: dict[str, Any]) -> None:
        """Forward one payload to Cloud Logging. Never raises."""
        logger = self._cloud_logger
        if logger is None:
            return
        try:
            severity = str(payload.get("severity", "INFO"))
            logger.log_struct(payload, severity=severity)
        except Exception:
            self._forward_failures += 1

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Queue a filtered, allowlisted copy. Always returns ``event_dict``.

        Never calls the SDK. Never blocks on a full queue — the entry
        is dropped and ``drop_count`` increments.
        """
        level = _LEVEL_MAP.get(method_name, logging.INFO)
        if level < self._min_level:
            return event_dict
        if not self._started or self._cloud_logger is None:
            return event_dict
        try:
            self._queue.put_nowait(self._build_cloud_entry(event_dict, method_name))
        except queue.Full:
            self._drop_count += 1
        return event_dict

    def _build_cloud_entry(
        self,
        event_dict: dict[str, Any],
        method_name: str,
    ) -> dict[str, Any]:
        """Copy allowlisted scalar keys; redact mission/NL fields."""
        event_name = event_dict.get("event", "")
        message = event_name if isinstance(event_name, str) else ""
        payload: dict[str, Any] = {
            "message": message,
            "severity": method_name.upper() if method_name in _LEVEL_MAP else "INFO",
            "robot_id": self._cfg.robot_id,
        }
        for key, value in event_dict.items():
            if key in {"event", "level"}:
                continue
            if key in CLOUD_LOG_REDACTED_KEYS:
                continue
            if key not in CLOUD_LOG_ALLOWED_KEYS:
                continue
            if not isinstance(value, _SCALAR_TYPES):
                continue
            payload[key] = value
        return payload
