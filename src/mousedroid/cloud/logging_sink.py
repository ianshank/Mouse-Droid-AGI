"""structlog processor that forwards log events to Google Cloud Logging.

Inserted into the structlog processor chain after the ``LogRingBuffer``
and before the renderer.  Forwarding is fire-and-forget to avoid
blocking the 30 Hz control loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import GCPConfig

_log = get_logger(__name__)

_LEVEL_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class CloudLoggingSink:
    """structlog processor that asynchronously forwards events to Cloud Logging.

    Args:
        cfg: GCP configuration.
    """

    def __init__(self, cfg: GCPConfig) -> None:
        self._cfg = cfg
        self._log_cfg = cfg.logging
        self._min_level = _LEVEL_MAP.get(self._log_cfg.min_level.lower(), logging.INFO)
        self._cloud_logger: Any | None = None
        self._started = False
        # Bounded counter for failed forwards. Exposed through the
        # ``forward_failure_count`` property for diagnostic tooling; we
        # deliberately avoid coupling to MetricsRegistry here because
        # this module runs inside the structlog processor chain, which
        # must stay dependency-light.
        self._forward_failures: int = 0

    async def start(self) -> None:
        """Initialise the Cloud Logging client."""
        from google.cloud import logging as cloud_logging

        from mousedroid.cloud._auth import resolve_credentials

        creds, _project = resolve_credentials(self._cfg)
        client_factory = cast(Callable[..., Any], cloud_logging.Client)
        client = client_factory(
            credentials=creds,
            project=self._cfg.project_id,
        )
        logger_factory = cast(Callable[[str], Any], client.logger)
        self._cloud_logger = logger_factory(self._log_cfg.log_name)
        self._started = True
        _log.info("cloud_logging_sink_started", log_name=self._log_cfg.log_name)

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Forward qualifying events to Cloud Logging.

        Args:
            logger: The wrapped logger object.
            method_name: The name of the log method called (e.g. ``"info"``).
            event_dict: The structured event dictionary.

        Returns:
            The event dictionary unchanged (pass-through processor).
        """
        level = _LEVEL_MAP.get(method_name, logging.INFO)
        if level < self._min_level:
            return event_dict

        if not self._started or self._cloud_logger is None:
            return event_dict

        # Fire-and-forget — never block the caller
        try:
            entry = {
                "message": event_dict.get("event", ""),
                "severity": method_name.upper(),
                **{
                    k: v
                    for k, v in event_dict.items()
                    if k != "event" and isinstance(v, str | int | float | bool | type(None))
                },
            }
            self._cloud_logger.log_struct(entry, severity=method_name.upper())
        except Exception:
            # Never let Cloud Logging errors propagate to the control loop.
            # Track occurrences instead of silently swallowing so
            # diagnose tooling can surface a persistent forwarding
            # outage. The structlog stderr writer is intentionally NOT
            # invoked here to avoid reentrancy into the processor chain.
            self._forward_failures += 1

        return event_dict

    @property
    def forward_failure_count(self) -> int:
        """Total Cloud Logging forward failures since process start."""
        return self._forward_failures

    async def close(self) -> None:
        """Release Cloud Logging resources."""
        self._cloud_logger = None
        self._started = False
