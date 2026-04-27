"""MCP resource providers for telemetry, logs, redacted config, and memory.

Each provider:

* Owns the URI scheme it claims (``mousedroid://<provider>/...``).
* Returns JSON-serialisable Python objects from :meth:`read`; the
  server is responsible for envelope encoding.
* Refuses to expose anything when its toggle in
  :class:`MCPResourcesConfig` is False.
* Caps every list/length parameter at the configured maximum so a
  client cannot exfiltrate unbounded amounts of data with a single
  request.
* Pipes everything through :func:`redact_value` so secrets — keyed by
  the configured regex — are never returned over MCP.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import MCPConfig, Settings
    from mousedroid.telemetry.log_buffer import LogRingBuffer
    from mousedroid.telemetry.protocol import (
        TelemetryFrame,
        TelemetryPublisherProtocol,
    )

_log = get_logger(__name__)

REDACTED = "<redacted>"

# Hard ceiling on recursion depth inside :func:`redact_value`. The
# redactor only walks dict/list structures, but a malicious or
# pathological input could nest beyond what's reasonable to serialise.
# Defined as a module constant rather than hardcoded inline so callers
# (and tests) can patch it without code changes elsewhere.
MAX_REDACT_DEPTH = 32


def redact_value(value: Any, *, key_pattern: re.Pattern[str], _depth: int = 0) -> Any:
    """Recursively replace values whose dict key matches *key_pattern*.

    The match is performed against the *most recent* dict key only — the
    full path is not collapsed because keys like ``api_key`` should be
    redacted whether they appear at the top level or nested. Strings,
    bytes, primitives, and tuples are passed through unchanged. Recursion
    is bounded to a sensible depth to defuse pathological cyclic inputs.

    Args:
        value: Arbitrary input (typically the result of
            :meth:`pydantic.BaseModel.model_dump`).
        key_pattern: Compiled regex; keys matching this pattern have their
            associated value replaced with :data:`REDACTED`.
        _depth: Internal recursion guard; do not pass.

    Returns:
        A structurally identical value with sensitive entries redacted.
    """
    if _depth > MAX_REDACT_DEPTH:
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and key_pattern.search(k):
                out[k] = REDACTED
            else:
                out[k] = redact_value(v, key_pattern=key_pattern, _depth=_depth + 1)
        return out
    if isinstance(value, list):
        return [redact_value(item, key_pattern=key_pattern, _depth=_depth + 1) for item in value]
    return value


def parse_resource_uri(uri: str) -> tuple[str, str, dict[str, str]]:
    """Split a ``mousedroid://`` URI into ``(scheme, path, query)``.

    Args:
        uri: Resource URI (e.g. ``"mousedroid://logs/tail?n=20"``).

    Returns:
        Tuple of ``(scheme, path, query_params)``. Path is normalised
        to a leading-slash form (``/tail``) and query values are scalar
        strings (last value wins for repeated keys).

    Raises:
        ValueError: If the URI is malformed.
    """
    parsed = urlsplit(uri)
    if not parsed.scheme:
        msg = f"resource URI missing scheme: {uri!r}"
        raise ValueError(msg)
    raw_query = parse_qs(parsed.query, keep_blank_values=False)
    query: dict[str, str] = {k: v[-1] for k, v in raw_query.items() if v}
    # urlsplit("mousedroid://telemetry/latest") gives netloc='telemetry',
    # path='/latest'; concat for a uniform "/<provider>/<resource>" form.
    path = f"/{parsed.netloc}{parsed.path}".rstrip("/") or "/"
    return parsed.scheme, path, query


class TelemetryResourceProvider:
    """Exposes recent :class:`TelemetryFrame` snapshots as MCP resources.

    Holds a private ring buffer fed by a sampler task — never reads the
    publisher queue from a request handler so MCP clients cannot starve
    the orchestrator's telemetry consumer.
    """

    URI_LATEST = "mousedroid://telemetry/latest"
    URI_RECENT = "mousedroid://telemetry/recent"

    def __init__(
        self,
        cfg: MCPConfig,
        publisher: TelemetryPublisherProtocol | None,
    ) -> None:
        """Wire the provider.

        Args:
            cfg: MCP configuration (defines the buffer cap).
            publisher: Optional telemetry publisher; ``None`` disables
                the provider entirely (its URIs become empty / refused).
        """
        self._cfg = cfg
        self._publisher = publisher
        self._buffer: deque[TelemetryFrame] = deque(maxlen=cfg.resources.recent_frames_max)

    @property
    def enabled(self) -> bool:
        """Whether telemetry resources are exposed at all."""
        return self._cfg.resources.telemetry_enabled and self._publisher is not None

    @property
    def buffer_size(self) -> int:
        """Current number of frames in the local buffer (for diagnostics)."""
        return len(self._buffer)

    async def sample_once(self) -> int:
        """Drain available frames from the publisher into the local buffer.

        Returns:
            Number of frames pulled in this call (``0`` when none ready).
        """
        if self._publisher is None:
            return 0
        queue = self._publisher.get_queue()
        pulled = 0
        # Drain everything available; the deque's maxlen evicts older
        # entries automatically so memory stays bounded. ``get_nowait``
        # is constant-time and the loop exits as soon as the queue is
        # empty, so the 30 Hz control loop is never starved.
        while True:
            try:
                frame = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._buffer.append(frame)
            pulled += 1
        if pulled:
            _log.debug("mcp_telemetry_sample", pulled=pulled, buffer=len(self._buffer))
        return pulled

    def list_uris(self) -> list[str]:
        """Return the resource URIs this provider serves."""
        if not self.enabled:
            return []
        return [self.URI_LATEST, self.URI_RECENT]

    def read(self, path: str, query: dict[str, str]) -> dict[str, Any]:
        """Serve a telemetry resource read.

        Args:
            path: Normalised resource path (``/telemetry/latest`` or
                ``/telemetry/recent``).
            query: Query parameters; ``n`` controls the recent count.

        Returns:
            JSON-friendly payload.

        Raises:
            KeyError: If the path is not recognised.
            PermissionError: If the resource is disabled by config.
        """
        if not self.enabled:
            msg = "telemetry resource disabled"
            raise PermissionError(msg)
        if path == "/telemetry/latest":
            if not self._buffer:
                return {"frame": None}
            return {"frame": self._buffer[-1].to_dict()}
        if path == "/telemetry/recent":
            n_str = query.get("n", str(self._cfg.resources.recent_frames_max))
            try:
                n = int(n_str)
            except ValueError:
                n = self._cfg.resources.recent_frames_max
            n = max(1, min(n, self._cfg.resources.recent_frames_max))
            frames = list(self._buffer)[-n:]
            return {"count": len(frames), "frames": [f.to_dict() for f in frames]}
        msg = f"unknown telemetry resource path: {path!r}"
        raise KeyError(msg)


class LogResourceProvider:
    """Exposes the structlog ring buffer over MCP, with secret redaction."""

    URI_TAIL = "mousedroid://logs/tail"

    def __init__(
        self,
        cfg: MCPConfig,
        log_buffer: LogRingBuffer | None,
        *,
        key_pattern: re.Pattern[str],
    ) -> None:
        """Wire the provider.

        Args:
            cfg: MCP configuration (defines the tail cap).
            log_buffer: Existing ring buffer; ``None`` disables the URI.
            key_pattern: Compiled regex applied to every entry's keys
                before serialisation. Sourced from
                :attr:`MCPConfig.redact_key_pattern`.
        """
        self._cfg = cfg
        self._buffer = log_buffer
        self._key_pattern = key_pattern

    @property
    def enabled(self) -> bool:
        """Whether log resources are exposed."""
        return self._cfg.resources.logs_enabled and self._buffer is not None

    def list_uris(self) -> list[str]:
        """Return the resource URIs this provider serves."""
        return [self.URI_TAIL] if self.enabled else []

    def read(self, path: str, query: dict[str, str]) -> dict[str, Any]:
        """Serve a log tail read with redaction.

        Args:
            path: Normalised path (must be ``/logs/tail``).
            query: ``n`` controls the count.

        Returns:
            ``{"count": int, "entries": [...]}``.

        Raises:
            KeyError: For any path other than ``/logs/tail``.
            PermissionError: When the resource is disabled.
        """
        if not self.enabled or self._buffer is None:
            msg = "logs resource disabled"
            raise PermissionError(msg)
        if path != "/logs/tail":
            msg = f"unknown log resource path: {path!r}"
            raise KeyError(msg)
        n_str = query.get("n", str(self._cfg.resources.log_tail_max))
        try:
            n = int(n_str)
        except ValueError:
            n = self._cfg.resources.log_tail_max
        n = max(1, min(n, self._cfg.resources.log_tail_max))
        entries = self._buffer.get_recent(n)
        redacted: list[dict[str, Any]] = [
            redact_value(entry, key_pattern=self._key_pattern) for entry in entries
        ]
        return {"count": len(redacted), "entries": redacted}


class ConfigResourceProvider:
    """Exposes the redacted root :class:`Settings` via MCP, with TTL cache."""

    URI_REDACTED = "mousedroid://config/redacted"

    def __init__(
        self,
        cfg: MCPConfig,
        root_cfg: Settings,
        *,
        key_pattern: re.Pattern[str],
    ) -> None:
        """Wire the provider.

        Args:
            cfg: MCP configuration (defines the cache TTL).
            root_cfg: Root settings (the source of truth being exposed).
            key_pattern: Compiled regex used by :func:`redact_value`.
        """
        self._cfg = cfg
        self._root_cfg = root_cfg
        self._key_pattern = key_pattern
        self._cache: dict[str, Any] | None = None
        self._cache_at: float = 0.0

    @property
    def enabled(self) -> bool:
        """Whether the redacted-config resource is exposed."""
        return self._cfg.resources.config_enabled

    def list_uris(self) -> list[str]:
        """Return the resource URIs this provider serves."""
        return [self.URI_REDACTED] if self.enabled else []

    def read(self, path: str, query: dict[str, str]) -> dict[str, Any]:
        """Serve the redacted-config snapshot.

        Args:
            path: Must be ``/config/redacted``.
            query: Unused (config snapshot has no parameters).

        Returns:
            ``{"settings": <redacted dict>}``.

        Raises:
            KeyError: For any path other than ``/config/redacted``.
            PermissionError: When the resource is disabled.
        """
        if not self.enabled:
            msg = "config resource disabled"
            raise PermissionError(msg)
        if path != "/config/redacted":
            msg = f"unknown config resource path: {path!r}"
            raise KeyError(msg)
        now = time.monotonic()
        ttl = self._cfg.resources.config_cache_ttl_s
        if self._cache is not None and (now - self._cache_at) < ttl:
            return {"settings": self._cache}
        snapshot = self._root_cfg.model_dump(mode="json")
        redacted = redact_value(snapshot, key_pattern=self._key_pattern)
        self._cache = redacted
        self._cache_at = now
        return {"settings": redacted}


class MemoryResourceProvider:
    """Exposes recent episodic memory snapshots (when memory tier is enabled)."""

    URI_EPISODES_RECENT = "mousedroid://memory/episodes/recent"

    def __init__(
        self,
        cfg: MCPConfig,
        memory_tier: Any | None,
        *,
        key_pattern: re.Pattern[str],
    ) -> None:
        """Wire the provider.

        Args:
            cfg: MCP configuration.
            memory_tier: Object exposing ``episodic.sample(n)`` /
                ``episodic.__len__``; ``None`` disables the URI.
            key_pattern: Regex applied to each sample (drops embedded
                tokens / secrets if any sneak into episode metadata).
        """
        self._cfg = cfg
        self._memory_tier = memory_tier
        self._key_pattern = key_pattern

    @property
    def enabled(self) -> bool:
        """Whether memory snapshots are exposed."""
        return self._cfg.resources.memory_enabled and self._memory_tier is not None

    def list_uris(self) -> list[str]:
        """Return the resource URIs this provider serves."""
        return [self.URI_EPISODES_RECENT] if self.enabled else []

    def read(self, path: str, query: dict[str, str]) -> dict[str, Any]:
        """Serve a recent-episodes snapshot.

        Args:
            path: Must be ``/memory/episodes/recent``.
            query: ``n`` controls the requested count (capped by
                ``recent_frames_max`` because memory snapshots can be
                heavier than telemetry frames).

        Returns:
            ``{"count": int, "episodes": [...]}``. Each episode is
            coerced to a dict via :func:`_episode_to_dict` and run
            through :func:`redact_value`.

        Raises:
            KeyError: For any path other than the supported one.
            PermissionError: When the resource is disabled.
        """
        if not self.enabled or self._memory_tier is None:
            msg = "memory resource disabled"
            raise PermissionError(msg)
        if path != "/memory/episodes/recent":
            msg = f"unknown memory resource path: {path!r}"
            raise KeyError(msg)
        n_str = query.get("n", str(self._cfg.resources.recent_frames_max))
        try:
            n = int(n_str)
        except ValueError:
            n = self._cfg.resources.recent_frames_max
        n = max(1, min(n, self._cfg.resources.recent_frames_max))
        episodic = getattr(self._memory_tier, "episodic", None)
        if episodic is None:
            return {"count": 0, "episodes": []}
        try:
            samples: Iterable[Any] = episodic.sample(n)
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning("mcp_memory_sample_failed", error=str(exc))
            return {"count": 0, "episodes": []}
        episodes = [
            redact_value(_episode_to_dict(s), key_pattern=self._key_pattern) for s in samples
        ]
        return {"count": len(episodes), "episodes": episodes}


def _episode_to_dict(episode: Any) -> dict[str, Any]:
    """Coerce an episodic memory sample into a JSON-friendly dict.

    Strips ``ndarray`` payloads to summary statistics (shape + dtype) so
    raw image tensors are never serialised over MCP.

    Args:
        episode: Whatever the episodic buffer returns (model-specific).

    Returns:
        A dict with primitive / list values only.
    """
    if isinstance(episode, dict):
        return {k: _coerce_field(v) for k, v in episode.items()}
    return {"episode": _coerce_field(episode)}


def _coerce_field(value: Any) -> Any:
    """Coerce a single field, summarising heavy numpy payloads."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dep
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(value, np.ndarray):
        return {
            "ndarray": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {k: _coerce_field(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_coerce_field(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)
