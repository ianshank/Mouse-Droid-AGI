"""Structured logging for Claude Code hooks.

Two hard constraints shape this module:

1. **stdout is the hook protocol.** Claude Code parses a hook's stdout as JSON
   (see :mod:`tools.claude_hooks.hookio`). A stray log line on stdout corrupts
   the decision payload, so every record here is written to **stderr**.
2. **No runtime-package import.** Hooks run on every ``Write``/``Edit``, so they
   must not import ``mousedroid`` (torch/faiss/lmdb). ``structlog`` is used when
   importable; otherwise a dependency-free fallback with the same call surface
   keeps behaviour identical.

Debug output (the pending-content preview, resolved config paths, subprocess
argv) is gated behind the environment variable named by :data:`DEBUG_ENV` so an
operator can turn on diagnostics without a code change.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Protocol

#: Set to a truthy value to raise hook logging to DEBUG level.
DEBUG_ENV = "MOUSEDROID_WORKFORCE_DEBUG"

#: Values accepted as "on" for :data:`DEBUG_ENV`.
_TRUTHY = frozenset({"1", "true", "yes", "on", "debug"})

_LEVEL_ORDER = {"debug": 10, "info": 20, "warning": 30, "error": 40}


class HookLogger(Protocol):
    """Minimal structured-logger surface shared by both backends."""

    def debug(self, event: str, **fields: Any) -> None:
        """Emit a debug-level event."""
        ...

    def info(self, event: str, **fields: Any) -> None:
        """Emit an info-level event."""
        ...

    def warning(self, event: str, **fields: Any) -> None:
        """Emit a warning-level event."""
        ...

    def error(self, event: str, **fields: Any) -> None:
        """Emit an error-level event."""
        ...


def debug_enabled(env: dict[str, str] | None = None) -> bool:
    """Return whether debug diagnostics are enabled.

    Args:
        env: Environment mapping to read. Defaults to :data:`os.environ`.

    Returns:
        ``True`` when the debug environment variable is set to a truthy value.
    """
    environ = os.environ if env is None else env
    return environ.get(DEBUG_ENV, "").strip().lower() in _TRUTHY


class _FallbackLogger:
    """Dependency-free structured logger writing JSON lines to stderr.

    Used when ``structlog`` is not importable — for example in a fresh clone
    before ``pip install -e .`` — so a hook still logs rather than crashing.
    """

    def __init__(self, name: str, *, stream: Any | None = None) -> None:
        """Bind the logger name and output stream.

        Args:
            name: Logger name, emitted as the ``logger`` field.
            stream: Output stream. Defaults to :data:`sys.stderr`.
        """
        self._name = name
        self._stream = stream

    def _emit(self, level: str, event: str, fields: dict[str, Any]) -> None:
        if _LEVEL_ORDER[level] < (10 if debug_enabled() else 20):
            return
        record = {
            "event": event,
            "level": level,
            "logger": self._name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **fields,
        }
        stream = sys.stderr if self._stream is None else self._stream
        try:
            stream.write(json.dumps(record, default=str) + "\n")
        except (TypeError, ValueError):  # pragma: no cover - defensive
            stream.write(f"{level} {event} {fields!r}\n")
        stream.flush()

    def debug(self, event: str, **fields: Any) -> None:
        """Emit a debug-level event."""
        self._emit("debug", event, fields)

    def info(self, event: str, **fields: Any) -> None:
        """Emit an info-level event."""
        self._emit("info", event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        """Emit a warning-level event."""
        self._emit("warning", event, fields)

    def error(self, event: str, **fields: Any) -> None:
        """Emit an error-level event."""
        self._emit("error", event, fields)


def _build_structlog_logger(name: str) -> HookLogger | None:
    """Return a stderr-bound structlog logger, or ``None`` when unavailable.

    Uses :func:`structlog.wrap_logger` rather than :func:`structlog.configure`:
    configuration is bound to this one logger instead of mutating structlog's
    process-global state. That matters because these modules are imported by the
    test suite, which pins its own structlog configuration.
    """
    try:
        import structlog
    except ImportError:
        return None

    level = 10 if debug_enabled() else 20
    logger: HookLogger = structlog.wrap_logger(
        # PrintLogger(file=sys.stderr) is the load-bearing part: structlog
        # defaults to stdout, which would corrupt the hook decision payload.
        structlog.PrintLogger(file=sys.stderr),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_name=name,
    )
    return logger


def get_logger(name: str, *, stream: Any | None = None) -> HookLogger:
    """Return a structured logger that writes to stderr.

    Args:
        name: Logger name, typically ``__name__``.
        stream: Optional explicit stream, which forces the fallback backend.
            Tests use this to capture records without touching global state.

    Returns:
        A logger exposing ``debug``/``info``/``warning``/``error``.
    """
    if stream is None:
        structlog_logger = _build_structlog_logger(name)
        if structlog_logger is not None:
            return structlog_logger
    return _FallbackLogger(name, stream=stream)
