"""Cross-cutting subsystem failure recorder.

Provides a ``FailureRecorder`` protocol that any subsystem can inject to
convert silent fallbacks into observable signals — structured log event plus
a Prometheus counter increment — without coupling callers to the telemetry
stack.

Usage::

    recorder = build_failure_recorder(cfg, metrics_registry)

    recorder.record(
        subsystem="voice",
        reason="device_disconnected",
        level="error",
        extra={"attempt": 3},
    )

This emits a ``subsystem_failure_recorded`` structlog event and increments
``mousedroid_subsystem_failures_total{subsystem="voice",reason="device_disconnected",level="error"}``.

When telemetry is disabled (or ``metrics_registry`` is ``None``), a
``NullFailureRecorder`` is returned and all calls are no-ops.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Literal

from typing_extensions import Protocol, runtime_checkable

from mousedroid.logging.setup import get_logger, safe_log_extra

if TYPE_CHECKING:
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)

SeverityLevel = Literal["warning", "error", "critical"]


@runtime_checkable
class FailureRecorder(Protocol):
    """Structural protocol for recording cross-cutting subsystem failure events.

    Any object with a matching ``record()`` signature satisfies this protocol —
    no explicit inheritance needed.  Implementations emit a structured log entry
    and update a Prometheus counter so operators can observe failure rates without
    reading logs.
    """

    def record(
        self,
        subsystem: str,
        reason: str,
        *,
        level: SeverityLevel = "warning",
        extra: Mapping[str, str | int | float] | None = None,
    ) -> None:
        """Record one failure event.

        Args:
            subsystem: Logical subsystem name (e.g. ``"voice"``, ``"telemetry"``).
                Use snake_case; bounded cardinality — no dynamic values.
            reason: Machine-readable failure reason (e.g. ``"device_disconnected"``).
                Use snake_case; bounded cardinality — no dynamic values.
            level: Severity level — ``"warning"``, ``"error"``, or ``"critical"``.
            extra: Optional mapping of additional structured key-value pairs to
                include in the log event. Values must be str, int, or float.
        """
        ...


class PrometheusFailureRecorder:
    """Failure recorder that increments a Prometheus counter and emits a structlog event.

    Args:
        metrics: ``MetricsRegistry`` instance to increment the
            ``mousedroid_subsystem_failures_total`` counter on.
    """

    # Module-level set so each call avoids rebuilding a dispatch table
    # (addresses PR #78 Gemini medium review). Using ``getattr(_log,
    # level)`` keeps dispatch dynamic so structlog test fixtures that
    # patch ``_log`` see the patched methods — a class-level mapping of
    # the bound methods captures them at import time and breaks test
    # capture.
    _ALLOWED_LEVELS: ClassVar[frozenset[str]] = frozenset({"warning", "error", "critical"})

    def __init__(self, metrics: MetricsRegistry) -> None:
        self._metrics = metrics

    def record(
        self,
        subsystem: str,
        reason: str,
        *,
        level: SeverityLevel = "warning",
        extra: Mapping[str, str | int | float] | None = None,
    ) -> None:
        """Record a failure: increment counter + emit structured log.

        Args:
            subsystem: Logical subsystem name.
            reason: Machine-readable failure reason.
            level: Severity level.
            extra: Optional additional structured log fields.
        """
        self._metrics.inc_subsystem_failure(subsystem, reason, level)

        log_kv: dict[str, object] = {
            "subsystem": subsystem,
            "reason": reason,
            "log_level": level,
        }
        # ``extra`` is caller-controlled, so it can carry a key that structlog
        # owns.  ``event`` is the worst case: the bound-logger signature is
        # ``meth(event, **kw)``, so splatting a caller ``event`` raised
        # TypeError and lost the whole log line (the metric above had already
        # been incremented, so failures went silently uncounted in the logs).
        # ``safe_log_extra`` namespaces those keys instead of dropping them,
        # and ``occupied`` additionally protects this recorder's own fields.
        if extra:
            log_kv.update(safe_log_extra(extra, occupied=log_kv))

        # ``getattr`` is O(1) on a Python object and produces no dict
        # allocation; the fallback to ``_log.warning`` happens only on
        # an unknown level (already constrained by ``SeverityLevel``).
        log_fn = getattr(_log, level) if level in self._ALLOWED_LEVELS else _log.warning
        log_fn("subsystem_failure_recorded", **log_kv)


class NullFailureRecorder:
    """No-op failure recorder used when telemetry is disabled or in unit tests."""

    def record(
        self,
        subsystem: str,
        reason: str,
        *,
        level: SeverityLevel = "warning",
        extra: Mapping[str, str | int | float] | None = None,
    ) -> None:
        """No-op implementation — discards all arguments silently."""
