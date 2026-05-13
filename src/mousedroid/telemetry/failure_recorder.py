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
from typing import TYPE_CHECKING, Literal

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)

SeverityLevel = Literal["warning", "error", "critical"]


class FailureRecorder:
    """Protocol for recording cross-cutting subsystem failure events.

    Implementations emit a structured log entry and update a Prometheus
    counter so operators can observe failure rates without reading logs.
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


class PrometheusFailureRecorder(FailureRecorder):
    """Failure recorder that increments a Prometheus counter and emits a structlog event.

    Args:
        metrics: ``MetricsRegistry`` instance to increment the
            ``mousedroid_subsystem_failures_total`` counter on.
    """

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
        if extra:
            for k, v in extra.items():
                log_kv[k] = v

        log_fn = {
            "warning": _log.warning,
            "error": _log.error,
            "critical": _log.critical,
        }.get(level, _log.warning)
        log_fn("subsystem_failure_recorded", **log_kv)


class NullFailureRecorder(FailureRecorder):
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
