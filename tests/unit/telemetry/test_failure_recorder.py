"""Unit tests for FailureRecorder protocol and implementations."""

from __future__ import annotations

import structlog.testing

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.failure_recorder import (
    FailureRecorder,
    NullFailureRecorder,
    PrometheusFailureRecorder,
)
from mousedroid.telemetry.metrics import MetricsRegistry


def _registry() -> MetricsRegistry:
    return MetricsRegistry(MetricsConfig())


class TestProtocolConformance:
    """Both implementations must conform to the FailureRecorder interface."""

    def test_prometheus_recorder_is_failure_recorder(self) -> None:
        """PrometheusFailureRecorder is a subclass of FailureRecorder."""
        assert issubclass(PrometheusFailureRecorder, FailureRecorder)

    def test_null_recorder_is_failure_recorder(self) -> None:
        """NullFailureRecorder is a subclass of FailureRecorder."""
        assert issubclass(NullFailureRecorder, FailureRecorder)

    def test_prometheus_recorder_has_record_method(self) -> None:
        """PrometheusFailureRecorder exposes .record()."""
        rec = PrometheusFailureRecorder(_registry())
        assert callable(rec.record)

    def test_null_recorder_has_record_method(self) -> None:
        """NullFailureRecorder exposes .record()."""
        rec = NullFailureRecorder()
        assert callable(rec.record)


class TestPrometheusFailureRecorder:
    """PrometheusFailureRecorder increments the metric and emits a log."""

    def test_increments_counter_on_record(self) -> None:
        """record() increments mousedroid_subsystem_failures_total."""
        registry = _registry()
        rec = PrometheusFailureRecorder(registry)

        rec.record("voice", "device_disconnected", level="error")

        snapshot = registry._subsystem_failures.snapshot()
        assert snapshot.get(("voice", "device_disconnected", "error")) == 1

    def test_default_level_is_warning(self) -> None:
        """record() defaults to level='warning' when omitted."""
        registry = _registry()
        rec = PrometheusFailureRecorder(registry)

        rec.record("telemetry", "bind_exhausted")

        snapshot = registry._subsystem_failures.snapshot()
        assert snapshot.get(("telemetry", "bind_exhausted", "warning")) == 1

    def test_multiple_records_accumulate(self) -> None:
        """Repeated calls to record() accumulate counter values."""
        registry = _registry()
        rec = PrometheusFailureRecorder(registry)

        for _ in range(5):
            rec.record("orchestrator", "latent_nan", level="critical")

        snapshot = registry._subsystem_failures.snapshot()
        assert snapshot.get(("orchestrator", "latent_nan", "critical")) == 5

    def test_different_labels_tracked_independently(self) -> None:
        """Different (subsystem, reason, level) triples are independent."""
        registry = _registry()
        rec = PrometheusFailureRecorder(registry)

        rec.record("voice", "device_disconnected", level="error")
        rec.record("voice", "device_disconnected", level="warning")
        rec.record("telemetry", "bind_exhausted", level="warning")

        snapshot = registry._subsystem_failures.snapshot()
        assert snapshot.get(("voice", "device_disconnected", "error")) == 1
        assert snapshot.get(("voice", "device_disconnected", "warning")) == 1
        assert snapshot.get(("telemetry", "bind_exhausted", "warning")) == 1

    def test_emits_structlog_warning_event(self) -> None:
        """record() emits subsystem_failure_recorded at warning level."""
        rec = PrometheusFailureRecorder(_registry())

        with structlog.testing.capture_logs() as logs:
            rec.record("voice", "piper_timeout", level="warning")

        assert len(logs) == 1
        assert logs[0]["event"] == "subsystem_failure_recorded"
        assert logs[0]["log_level"] == "warning"
        assert logs[0]["subsystem"] == "voice"
        assert logs[0]["reason"] == "piper_timeout"

    def test_emits_structlog_error_event(self) -> None:
        """record() emits at error level when level='error'."""
        rec = PrometheusFailureRecorder(_registry())

        with structlog.testing.capture_logs() as logs:
            rec.record("telemetry", "bind_exhausted", level="error")

        assert logs[0]["log_level"] == "error"

    def test_emits_structlog_critical_event(self) -> None:
        """record() emits at critical level when level='critical'."""
        rec = PrometheusFailureRecorder(_registry())

        with structlog.testing.capture_logs() as logs:
            rec.record("world_model", "latent_nan", level="critical")

        assert logs[0]["log_level"] == "critical"

    def test_extra_fields_included_in_log(self) -> None:
        """extra kwarg is included as structured fields in the log event."""
        rec = PrometheusFailureRecorder(_registry())

        with structlog.testing.capture_logs() as logs:
            rec.record("voice", "retry_exhausted", extra={"attempt": 3, "device": "USB Audio"})

        log = logs[0]
        assert log["attempt"] == 3
        assert log["device"] == "USB Audio"

    def test_extra_none_does_not_crash(self) -> None:
        """record() with extra=None is safe."""
        rec = PrometheusFailureRecorder(_registry())

        with structlog.testing.capture_logs():
            rec.record("voice", "retry_exhausted", extra=None)

    def test_counter_appears_in_prometheus_output(self) -> None:
        """After recording, the counter appears in render_prometheus() output."""
        registry = _registry()
        rec = PrometheusFailureRecorder(registry)
        rec.record("voice", "device_disconnected", level="error")

        output = registry.render_prometheus()

        assert "mousedroid_subsystem_failures_total" in output
        assert 'subsystem="voice"' in output
        assert 'reason="device_disconnected"' in output
        assert 'level="error"' in output


class TestNullFailureRecorder:
    """NullFailureRecorder is a silent no-op."""

    def test_record_does_not_raise(self) -> None:
        """record() never raises under any inputs."""
        rec = NullFailureRecorder()
        rec.record("voice", "any_reason", level="critical", extra={"k": 1})

    def test_record_emits_no_logs(self) -> None:
        """record() emits no log events."""
        rec = NullFailureRecorder()

        with structlog.testing.capture_logs() as logs:
            rec.record("voice", "any_reason")

        assert len(logs) == 0


class TestBuildFailureRecorder:
    """build_failure_recorder factory selects the correct implementation."""

    def test_returns_prometheus_recorder_when_metrics_present(self) -> None:
        """build_failure_recorder returns PrometheusFailureRecorder when metrics given."""
        from mousedroid.config.schema import Settings
        from mousedroid.factory import build_failure_recorder

        cfg = Settings()  # type: ignore[call-arg]
        registry = _registry()

        rec = build_failure_recorder(cfg, metrics=registry)

        assert isinstance(rec, PrometheusFailureRecorder)

    def test_returns_null_recorder_when_metrics_none(self) -> None:
        """build_failure_recorder returns NullFailureRecorder when metrics=None."""
        from mousedroid.config.schema import Settings
        from mousedroid.factory import build_failure_recorder

        cfg = Settings()  # type: ignore[call-arg]

        rec = build_failure_recorder(cfg, metrics=None)

        assert isinstance(rec, NullFailureRecorder)
