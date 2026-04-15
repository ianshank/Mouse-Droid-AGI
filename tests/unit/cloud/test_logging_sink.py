"""Unit tests for CloudLoggingSink structlog processor."""

from __future__ import annotations

from typing import Any

from mousedroid.config.schema import (
    CircuitBreakerConfig,
    GCPConfig,
    GCPLoggingConfig,
    RetryConfig,
)


def _make_gcp_cfg(**overrides: Any) -> GCPConfig:
    """Create a GCPConfig with test defaults."""
    return GCPConfig(
        project_id="test-project",
        robot_id="droid-test",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_s=1.0,
            half_open_max_calls=1,
        ),
        retry=RetryConfig(max_attempts=1, base_delay_s=0.01),
        **overrides,
    )


def test_logging_sink_init() -> None:
    """CloudLoggingSink should be constructable without starting."""
    from mousedroid.cloud.logging_sink import CloudLoggingSink

    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    assert sink._started is False


def test_logging_sink_passthrough_before_start() -> None:
    """Sink should pass through events unchanged before start()."""
    from mousedroid.cloud.logging_sink import CloudLoggingSink

    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)

    event_dict = {"event": "test_event", "key": "value"}
    result = sink(None, "info", event_dict)
    assert result is event_dict
    assert result["event"] == "test_event"


def test_logging_sink_min_level_filtering() -> None:
    """Sink should respect min_level configuration."""
    from mousedroid.cloud.logging_sink import CloudLoggingSink

    cfg = _make_gcp_cfg(
        logging=GCPLoggingConfig(min_level="WARNING"),
    )
    sink = CloudLoggingSink(cfg)

    # Debug events should be below min_level threshold
    event_dict = {"event": "debug_event"}
    result = sink(None, "debug", event_dict)
    assert result is event_dict


def test_logging_sink_conforms_to_protocol() -> None:
    """CloudLoggingSink should satisfy CloudLoggingSinkProtocol."""
    from mousedroid.cloud.logging_sink import CloudLoggingSink
    from mousedroid.cloud.protocol import CloudLoggingSinkProtocol

    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)
    assert isinstance(sink, CloudLoggingSinkProtocol)


def test_logging_sink_preserves_event_dict() -> None:
    """Sink should never modify the event dict passed to it."""
    from mousedroid.cloud.logging_sink import CloudLoggingSink

    cfg = _make_gcp_cfg()
    sink = CloudLoggingSink(cfg)

    original = {"event": "test", "extra": 42, "nested": {"a": 1}}
    result = sink(None, "info", original)
    assert result["event"] == "test"
    assert result["extra"] == 42
    assert result["nested"]["a"] == 1
