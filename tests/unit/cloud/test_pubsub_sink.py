"""Unit tests for CloudTelemetrySink Pub/Sub publisher."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema import (
    CircuitBreakerConfig,
    GCPConfig,
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


def test_sink_init_without_start() -> None:
    """Sink should be constructable without starting."""
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg)
    assert sink._publisher is None


@pytest.mark.asyncio
async def test_publish_telemetry_noop_before_start() -> None:
    """publish_telemetry should be a no-op before start() is called."""
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg)
    # Should not raise
    await sink.publish_telemetry({"test": "data"})


@pytest.mark.asyncio
async def test_publish_experience_noop_before_start() -> None:
    """publish_experience should be a no-op before start() is called."""
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg)

    record = MagicMock()
    record.serialize.return_value = b"test"
    record.schema_version = 1
    await sink.publish_experience(record)


@pytest.mark.asyncio
async def test_close_noop_before_start() -> None:
    """close() should be safe to call before start()."""
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg)
    await sink.close()


@pytest.mark.asyncio
async def test_flush_noop_before_start() -> None:
    """flush() should be safe to call before start()."""
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg)
    await sink.flush()


def test_sink_conforms_to_protocol() -> None:
    """CloudTelemetrySink should satisfy CloudTelemetrySinkProtocol."""
    from mousedroid.cloud.protocol import CloudTelemetrySinkProtocol
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg)
    assert isinstance(sink, CloudTelemetrySinkProtocol)
