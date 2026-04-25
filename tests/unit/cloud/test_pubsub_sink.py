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


@pytest.mark.asyncio
async def test_publish_telemetry_with_mock_publisher() -> None:
    """publish_telemetry should call _publish with correct topic and msgpack data."""
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg)
    sink._publisher = MagicMock()
    sink._telemetry_topic = "projects/test/topics/telemetry"
    sink._experience_topic = "projects/test/topics/experience"

    # Mock the circuit breaker to pass through
    async def passthrough(func: Any) -> Any:
        return await func()

    sink._cb = MagicMock()
    sink._cb.call = passthrough

    # Mock the publish future
    future = MagicMock()
    future.result.return_value = None
    sink._publisher.publish.return_value = future

    await sink.publish_telemetry({"test": "data", "loop_time_ms": 5.0})
    sink._publisher.publish.assert_called_once()
    call_args = sink._publisher.publish.call_args
    assert call_args[0][0] == "projects/test/topics/telemetry"
    assert call_args[1]["type"] == "telemetry"
    assert call_args[1]["schema_version"] == "1"
    assert call_args[1]["source_id"] == "droid-test"


@pytest.mark.asyncio
async def test_publish_experience_with_mock_publisher() -> None:
    """publish_experience should serialize record and publish."""
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg)
    sink._publisher = MagicMock()
    sink._telemetry_topic = "projects/test/topics/telemetry"
    sink._experience_topic = "projects/test/topics/experience"

    async def passthrough(func: Any) -> Any:
        return await func()

    sink._cb = MagicMock()
    sink._cb.call = passthrough

    future = MagicMock()
    future.result.return_value = None
    sink._publisher.publish.return_value = future

    record = MagicMock()
    record.serialize.return_value = b"test_data"
    record.schema_version = 1

    await sink.publish_experience(record)
    sink._publisher.publish.assert_called_once()
    call_args = sink._publisher.publish.call_args
    assert call_args[0][0] == "projects/test/topics/experience"
    assert call_args[1]["type"] == "experience"


@pytest.mark.asyncio
async def test_publish_circuit_open_silently_drops() -> None:
    """When circuit breaker is open, publish should silently drop messages."""
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink
    from mousedroid.resilience.circuit_breaker import CircuitOpenError

    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg)
    sink._publisher = MagicMock()
    sink._telemetry_topic = "projects/test/topics/telemetry"

    async def raise_circuit_open(func: Any) -> Any:
        raise CircuitOpenError("gcp_pubsub", 30.0)

    sink._cb = MagicMock()
    sink._cb.call = raise_circuit_open

    # Should not raise
    await sink.publish_telemetry({"test": "data"})


@pytest.mark.asyncio
async def test_publish_generic_exception_caught() -> None:
    """Generic exceptions during publish should be caught and logged."""
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg)
    sink._publisher = MagicMock()
    sink._telemetry_topic = "projects/test/topics/telemetry"

    async def raise_error(func: Any) -> Any:
        raise ConnectionError("network failure")

    sink._cb = MagicMock()
    sink._cb.call = raise_error

    # Should not raise
    await sink.publish_telemetry({"test": "data"})


@pytest.mark.asyncio
async def test_close_with_publisher() -> None:
    """close() should stop the publisher via executor."""
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg)
    mock_publisher = MagicMock()
    sink._publisher = mock_publisher

    await sink.close()
    assert sink._publisher is None
    mock_publisher.stop.assert_called_once()


@pytest.mark.asyncio
async def test_flush_with_publisher() -> None:
    """flush() should log when publisher is present."""
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg)
    sink._publisher = MagicMock()
    # Should not raise, just logs debug
    await sink.flush()


@pytest.mark.asyncio
async def test_start_initialises_publisher() -> None:
    """start() should create a publisher client and topic paths."""
    import sys
    from unittest.mock import patch

    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg)

    mock_publisher_cls = MagicMock()
    mock_publisher_instance = MagicMock()
    mock_publisher_instance.topic_path.side_effect = [
        "projects/test-project/topics/mousedroid-telemetry",
        "projects/test-project/topics/mousedroid-experience",
    ]
    mock_publisher_cls.return_value = mock_publisher_instance
    mock_batch_settings = MagicMock()

    mock_pubsub_module = MagicMock()
    mock_pubsub_module.PublisherClient = mock_publisher_cls
    mock_types_module = MagicMock()
    mock_types_module.BatchSettings = mock_batch_settings

    with (
        patch.dict(
            sys.modules,
            {
                "google.cloud.pubsub_v1": mock_pubsub_module,
                "google.cloud.pubsub_v1.types": mock_types_module,
            },
        ),
        patch("mousedroid.cloud.pubsub_sink.resolve_credentials") as mock_creds,
    ):
        mock_creds.return_value = (MagicMock(), "test-project")
        await sink.start()

    assert sink._publisher is mock_publisher_instance
    assert sink._telemetry_topic == "projects/test-project/topics/mousedroid-telemetry"
    assert sink._experience_topic == "projects/test-project/topics/mousedroid-experience"


def test_config_values_used() -> None:
    """CloudTelemetrySink should read all config values, not hardcode them."""
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink
    from mousedroid.config.schema import GCPPubSubConfig

    cfg = _make_gcp_cfg(
        pubsub=GCPPubSubConfig(
            telemetry_topic="custom-telemetry",
            experience_topic="custom-experience",
            publish_timeout_s=30.0,
        ),
    )
    sink = CloudTelemetrySink(cfg)
    assert sink._pubsub_cfg.telemetry_topic == "custom-telemetry"
    assert sink._pubsub_cfg.experience_topic == "custom-experience"
    assert sink._pubsub_cfg.publish_timeout_s == 30.0


# ---------------------------------------------------------------------------
# Metrics + breaker callback wiring
# ---------------------------------------------------------------------------


def _make_registry() -> Any:
    from mousedroid.config.schema import MetricsConfig
    from mousedroid.telemetry.metrics import MetricsRegistry

    return MetricsRegistry(MetricsConfig())


def test_metrics_registered_seeds_circuit_state_closed() -> None:
    """When a registry is passed, gauge must be seeded at CLOSED (=0)."""
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    registry = _make_registry()
    cfg = _make_gcp_cfg()
    CloudTelemetrySink(cfg, metrics=registry)
    text = registry.render_prometheus()
    assert 'mousedroid_cloud_circuit_state{breaker="cloud_pubsub"} 0' in text


@pytest.mark.asyncio
async def test_publish_telemetry_success_increments_success_counter() -> None:
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    registry = _make_registry()
    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg, metrics=registry)
    sink._publisher = MagicMock()
    sink._telemetry_topic = "projects/p/topics/t"

    async def passthrough(func: Any) -> Any:
        return await func()

    sink._cb = MagicMock()
    sink._cb.call = passthrough

    future = MagicMock()
    future.result.return_value = None
    sink._publisher.publish.return_value = future

    await sink.publish_telemetry({"k": 1})

    text = registry.render_prometheus()
    assert 'mousedroid_cloud_telemetry_publish_total{result="success"} 1' in text
    assert "mousedroid_cloud_telemetry_publish_latency_ms_count 1" in text


@pytest.mark.asyncio
async def test_publish_experience_records_error_label_on_failure() -> None:
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink

    registry = _make_registry()
    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg, metrics=registry)
    sink._publisher = MagicMock()
    sink._experience_topic = "projects/p/topics/e"

    async def raise_error(func: Any) -> Any:
        raise ConnectionError("down")

    sink._cb = MagicMock()
    sink._cb.call = raise_error

    record = MagicMock()
    record.serialize.return_value = b"x"
    record.schema_version = 1
    await sink.publish_experience(record)

    text = registry.render_prometheus()
    assert 'mousedroid_cloud_experience_publish_total{result="error"} 1' in text


@pytest.mark.asyncio
async def test_publish_circuit_open_records_circuit_open_label() -> None:
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink
    from mousedroid.resilience.circuit_breaker import CircuitOpenError

    registry = _make_registry()
    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg, metrics=registry)
    sink._publisher = MagicMock()
    sink._telemetry_topic = "projects/p/topics/t"

    async def raise_open(func: Any) -> Any:
        raise CircuitOpenError("cloud_pubsub", 30.0)

    sink._cb = MagicMock()
    sink._cb.call = raise_open

    await sink.publish_telemetry({"k": 1})

    text = registry.render_prometheus()
    assert 'mousedroid_cloud_telemetry_publish_total{result="circuit_open"} 1' in text


def test_breaker_state_change_updates_registry() -> None:
    """Firing the breaker's on_state_change callback must update the gauge."""
    from mousedroid.cloud.pubsub_sink import CloudTelemetrySink
    from mousedroid.resilience.circuit_breaker import CircuitState

    registry = _make_registry()
    cfg = _make_gcp_cfg()
    sink = CloudTelemetrySink(cfg, metrics=registry)

    # Simulate the breaker observing a transition to OPEN
    sink._on_breaker_state_change("cloud_pubsub", CircuitState.CLOSED, CircuitState.OPEN)
    text = registry.render_prometheus()
    assert 'mousedroid_cloud_circuit_state{breaker="cloud_pubsub"} 2' in text
