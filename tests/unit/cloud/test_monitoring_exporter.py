"""Unit tests for CloudMetricsExporter."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema import (
    CircuitBreakerConfig,
    GCPConfig,
    GCPMonitoringConfig,
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


def test_exporter_init() -> None:
    """CloudMetricsExporter should be constructable without starting."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg()
    registry = MagicMock()
    exporter = CloudMetricsExporter(cfg, registry)
    assert exporter._client is None
    assert exporter._running is False


def test_exporter_conforms_to_protocol() -> None:
    """CloudMetricsExporter should satisfy CloudMetricsExporterProtocol."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter
    from mousedroid.cloud.protocol import CloudMetricsExporterProtocol

    cfg = _make_gcp_cfg()
    registry = MagicMock()
    exporter = CloudMetricsExporter(cfg, registry)
    assert isinstance(exporter, CloudMetricsExporterProtocol)


def test_exporter_config_metric_prefix() -> None:
    """Exporter should use configured metric prefix."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg(
        monitoring=GCPMonitoringConfig(
            metric_prefix="custom.googleapis.com/test",
        ),
    )
    registry = MagicMock()
    exporter = CloudMetricsExporter(cfg, registry)
    assert exporter._prefix == "custom.googleapis.com/test"


def test_exporter_interval_from_config() -> None:
    """Exporter should use configured export interval."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg(
        monitoring=GCPMonitoringConfig(export_interval_s=30.0),
    )
    registry = MagicMock()
    exporter = CloudMetricsExporter(cfg, registry)
    assert exporter._interval_s == 30.0


@pytest.mark.asyncio
async def test_export_once_noop_before_start() -> None:
    """export_once should be a no-op before start() is called."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg()
    registry = MagicMock()
    exporter = CloudMetricsExporter(cfg, registry)
    # Should not raise
    await exporter.export_once()


@pytest.mark.asyncio
async def test_stop_noop_before_start() -> None:
    """stop() should be safe to call before start()."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg()
    registry = MagicMock()
    exporter = CloudMetricsExporter(cfg, registry)
    await exporter.stop()
