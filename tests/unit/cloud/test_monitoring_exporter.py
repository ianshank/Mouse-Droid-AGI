"""Unit tests for CloudMetricsExporter."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

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


def test_parse_gauge_metrics_basic() -> None:
    """_parse_gauge_metrics should extract gauge values from Prometheus text."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg()
    registry = MagicMock()
    registry.render_prometheus.return_value = (
        "# HELP mousedroid_loop_time_ms Control loop time\n"
        "# TYPE mousedroid_loop_time_ms gauge\n"
        "mousedroid_loop_time_ms 5.2\n"
        "\n"
        "# HELP mousedroid_battery_voltage_v Battery voltage\n"
        "# TYPE mousedroid_battery_voltage_v gauge\n"
        "mousedroid_battery_voltage_v 11.8\n"
    )
    exporter = CloudMetricsExporter(cfg, registry)
    metrics = exporter._parse_gauge_metrics()
    assert metrics["mousedroid_loop_time_ms"] == pytest.approx(5.2)
    assert metrics["mousedroid_battery_voltage_v"] == pytest.approx(11.8)


def test_parse_gauge_metrics_skips_counters() -> None:
    """_parse_gauge_metrics should skip counter-type metrics."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg()
    registry = MagicMock()
    registry.render_prometheus.return_value = (
        "# HELP mousedroid_frame_drops_total Frame drops\n"
        "# TYPE mousedroid_frame_drops_total counter\n"
        "mousedroid_frame_drops_total 42\n"
        "\n"
        "# HELP mousedroid_gpu_temp_celsius GPU temp\n"
        "# TYPE mousedroid_gpu_temp_celsius gauge\n"
        "mousedroid_gpu_temp_celsius 65.3\n"
    )
    exporter = CloudMetricsExporter(cfg, registry)
    metrics = exporter._parse_gauge_metrics()
    assert "mousedroid_frame_drops_total" not in metrics
    assert metrics["mousedroid_gpu_temp_celsius"] == pytest.approx(65.3)


def test_parse_gauge_metrics_empty_output() -> None:
    """_parse_gauge_metrics should return empty dict for empty Prometheus output."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg()
    registry = MagicMock()
    registry.render_prometheus.return_value = ""
    exporter = CloudMetricsExporter(cfg, registry)
    metrics = exporter._parse_gauge_metrics()
    assert metrics == {}


def test_parse_gauge_metrics_malformed_lines() -> None:
    """_parse_gauge_metrics should skip lines with non-numeric values."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg()
    registry = MagicMock()
    registry.render_prometheus.return_value = (
        "# TYPE mousedroid_test gauge\nmousedroid_test not_a_number\nmousedroid_valid 1.5\n"
    )
    exporter = CloudMetricsExporter(cfg, registry)
    metrics = exporter._parse_gauge_metrics()
    assert "mousedroid_test" not in metrics
    assert metrics["mousedroid_valid"] == pytest.approx(1.5)


def test_parse_gauge_metrics_skips_histograms() -> None:
    """_parse_gauge_metrics should skip histogram-type metrics."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg()
    registry = MagicMock()
    registry.render_prometheus.return_value = (
        "# TYPE mousedroid_loop_latency_ms histogram\n"
        'mousedroid_loop_latency_ms_bucket{le="1.0"} 5\n'
        "mousedroid_loop_latency_ms_sum 10.5\n"
        "mousedroid_loop_latency_ms_count 5\n"
    )
    exporter = CloudMetricsExporter(cfg, registry)
    metrics = exporter._parse_gauge_metrics()
    assert len(metrics) == 0


@pytest.mark.asyncio
async def test_export_once_with_metrics() -> None:
    """export_once should call _write_metrics when metrics exist."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg()
    registry = MagicMock()
    registry.render_prometheus.return_value = "# TYPE test_gauge gauge\ntest_gauge 42.0\n"
    exporter = CloudMetricsExporter(cfg, registry)
    exporter._client = MagicMock()

    # Mock the _write_metrics since it imports google.cloud
    with patch.object(exporter, "_write_metrics") as mock_write:
        await exporter.export_once()
        mock_write.assert_called_once()
        args = mock_write.call_args[0]
        assert "test_gauge" in args[0]


@pytest.mark.asyncio
async def test_export_once_no_metrics_skips_write() -> None:
    """export_once should not write when no gauge metrics found."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg()
    registry = MagicMock()
    registry.render_prometheus.return_value = ""
    exporter = CloudMetricsExporter(cfg, registry)
    exporter._client = MagicMock()

    with patch.object(exporter, "_write_metrics") as mock_write:
        await exporter.export_once()
        mock_write.assert_not_called()


@pytest.mark.asyncio
async def test_export_once_handles_write_error() -> None:
    """export_once should catch and log errors from _write_metrics."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg()
    registry = MagicMock()
    registry.render_prometheus.return_value = "# TYPE test gauge\ntest 1.0\n"
    exporter = CloudMetricsExporter(cfg, registry)
    exporter._client = MagicMock()

    with patch.object(exporter, "_write_metrics", side_effect=RuntimeError("boom")):
        # Should not raise
        await exporter.export_once()


@pytest.mark.asyncio
async def test_export_loop_runs_and_stops() -> None:
    """_export_loop should periodically call export_once and stop on cancel."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg(monitoring=GCPMonitoringConfig(export_interval_s=0.01))
    registry = MagicMock()
    registry.render_prometheus.return_value = ""
    exporter = CloudMetricsExporter(cfg, registry)
    exporter._running = True

    call_count = 0

    async def counting_export() -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            exporter._running = False

    exporter.export_once = counting_export
    await exporter._export_loop()
    assert call_count >= 2


@pytest.mark.asyncio
async def test_export_loop_handles_exceptions() -> None:
    """_export_loop should catch and continue on exceptions."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg(monitoring=GCPMonitoringConfig(export_interval_s=0.01))
    registry = MagicMock()
    exporter = CloudMetricsExporter(cfg, registry)
    exporter._running = True

    call_count = 0

    async def failing_export() -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            exporter._running = False
            return
        raise RuntimeError("test error")

    exporter.export_once = failing_export
    await exporter._export_loop()
    assert call_count >= 2


@pytest.mark.asyncio
async def test_start_initialises_client() -> None:
    """start() should create a Cloud Monitoring client and start loop."""
    import sys

    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg()
    registry = MagicMock()
    registry.render_prometheus.return_value = ""
    exporter = CloudMetricsExporter(cfg, registry)

    mock_client_cls = MagicMock()
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_monitoring_module = MagicMock()
    mock_monitoring_module.MetricServiceClient = mock_client_cls

    with (
        patch.dict(
            sys.modules,
            {
                "google": MagicMock(),
                "google.cloud": MagicMock(),
                "google.cloud.monitoring_v3": mock_monitoring_module,
            },
        ),
        patch("mousedroid.cloud._auth.resolve_credentials") as mock_creds,
    ):
        mock_creds.return_value = (MagicMock(), "test-project")
        await exporter.start()

    assert exporter._client is not None
    assert exporter._running is True
    assert exporter._task is not None
    assert exporter._task in exporter._background_tasks
    # Clean up
    await exporter.stop()
    assert not exporter._background_tasks


@pytest.mark.asyncio
async def test_stop_cancels_task() -> None:
    """stop() should cancel the background task."""
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = _make_gcp_cfg()
    registry = MagicMock()
    exporter = CloudMetricsExporter(cfg, registry)
    exporter._running = True
    exporter._client = MagicMock()

    async def fake_loop() -> None:
        await asyncio.sleep(100)

    exporter._task = asyncio.create_task(fake_loop())
    await exporter.stop()
    assert exporter._task is None
    assert exporter._client is None
