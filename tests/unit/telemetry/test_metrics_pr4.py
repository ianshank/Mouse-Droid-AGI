"""Tests for the new PR #4 metric setters on :class:`MetricsRegistry`."""

from __future__ import annotations

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import MetricsRegistry


def _registry() -> MetricsRegistry:
    return MetricsRegistry(MetricsConfig())


class TestSensorLivenessGauge:
    """``set_sensor_liveness`` sets exactly one state to 1.0 per sensor."""

    def test_single_sensor_mutual_exclusion(self) -> None:
        reg = _registry()
        reg.set_sensor_liveness({"lidar": "live"})
        snap = reg._sensor_liveness.snapshot()  # type: ignore[attr-defined]
        # Every known state must exist; only "live" is 1.0.
        assert snap[("lidar", "live")] == 1.0
        assert snap[("lidar", "stale")] == 0.0
        assert snap[("lidar", "awaiting")] == 0.0
        assert snap[("lidar", "disabled")] == 0.0

    def test_render_includes_double_label(self) -> None:
        reg = _registry()
        reg.set_sensor_liveness({"lidar": "stale", "vision": "live"})
        out = reg.render_prometheus()
        assert 'mousedroid_telemetry_sensor_liveness{sensor="lidar",state="stale"} 1' in out
        assert 'mousedroid_telemetry_sensor_liveness{sensor="vision",state="live"} 1' in out

    def test_swapping_state_zeroes_previous(self) -> None:
        reg = _registry()
        reg.set_sensor_liveness({"lidar": "live"})
        reg.set_sensor_liveness({"lidar": "stale"})
        snap = reg._sensor_liveness.snapshot()  # type: ignore[attr-defined]
        assert snap[("lidar", "live")] == 0.0
        assert snap[("lidar", "stale")] == 1.0


class TestMdnsAndBoundPort:
    """Single-label gauges expose mDNS state and bound port correctly."""

    def test_set_mdns_registered_ok(self) -> None:
        reg = _registry()
        reg.set_mdns_registered("MouseDroid Telemetry", ok=True)
        out = reg.render_prometheus()
        assert 'mousedroid_telemetry_mdns_registered{service="MouseDroid Telemetry"} 1' in out

    def test_set_mdns_registered_failed(self) -> None:
        reg = _registry()
        reg.set_mdns_registered("svc", ok=False)
        out = reg.render_prometheus()
        assert 'mousedroid_telemetry_mdns_registered{service="svc"} 0' in out

    def test_set_bound_port_renders_when_positive(self) -> None:
        reg = _registry()
        reg.set_bound_port(8082)
        out = reg.render_prometheus()
        assert "mousedroid_telemetry_bound_port 8082" in out

    def test_bound_port_omitted_when_zero(self) -> None:
        reg = _registry()
        out = reg.render_prometheus()
        assert "telemetry_bound_port" not in out


class TestLidarRawCounters:
    """Lidar raw publish/drop counters emit when non-zero."""

    def test_published_counter(self) -> None:
        reg = _registry()
        reg.inc_lidar_raw_published(3)
        out = reg.render_prometheus()
        assert "mousedroid_telemetry_lidar_raw_published_total 3" in out

    def test_dropped_counter(self) -> None:
        reg = _registry()
        reg.inc_lidar_raw_dropped(2)
        out = reg.render_prometheus()
        assert "mousedroid_telemetry_lidar_raw_dropped_total 2" in out

    def test_counters_omitted_when_zero(self) -> None:
        reg = _registry()
        out = reg.render_prometheus()
        assert "telemetry_lidar_raw_published_total" not in out
