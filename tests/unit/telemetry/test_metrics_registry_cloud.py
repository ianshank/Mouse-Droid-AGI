"""Tests for cloud digital twin metrics exposed by :class:`MetricsRegistry`.

These tests guarantee the namespace-derived naming pattern holds for
every new cloud metric, that toggle flags disable emission, and that
the breaker-state encoding matches the Grafana dashboard contract.
"""

from __future__ import annotations

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import MetricsRegistry


def _make() -> MetricsRegistry:
    return MetricsRegistry(MetricsConfig())


def test_cloud_publish_counters_use_namespace() -> None:
    reg = _make()
    reg.inc_cloud_telemetry_publish("success", 2)
    reg.inc_cloud_experience_publish("circuit_open")
    reg.observe_cloud_telemetry_publish_latency_ms(12.5)
    reg.observe_cloud_experience_publish_latency_ms(8.0)

    text = reg.render_prometheus()

    # Namespace prefix applied to every new cloud family
    assert "mousedroid_cloud_telemetry_publish_total" in text
    assert "mousedroid_cloud_experience_publish_total" in text
    assert "mousedroid_cloud_telemetry_publish_latency_ms" in text
    assert "mousedroid_cloud_experience_publish_latency_ms" in text

    # Label-values rendered verbatim
    assert 'result="success"' in text
    assert 'result="circuit_open"' in text


def test_cloud_circuit_state_encodes_open_half_open_closed() -> None:
    reg = _make()
    reg.set_cloud_circuit_state("cloud_pubsub", "closed")
    reg.set_cloud_circuit_state("cloud_experience", "open")
    reg.set_cloud_circuit_state("cloud_other", "half_open")

    text = reg.render_prometheus()

    assert 'mousedroid_cloud_circuit_state{breaker="cloud_pubsub"} 0' in text
    assert 'mousedroid_cloud_circuit_state{breaker="cloud_experience"} 2' in text
    assert 'mousedroid_cloud_circuit_state{breaker="cloud_other"} 1' in text


def test_cloud_experience_export_and_lag_gauges() -> None:
    reg = _make()
    reg.inc_cloud_experience_export_records("success", 10)
    reg.inc_cloud_experience_export_records("error", 2)
    reg.set_cloud_experience_hwm_lag(500)
    reg.set_cloud_experience_queue_depth(17)

    text = reg.render_prometheus()

    assert 'mousedroid_cloud_experience_export_records_total{result="success"} 10' in text
    assert 'mousedroid_cloud_experience_export_records_total{result="error"} 2' in text
    assert "mousedroid_cloud_experience_hwm_lag 500" in text
    assert "mousedroid_cloud_experience_queue_depth 17" in text


def test_cloud_disabled_suppresses_emission() -> None:
    cfg = MetricsConfig(track_cloud=False)
    reg = MetricsRegistry(cfg)
    reg.inc_cloud_telemetry_publish("success")
    reg.set_cloud_circuit_state("cloud_pubsub", "closed")
    reg.set_cloud_experience_hwm_lag(100)

    text = reg.render_prometheus()
    assert "cloud_telemetry_publish" not in text
    assert "cloud_circuit_state" not in text
    assert "cloud_experience_hwm_lag" not in text


def test_custom_namespace_propagates_to_cloud_metrics() -> None:
    reg = MetricsRegistry(MetricsConfig(namespace="swarm_alpha"))
    reg.inc_cloud_telemetry_publish("success")
    reg.set_cloud_circuit_state("cloud_pubsub", "closed")

    text = reg.render_prometheus()
    assert "swarm_alpha_cloud_telemetry_publish_total" in text
    assert "swarm_alpha_cloud_circuit_state" in text
    assert "mousedroid_cloud" not in text


def test_export_records_zero_amount_is_noop() -> None:
    reg = _make()
    reg.inc_cloud_experience_export_records("success", 0)
    text = reg.render_prometheus()
    assert "mousedroid_cloud_experience_export_records_total" not in text


def test_cloud_health_snapshot_reports_status_and_breakers() -> None:
    reg = _make()
    reg.set_cloud_circuit_state("cloud_pubsub", "open")
    reg.set_cloud_experience_queue_depth(3)
    reg.set_cloud_experience_hwm_lag(7)
    reg.inc_cloud_telemetry_publish("success", 2)

    snapshot = reg.get_cloud_health_snapshot()

    assert snapshot["enabled"] is True
    assert snapshot["status"] == "degraded"
    assert snapshot["breaker_states"] == {"cloud_pubsub": "open"}
    assert snapshot["queue_depth"] == 3
    assert snapshot["hwm_lag"] == 7


def test_cloud_health_snapshot_disabled_when_track_cloud_off() -> None:
    reg = MetricsRegistry(MetricsConfig(track_cloud=False))
    assert reg.get_cloud_health_snapshot() == {"enabled": False, "status": "disabled"}
