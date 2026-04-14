"""Tests for MetricsRegistry — Prometheus text exposition format."""

from __future__ import annotations

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import (
    MetricsRegistry,
    _Counter,
    _escape_help_text,
    _escape_label_value,
    _fmt_float,
    _Gauge,
    _Histogram,
    _LabeledCounter,
)

# ---------------------------------------------------------------------------
# Primitive metric types
# ---------------------------------------------------------------------------


def test_counter_default_zero() -> None:
    c = _Counter()
    assert c.value == 0


def test_counter_inc() -> None:
    c = _Counter()
    c.inc()
    c.inc(5)
    assert c.value == 6


def test_counter_reset() -> None:
    c = _Counter()
    c.inc(10)
    c.reset()
    assert c.value == 0


def test_gauge_default_zero() -> None:
    g = _Gauge()
    assert g.value == 0.0


def test_gauge_set() -> None:
    g = _Gauge(initial=5.0)
    assert g.value == 5.0
    g.set(42.0)
    assert g.value == 42.0


def test_labeled_counter_inc_and_snapshot() -> None:
    lc = _LabeledCounter()
    lc.inc("law1")
    lc.inc("law2", 3)
    lc.inc("law1")
    snap = lc.snapshot()
    assert snap == {"law1": 2, "law2": 3}


def test_labeled_counter_reset() -> None:
    lc = _LabeledCounter()
    lc.inc("a")
    lc.reset()
    assert lc.snapshot() == {}


def test_histogram_observe_and_snapshot() -> None:
    h = _Histogram((10.0, 50.0, 100.0, float("inf")))
    h.observe(5.0)   # bucket 10
    h.observe(15.0)  # bucket 50
    h.observe(75.0)  # bucket 100
    h.observe(200.0)  # bucket inf

    buckets, total_sum, total_count = h.snapshot()
    assert total_count == 4
    assert total_sum == 5.0 + 15.0 + 75.0 + 200.0
    # Cumulative: 10→1, 50→2, 100→3, inf→4
    assert buckets == [(10.0, 1), (50.0, 2), (100.0, 3), (float("inf"), 4)]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def test_fmt_float_normal() -> None:
    assert _fmt_float(42.5) == "42.5"


def test_fmt_float_inf() -> None:
    assert _fmt_float(float("inf")) == "+Inf"


def test_fmt_float_nan() -> None:
    assert _fmt_float(float("nan")) == "NaN"


def test_escape_label_value() -> None:
    assert _escape_label_value('hello\n"world"\\') == 'hello\\n\\"world\\"\\\\'


def test_escape_help_text() -> None:
    assert _escape_help_text("line1\nline2\\end") == "line1\\nline2\\\\end"


# ---------------------------------------------------------------------------
# MetricsRegistry — write helpers
# ---------------------------------------------------------------------------


def _make_registry(**overrides: object) -> MetricsRegistry:
    """Create a MetricsRegistry with all tracking enabled."""
    cfg = MetricsConfig(**overrides)  # type: ignore[arg-type]
    return MetricsRegistry(cfg)


def test_registry_inc_frame_drops() -> None:
    reg = _make_registry()
    reg.inc_frame_drops()
    reg.inc_frame_drops(3)
    assert reg.frame_drops_total == 4


def test_registry_inc_frame_drops_disabled() -> None:
    reg = _make_registry(track_frame_drops=False)
    reg.inc_frame_drops(10)
    assert reg.frame_drops_total == 0


def test_registry_inc_safety_violation() -> None:
    reg = _make_registry()
    reg.inc_safety_violation("law1")
    reg.inc_safety_violation("law2")
    reg.inc_safety_violation("law1")
    assert reg.safety_violations == {"law1": 2, "law2": 1}


def test_registry_set_loop_time_ms() -> None:
    reg = _make_registry()
    reg.set_loop_time_ms(18.5)
    # Should also observe into histogram
    output = reg.render_prometheus()
    assert "loop_latency_ms_bucket" in output


def test_registry_set_battery_voltage() -> None:
    reg = _make_registry()
    reg.set_battery_voltage(12.6)
    output = reg.render_prometheus()
    assert "battery_voltage_v" in output
    assert "12.6" in output


def test_registry_set_ws_client_count() -> None:
    reg = _make_registry()
    reg.set_ws_client_count(3)
    output = reg.render_prometheus()
    assert "ws_client_count" in output


def test_registry_set_gpu_temp() -> None:
    reg = _make_registry()
    reg.set_gpu_temp_celsius(67.5)
    output = reg.render_prometheus()
    assert "gpu_temp_celsius" in output
    assert "67.5" in output


def test_registry_set_publish_hz() -> None:
    reg = _make_registry()
    reg.set_publish_hz(30.0)
    output = reg.render_prometheus()
    assert "publish_hz" in output


# -- Phase 7 metrics --


def test_registry_set_episodic_size() -> None:
    reg = _make_registry()
    reg.set_episodic_size(500)
    output = reg.render_prometheus()
    assert "memory_episodic_size" in output
    assert "500" in output


def test_registry_set_semantic_size() -> None:
    reg = _make_registry()
    reg.set_semantic_size(120)
    output = reg.render_prometheus()
    assert "memory_semantic_size" in output


def test_registry_set_working_size() -> None:
    reg = _make_registry()
    reg.set_working_size(64)
    output = reg.render_prometheus()
    assert "memory_working_size" in output


def test_registry_memory_tier_disabled() -> None:
    reg = _make_registry(track_memory_tier=False)
    reg.set_episodic_size(999)
    output = reg.render_prometheus()
    assert "memory_episodic_size" not in output


def test_registry_inc_voice_event() -> None:
    reg = _make_registry()
    reg.inc_voice_event("startup")
    reg.inc_voice_event("obstacle_detected")
    reg.inc_voice_event("startup")
    output = reg.render_prometheus()
    assert "voice_events" in output
    assert "startup" in output


def test_registry_voice_events_disabled() -> None:
    reg = _make_registry(track_voice_events=False)
    reg.inc_voice_event("startup")
    output = reg.render_prometheus()
    assert "voice_events" not in output


def test_registry_set_llm_latency_ms() -> None:
    reg = _make_registry()
    reg.set_llm_latency_ms(42.0)
    output = reg.render_prometheus()
    assert "llm_latency_ms" in output
    assert "llm_requests" in output


def test_registry_set_curiosity_reward() -> None:
    reg = _make_registry()
    reg.set_curiosity_reward(0.75)
    output = reg.render_prometheus()
    assert "curiosity_intrinsic_reward" in output
    assert "0.75" in output


def test_registry_curiosity_disabled() -> None:
    reg = _make_registry(track_curiosity=False)
    reg.set_curiosity_reward(1.0)
    output = reg.render_prometheus()
    assert "curiosity" not in output


def test_registry_inc_sensor_recoveries() -> None:
    reg = _make_registry()
    reg.inc_sensor_recoveries(2)
    reg.inc_sensor_recovery_failures(1)
    output = reg.render_prometheus()
    assert "sensor_recoveries" in output
    assert "sensor_recovery_failures" in output


def test_registry_sensor_recovery_disabled() -> None:
    reg = _make_registry(track_sensor_recovery=False)
    reg.inc_sensor_recoveries(5)
    output = reg.render_prometheus()
    assert "sensor_recoveries" not in output


# ---------------------------------------------------------------------------
# Prometheus text format
# ---------------------------------------------------------------------------


def test_render_prometheus_includes_uptime() -> None:
    reg = _make_registry()
    output = reg.render_prometheus()
    assert "uptime_seconds" in output
    assert "# TYPE" in output
    assert "# HELP" in output


def test_render_prometheus_histogram_format() -> None:
    reg = _make_registry()
    reg.set_loop_time_ms(10.0)
    reg.set_loop_time_ms(25.0)
    output = reg.render_prometheus()
    assert "_bucket{le=" in output
    assert "_sum" in output
    assert "_count" in output


def test_render_prometheus_uses_config_namespace() -> None:
    reg = _make_registry(namespace="testbot")
    output = reg.render_prometheus()
    assert "testbot_uptime_seconds" in output


def test_render_prometheus_safety_violations_empty() -> None:
    """When no violations, the section should be absent (empty snapshot)."""
    reg = _make_registry()
    output = reg.render_prometheus()
    assert "safety_violations" not in output


def test_render_prometheus_voice_events_empty() -> None:
    """When no voice events, the section should be absent (empty snapshot)."""
    reg = _make_registry()
    output = reg.render_prometheus()
    assert "voice_events" not in output


def test_render_prometheus_config_buckets() -> None:
    """Histogram uses bucket boundaries from config."""
    reg = _make_registry(loop_latency_buckets_ms=[5.0, 25.0])
    reg.set_loop_time_ms(10.0)
    output = reg.render_prometheus()
    assert 'le="5"' in output
    assert 'le="25"' in output


def test_render_prometheus_ends_with_newline() -> None:
    reg = _make_registry()
    output = reg.render_prometheus()
    assert output.endswith("\n")
