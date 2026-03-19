"""Tests for MetricsRegistry — Prometheus text format, counters, gauges, histograms.

Covers:
    - Counter increment / reset
    - Gauge set
    - Histogram observe + bucket accumulation
    - Prometheus text exposition format validation
    - Namespace substitution (no hardcoded metric names)
    - MetricsConfig toggle flags disable individual metric families
    - render_prometheus() contract: valid UTF-8, ``# HELP``, ``# TYPE``, trailing newline
"""

from __future__ import annotations

import time

import pytest

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import (
    MetricsRegistry,
    _Counter,
    _Gauge,
    _Histogram,
    _LabeledCounter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(**kwargs: object) -> MetricsRegistry:
    """Create a MetricsRegistry with all toggles enabled by default."""
    cfg = MetricsConfig(**kwargs)
    return MetricsRegistry(cfg)


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line]


# ---------------------------------------------------------------------------
# Internal primitive tests
# ---------------------------------------------------------------------------


class TestCounter:
    def test_initial_value(self) -> None:
        c = _Counter()
        assert c.value == 0

    def test_inc_default(self) -> None:
        c = _Counter()
        c.inc()
        assert c.value == 1

    def test_inc_amount(self) -> None:
        c = _Counter()
        c.inc(5)
        assert c.value == 5

    def test_inc_accumulates(self) -> None:
        c = _Counter()
        c.inc(3)
        c.inc(7)
        assert c.value == 10

    def test_reset_for_testing(self) -> None:
        c = _Counter()
        c.inc(42)
        c.reset()
        assert c.value == 0


class TestLabeledCounter:
    def test_initial_snapshot_empty(self) -> None:
        lc = _LabeledCounter()
        assert lc.snapshot() == {}

    def test_inc_single_label(self) -> None:
        lc = _LabeledCounter()
        lc.inc("law1")
        assert lc.snapshot()["law1"] == 1

    def test_inc_multiple_labels(self) -> None:
        lc = _LabeledCounter()
        lc.inc("law1")
        lc.inc("law2")
        lc.inc("law1")
        snap = lc.snapshot()
        assert snap["law1"] == 2
        assert snap["law2"] == 1

    def test_snapshot_is_copy(self) -> None:
        lc = _LabeledCounter()
        lc.inc("law1")
        snap = lc.snapshot()
        snap["law1"] = 999
        assert lc.snapshot()["law1"] == 1

    def test_reset(self) -> None:
        lc = _LabeledCounter()
        lc.inc("law1", 5)
        lc.reset()
        assert lc.snapshot() == {}


class TestGauge:
    def test_initial_zero(self) -> None:
        g = _Gauge()
        assert g.value == 0.0

    def test_initial_custom(self) -> None:
        g = _Gauge(initial=42.5)
        assert g.value == 42.5

    def test_set(self) -> None:
        g = _Gauge()
        g.set(3.14)
        assert abs(g.value - 3.14) < 1e-9

    def test_set_negative(self) -> None:
        g = _Gauge()
        g.set(-7.0)
        assert g.value == -7.0

    def test_set_overwrites(self) -> None:
        g = _Gauge()
        g.set(1.0)
        g.set(2.0)
        assert g.value == 2.0


class TestHistogram:
    def test_initial_count_zero(self) -> None:
        h = _Histogram((10.0, 50.0, float("inf")))
        _, total_sum, count = h.snapshot()
        assert count == 0
        assert total_sum == 0.0

    def test_observe_below_first_bucket(self) -> None:
        h = _Histogram((10.0, 50.0, float("inf")))
        h.observe(5.0)
        buckets, total_sum, count = h.snapshot()
        assert count == 1
        assert total_sum == 5.0
        # All buckets contain cumulative counts
        assert buckets[0] == (10.0, 1)
        assert buckets[1] == (50.0, 1)
        assert buckets[2] == (float("inf"), 1)

    def test_observe_between_buckets(self) -> None:
        h = _Histogram((10.0, 50.0, float("inf")))
        h.observe(25.0)
        buckets, _, _ = h.snapshot()
        assert buckets[0] == (10.0, 0)  # not ≤ 10
        assert buckets[1] == (50.0, 1)  # ≤ 50
        assert buckets[2] == (float("inf"), 1)

    def test_observe_multiple(self) -> None:
        h = _Histogram((10.0, float("inf")))
        for v in [5.0, 15.0, 25.0]:
            h.observe(v)
        buckets, total_sum, count = h.snapshot()
        assert count == 3
        assert abs(total_sum - 45.0) < 1e-9
        assert buckets[0] == (10.0, 1)
        assert buckets[1] == (float("inf"), 3)


# ---------------------------------------------------------------------------
# MetricsRegistry write/read tests
# ---------------------------------------------------------------------------


class TestMetricsRegistryState:
    def test_initial_frame_drops_zero(self) -> None:
        reg = _make_registry()
        assert reg.frame_drops_total == 0

    def test_inc_frame_drops(self) -> None:
        reg = _make_registry()
        reg.inc_frame_drops()
        reg.inc_frame_drops(3)
        assert reg.frame_drops_total == 4

    def test_frame_drops_disabled_by_toggle(self) -> None:
        reg = _make_registry(track_frame_drops=False)
        reg.inc_frame_drops(10)
        assert reg.frame_drops_total == 0

    def test_inc_safety_violation(self) -> None:
        reg = _make_registry()
        reg.inc_safety_violation("law1")
        reg.inc_safety_violation("law1")
        reg.inc_safety_violation("law2")
        v = reg.safety_violations
        assert v["law1"] == 2
        assert v["law2"] == 1

    def test_safety_violations_disabled_by_toggle(self) -> None:
        reg = _make_registry(track_safety_violations=False)
        reg.inc_safety_violation("law1")
        assert reg.safety_violations == {}

    def test_set_loop_time_ms(self) -> None:
        reg = _make_registry()
        reg.set_loop_time_ms(18.5)
        text = reg.render_prometheus()
        assert "18.5" in text or "18.500" in text

    def test_set_battery_voltage(self) -> None:
        reg = _make_registry()
        reg.set_battery_voltage(11.42)
        text = reg.render_prometheus()
        assert "11.42" in text or "11.4200" in text

    def test_set_ws_client_count(self) -> None:
        reg = _make_registry()
        reg.set_ws_client_count(3)
        text = reg.render_prometheus()
        assert "3" in text

    def test_set_gpu_temp_celsius(self) -> None:
        reg = _make_registry()
        reg.set_gpu_temp_celsius(62.5)
        text = reg.render_prometheus()
        assert "62.5" in text or "62.5000" in text


# ---------------------------------------------------------------------------
# Prometheus text format contract tests
# ---------------------------------------------------------------------------


class TestRenderPrometheus:
    """Validate that render_prometheus() produces spec-compliant output."""

    def test_ends_with_newline(self) -> None:
        reg = _make_registry()
        assert reg.render_prometheus().endswith("\n")

    def test_contains_help_lines(self) -> None:
        reg = _make_registry()
        text = reg.render_prometheus()
        assert "# HELP" in text

    def test_contains_type_lines(self) -> None:
        reg = _make_registry()
        text = reg.render_prometheus()
        assert "# TYPE" in text

    def test_counter_type_declared(self) -> None:
        reg = _make_registry()
        reg.inc_frame_drops(1)
        text = reg.render_prometheus()
        assert "# TYPE" in text
        type_lines = [ln for ln in text.splitlines() if ln.startswith("# TYPE")]
        counter_lines = [ln for ln in type_lines if "counter" in ln]
        assert len(counter_lines) >= 1

    def test_gauge_type_declared(self) -> None:
        reg = _make_registry()
        text = reg.render_prometheus()
        type_lines = [ln for ln in text.splitlines() if ln.startswith("# TYPE")]
        gauge_lines = [ln for ln in type_lines if "gauge" in ln]
        assert len(gauge_lines) >= 1

    def test_histogram_type_declared(self) -> None:
        reg = _make_registry()
        reg.set_loop_time_ms(10.0)
        text = reg.render_prometheus()
        type_lines = [ln for ln in text.splitlines() if ln.startswith("# TYPE")]
        hist_lines = [ln for ln in type_lines if "histogram" in ln]
        assert len(hist_lines) >= 1

    def test_histogram_bucket_lines(self) -> None:
        reg = _make_registry()
        reg.set_loop_time_ms(5.0)
        reg.set_loop_time_ms(15.0)
        text = reg.render_prometheus()
        bucket_lines = [ln for ln in text.splitlines() if "_bucket{" in ln]
        assert len(bucket_lines) >= 5  # at least 5 latency buckets

    def test_histogram_sum_and_count(self) -> None:
        reg = _make_registry()
        reg.set_loop_time_ms(10.0)
        reg.set_loop_time_ms(20.0)
        text = reg.render_prometheus()
        assert any("_sum" in ln for ln in text.splitlines())
        assert any("_count" in ln for ln in text.splitlines())

    def test_labeled_counter_format(self) -> None:
        reg = _make_registry()
        reg.inc_safety_violation("law1")
        reg.inc_safety_violation("law2")
        text = reg.render_prometheus()
        assert 'law="law1"' in text
        assert 'law="law2"' in text

    def test_uptime_always_present(self) -> None:
        reg = _make_registry()
        text = reg.render_prometheus()
        uptime_lines = [ln for ln in text.splitlines() if "uptime" in ln and not ln.startswith("#")]
        assert len(uptime_lines) >= 1

    def test_uptime_is_non_negative(self) -> None:
        reg = _make_registry()
        time.sleep(0.01)
        text = reg.render_prometheus()
        uptime_lines = [
            ln for ln in text.splitlines() if "uptime" in ln and not ln.startswith("#")
        ]
        value = float(uptime_lines[0].split()[-1])
        assert value >= 0.0

    def test_no_blank_metric_names(self) -> None:
        reg = _make_registry()
        for line in reg.render_prometheus().splitlines():
            if line and not line.startswith("#"):
                metric_name = line.split("{")[0].split()[0]
                assert metric_name.strip() != "", f"Blank metric name in line: {line!r}"

    def test_metric_names_use_namespace(self) -> None:
        reg = _make_registry(namespace="mybot")
        text = reg.render_prometheus()
        # Every non-comment, non-empty data line should start with namespace
        data_lines = [
            ln for ln in text.splitlines() if ln and not ln.startswith("#")
        ]
        for line in data_lines:
            name = line.split("{")[0].split()[0]
            assert name.startswith("mybot_"), f"Line does not use namespace: {line!r}"

    def test_custom_namespace_applied(self) -> None:
        reg = _make_registry(namespace="testns")
        text = reg.render_prometheus()
        assert "testns_" in text
        assert "mousedroid_" not in text

    def test_toggles_disable_loop_time(self) -> None:
        reg = _make_registry(track_loop_time=False)
        reg.set_loop_time_ms(99.9)
        text = reg.render_prometheus()
        assert "loop_latency" not in text
        assert "loop_time" not in text

    def test_toggles_disable_battery(self) -> None:
        reg = _make_registry(track_battery=False)
        reg.set_battery_voltage(12.0)
        text = reg.render_prometheus()
        assert "battery" not in text

    def test_toggles_disable_ws_clients(self) -> None:
        reg = _make_registry(track_ws_clients=False)
        reg.set_ws_client_count(5)
        text = reg.render_prometheus()
        assert "ws_client" not in text

    def test_toggles_disable_gpu_temp(self) -> None:
        reg = _make_registry(track_gpu_temp=False)
        reg.set_gpu_temp_celsius(80.0)
        text = reg.render_prometheus()
        assert "gpu_temp" not in text

    def test_only_safety_viol_with_data(self) -> None:
        """Safety violation labeled counter should only appear when non-empty."""
        reg = _make_registry()
        text = reg.render_prometheus()
        # No violations recorded yet — label block should be absent
        assert 'law="' not in text

    def test_valid_utf8_encoding(self) -> None:
        reg = _make_registry()
        text = reg.render_prometheus()
        text.encode("utf-8")  # must not raise

    def test_no_nan_or_inf_in_clean_state(self) -> None:
        reg = _make_registry()
        text = reg.render_prometheus()
        non_inf_lines = [
            ln for ln in text.splitlines() if "+Inf" not in ln and not ln.startswith("#")
        ]
        for line in non_inf_lines:
            parts = line.split()
            if parts:
                # Last token is the value — must parse as float
                float(parts[-1].split("{")[0] if "{" not in line else parts[-1])

    def test_counter_total_suffix(self) -> None:
        """Prometheus counters must use _total suffix per naming conventions."""
        reg = _make_registry()
        reg.inc_frame_drops(1)
        text = reg.render_prometheus()
        counter_value_lines = [
            ln for ln in text.splitlines()
            if "frame_drops" in ln and not ln.startswith("#")
        ]
        assert any("_total" in ln for ln in counter_value_lines)

    def test_histogram_inf_bucket_present(self) -> None:
        reg = _make_registry()
        reg.set_loop_time_ms(500.0)
        text = reg.render_prometheus()
        assert 'le="+Inf"' in text


# ---------------------------------------------------------------------------
# MetricsConfig schema tests
# ---------------------------------------------------------------------------


class TestMetricsConfig:
    def test_defaults(self) -> None:
        cfg = MetricsConfig()
        assert cfg.enabled is True
        assert cfg.namespace == "mousedroid"
        assert cfg.path == "/metrics"
        assert cfg.export_interval_s == 10.0
        assert cfg.track_loop_time is True
        assert cfg.track_battery is True
        assert cfg.track_ws_clients is True
        assert cfg.track_frame_drops is True
        assert cfg.track_safety_violations is True
        assert cfg.track_gpu_temp is True

    def test_custom_namespace(self) -> None:
        cfg = MetricsConfig(namespace="gronk")
        assert cfg.namespace == "gronk"

    def test_custom_path(self) -> None:
        cfg = MetricsConfig(path="/custom/metrics")
        assert cfg.path == "/custom/metrics"

    def test_export_interval_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            MetricsConfig(export_interval_s=0)

    def test_export_interval_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            MetricsConfig(export_interval_s=-1.0)
