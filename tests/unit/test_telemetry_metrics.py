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
    _classify_dropped_observation,
    _Counter,
    _Gauge,
    _Histogram,
    _LabeledCounter,
    _render_counter,
    _render_labeled_counter,
    generate_metrics_sample,
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


class TestClassifyDroppedObservation:
    """Centralised defensive-guard helper shared by every latency histogram.

    Tier C3.1 review (Gemini #3) tightened the drop predicate to also reject
    ``+Inf`` — without this guard, a single watchdog-flagged hang would taint
    the histogram ``_sum`` forever. This test class pins the exact
    classification surface so a future refactor cannot silently re-enable
    ``+Inf`` (or drop a previously-rejected value).
    """

    def test_nan_returns_nan(self) -> None:
        assert _classify_dropped_observation(float("nan")) == "nan"

    def test_positive_inf_returns_inf(self) -> None:
        assert _classify_dropped_observation(float("inf")) == "inf"

    def test_negative_inf_returns_negative(self) -> None:
        """``-inf < _MIN_OBSERVABLE_SECONDS`` so it falls under the negative branch.

        Pinning this prevents an asymmetric guard (``+Inf`` rejected but
        ``-Inf`` slipping past the negativity check on some unusual platform).
        """
        assert _classify_dropped_observation(float("-inf")) == "negative"

    def test_negative_finite_returns_negative(self) -> None:
        assert _classify_dropped_observation(-0.001) == "negative"

    def test_zero_passes_through(self) -> None:
        """Zero is a valid (lower-bound) latency observation.

        Some operations genuinely complete in <1µs and ``time.perf_counter()``
        delta-of-deltas can round to zero. Rejecting zero would create a
        survivorship bias in the histogram sum.
        """
        assert _classify_dropped_observation(0.0) is None

    def test_positive_finite_passes_through(self) -> None:
        assert _classify_dropped_observation(0.05) is None


class TestPrepareBucketBoundaries:
    """``MetricsRegistry._prepare_bucket_boundaries`` — shared bucket normaliser.

    Tier C3.1 review (Gemini #2) DRY-ed five duplicated copies of the same
    sort-and-guarantee-Inf-sentinel logic. This test class pins the helper's
    contract so the DRY refactor stays correct as new histograms are added.
    """

    def test_sorts_ascending(self) -> None:
        result = MetricsRegistry._prepare_bucket_boundaries((0.5, 0.1, 1.0))
        # Last element MUST be +Inf even if input was monotonic.
        assert result == (0.1, 0.5, 1.0, float("inf"))

    def test_appends_inf_when_missing(self) -> None:
        result = MetricsRegistry._prepare_bucket_boundaries((0.01, 0.1))
        assert result[-1] == float("inf")
        assert result == (0.01, 0.1, float("inf"))

    def test_does_not_duplicate_inf_when_already_present(self) -> None:
        result = MetricsRegistry._prepare_bucket_boundaries((0.01, 0.1, float("inf")))
        # Exactly one +Inf, no doubling.
        assert result.count(float("inf")) == 1
        assert result == (0.01, 0.1, float("inf"))

    def test_empty_input_yields_single_inf_bucket(self) -> None:
        """Edge: empty buckets still produce a renderable histogram (le=+Inf only)."""
        result = MetricsRegistry._prepare_bucket_boundaries(())
        assert result == (float("inf"),)

    def test_returns_tuple(self) -> None:
        """Caller depends on the return being hashable / immutable."""
        result = MetricsRegistry._prepare_bucket_boundaries([0.01, 0.1])
        assert isinstance(result, tuple)


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

    def test_inc_llm_translation_result(self) -> None:
        reg = _make_registry()
        reg.inc_llm_translation("translated")
        text = reg.render_prometheus()
        assert 'result="translated"' in text

    def test_observe_llm_translation_latency_histogram(self) -> None:
        reg = _make_registry()
        reg.observe_llm_translation_latency_ms(120.0)
        text = reg.render_prometheus()
        assert "llm_translation_latency_ms_bucket" in text


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

    def test_metric_families_are_separated_by_blank_lines(self) -> None:
        reg = _make_registry()
        reg.inc_frame_drops(1)
        reg.set_loop_time_ms(10.0)
        text = reg.render_prometheus()
        assert "\n\n# HELP" in text

    def test_labeled_counter_escapes_label_values(self) -> None:
        lines = _render_labeled_counter(
            "mousedroid_safety_violations",
            "Safety law violations",
            "law",
            {'law\\"\n1': 2},
        )
        assert lines[-1] == 'mousedroid_safety_violations_total{law="law\\\\\\"\\n1"} 2'

    def test_help_text_escapes_newlines_and_backslashes(self) -> None:
        lines = _render_counter("mousedroid_frame_drops", "line1\\line2\nline3", 1)
        assert lines[0] == "# HELP mousedroid_frame_drops_total line1\\\\line2\\nline3"

    def test_uptime_always_present(self) -> None:
        reg = _make_registry()
        text = reg.render_prometheus()
        uptime_lines = [ln for ln in text.splitlines() if "uptime" in ln and not ln.startswith("#")]
        assert len(uptime_lines) >= 1

    def test_uptime_is_non_negative(self) -> None:
        reg = _make_registry()
        time.sleep(0.01)
        text = reg.render_prometheus()
        uptime_lines = [ln for ln in text.splitlines() if "uptime" in ln and not ln.startswith("#")]
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
        data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
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

    def test_toggles_disable_llm_metrics(self) -> None:
        reg = _make_registry(track_llm_translations=False)
        reg.inc_llm_translation("translated")
        reg.observe_llm_translation_latency_ms(42.0)
        text = reg.render_prometheus()
        assert "llm_translation" not in text

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
            ln for ln in text.splitlines() if "frame_drops" in ln and not ln.startswith("#")
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
        assert cfg.track_llm_translations is True

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


# ---------------------------------------------------------------------------
# generate_metrics_sample() — CI promtool integration
# ---------------------------------------------------------------------------


class TestGenerateMetricsSample:
    def test_returns_nonempty_string(self) -> None:
        sample = generate_metrics_sample()
        assert isinstance(sample, str)
        assert len(sample) > 0

    def test_ends_with_newline(self) -> None:
        sample = generate_metrics_sample()
        assert sample.endswith("\n")

    def test_contains_all_expected_families(self) -> None:
        sample = generate_metrics_sample()
        ns = MetricsConfig().namespace
        for name in (
            f"{ns}_uptime_seconds",
            f"{ns}_loop_time_ms",
            f"{ns}_battery_voltage_v",
            f"{ns}_ws_client_count",
            f"{ns}_gpu_temp_celsius",
            f"{ns}_publish_hz",
            f"{ns}_frame_drops_total",
            f"{ns}_safety_violations_total",
            f"{ns}_loop_latency_ms",
            f"{ns}_llm_translation_total",
            f"{ns}_llm_translation_latency_ms",
        ):
            assert name in sample, f"Missing metric family: {name}"

    def test_has_help_and_type_lines(self) -> None:
        import re

        sample = generate_metrics_sample()
        assert re.search(r"^# HELP \S+ .+$", sample, re.MULTILINE)
        assert re.search(r"^# TYPE \S+ (counter|gauge|histogram)$", sample, re.MULTILINE)


# ---------------------------------------------------------------------------
# PR-A2 — replay / VLA / VLM observability metrics
# ---------------------------------------------------------------------------


class TestReplayRecordsCounter:
    """``mousedroid_replay_records_total{outcome}`` — labeled by ok|schema_mismatch."""

    def test_zero_observations_omits_metric_family(self) -> None:
        """No replay reads → no metric family rendered (pure-add contract)."""
        registry = _make_registry()
        text = registry.render_prometheus()
        assert "replay_records_total" not in text

    def test_records_ok_outcome(self) -> None:
        registry = _make_registry()
        registry.inc_replay_record("ok")
        registry.inc_replay_record("ok")
        registry.inc_replay_record("ok")

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f'{ns}_replay_records_total{{outcome="ok"}} 3' in text
        assert f"# TYPE {ns}_replay_records_total counter" in text

    def test_records_schema_mismatch_outcome(self) -> None:
        registry = _make_registry()
        registry.inc_replay_record("schema_mismatch")
        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f'{ns}_replay_records_total{{outcome="schema_mismatch"}} 1' in text

    def test_records_both_outcomes_independently(self) -> None:
        registry = _make_registry()
        registry.inc_replay_record("ok")
        registry.inc_replay_record("ok")
        registry.inc_replay_record("schema_mismatch")

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f'{ns}_replay_records_total{{outcome="ok"}} 2' in text
        assert f'{ns}_replay_records_total{{outcome="schema_mismatch"}} 1' in text


class TestVlaInferenceSecondsHistogram:
    """``mousedroid_vla_inference_seconds`` — config-driven buckets, sum, count."""

    def test_zero_observations_omits_metric_family(self) -> None:
        registry = _make_registry()
        text = registry.render_prometheus()
        assert "vla_inference_seconds" not in text

    def test_records_inference_latency(self) -> None:
        registry = _make_registry()
        registry.observe_vla_inference_seconds(0.012)
        registry.observe_vla_inference_seconds(0.025)
        registry.observe_vla_inference_seconds(0.5)

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f"# TYPE {ns}_vla_inference_seconds histogram" in text
        assert f"{ns}_vla_inference_seconds_count 3" in text
        # Each observation lands in exactly one bucket; cumulative counts grow.
        assert f"{ns}_vla_inference_seconds_bucket" in text
        assert f"{ns}_vla_inference_seconds_sum" in text

    def test_negative_observation_is_dropped(self) -> None:
        """Defensive: negative latencies (clock skew) must not corrupt sum/count."""
        registry = _make_registry()
        registry.observe_vla_inference_seconds(-1.0)  # rejected
        registry.observe_vla_inference_seconds(0.05)  # accepted

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f"{ns}_vla_inference_seconds_count 1" in text

    def test_buckets_come_from_config(self) -> None:
        """Bucket boundaries must mirror ``MetricsConfig.vla_inference_seconds_buckets``."""
        custom_buckets = (0.01, 0.1, 1.0)
        registry = _make_registry(vla_inference_seconds_buckets=custom_buckets)
        registry.observe_vla_inference_seconds(0.05)

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        for boundary in custom_buckets:
            assert f'{ns}_vla_inference_seconds_bucket{{le="{boundary:.6g}"}}' in text


class TestWorldModelObserveStepSecondsHistogram:
    """``mousedroid_world_model_observe_step_seconds`` — Tier C3.1 wiring.

    Mirrors :class:`TestVlaInferenceSecondsHistogram` so any future refactor
    that breaks the world-model histogram fails fast — same render contract,
    same config-driven bucket boundaries, same omit-when-empty behavior.
    """

    def test_zero_observations_omits_metric_family(self) -> None:
        """Empty histogram MUST NOT appear in the scrape — keeps Prometheus
        cardinality flat until the world-model runtime actually records a
        sample (matches the VLA inference family precedent)."""
        registry = _make_registry()
        text = registry.render_prometheus()
        assert "world_model_observe_step_seconds" not in text

    def test_records_inference_latency(self) -> None:
        """3 observations land in ``_count 3`` with bucket + sum lines present."""
        registry = _make_registry()
        registry.observe_world_model_observe_step_seconds(0.002)
        registry.observe_world_model_observe_step_seconds(0.008)
        registry.observe_world_model_observe_step_seconds(0.04)

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f"# TYPE {ns}_world_model_observe_step_seconds histogram" in text
        assert f"{ns}_world_model_observe_step_seconds_count 3" in text
        assert f"{ns}_world_model_observe_step_seconds_bucket" in text
        assert f"{ns}_world_model_observe_step_seconds_sum" in text

    def test_buckets_come_from_config(self) -> None:
        """Bucket boundaries mirror ``MetricsConfig.world_model_observe_step_seconds_buckets``."""
        custom_buckets = (0.005, 0.01, 0.05)
        registry = _make_registry(world_model_observe_step_seconds_buckets=custom_buckets)
        registry.observe_world_model_observe_step_seconds(0.008)

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        for boundary in custom_buckets:
            assert f'{ns}_world_model_observe_step_seconds_bucket{{le="{boundary:.6g}"}}' in text


class TestVlaTimeoutCounter:
    """``mousedroid_vla_timeouts_total{mode}`` — labeled by backend mode."""

    def test_zero_timeouts_omits_metric_family(self) -> None:
        registry = _make_registry()
        text = registry.render_prometheus()
        assert "vla_timeouts_total" not in text

    @pytest.mark.parametrize("mode", ["mock", "distilled_onnx"])
    def test_records_timeout_by_mode(self, mode: str) -> None:
        registry = _make_registry()
        registry.inc_vla_timeout(mode)
        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f'{ns}_vla_timeouts_total{{mode="{mode}"}} 1' in text


class TestVlmProgressCacheCounters:
    """``mousedroid_vlm_progress_cache_hits_total`` and ``..._misses_total``."""

    def test_zero_hits_and_misses_omits_families(self) -> None:
        registry = _make_registry()
        text = registry.render_prometheus()
        assert "vlm_progress_cache_hits" not in text
        assert "vlm_progress_cache_misses" not in text

    def test_records_cache_hit(self) -> None:
        registry = _make_registry()
        registry.inc_vlm_cache_hit()
        registry.inc_vlm_cache_hit(amount=2)

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f"{ns}_vlm_progress_cache_hits_total 3" in text

    def test_records_cache_miss(self) -> None:
        registry = _make_registry()
        registry.inc_vlm_cache_miss()
        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f"{ns}_vlm_progress_cache_misses_total 1" in text

    def test_hit_and_miss_increment_independently(self) -> None:
        """Hit and miss counters must not bleed across each other."""
        registry = _make_registry()
        registry.inc_vlm_cache_hit()
        registry.inc_vlm_cache_miss()
        registry.inc_vlm_cache_miss()

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f"{ns}_vlm_progress_cache_hits_total 1" in text
        assert f"{ns}_vlm_progress_cache_misses_total 2" in text


class TestGenerateMetricsSampleNewMetrics:
    """Ensure the PR-A2 metrics appear in the CI promtool sample."""

    def test_sample_exercises_all_four_new_metrics(self) -> None:
        sample = generate_metrics_sample()
        # Pull namespace from default config so the assertions stay
        # config-driven (no hardcoded ``mousedroid_`` prefix).
        ns = MetricsConfig().namespace
        for name in (
            f"{ns}_replay_records_total",
            f"{ns}_vla_inference_seconds_bucket",
            f"{ns}_vla_inference_seconds_sum",
            f"{ns}_vla_inference_seconds_count",
            f"{ns}_vla_timeouts_total",
            f"{ns}_vlm_progress_cache_hits_total",
            f"{ns}_vlm_progress_cache_misses_total",
        ):
            assert name in sample, f"PR-A2 metric missing from sample: {name}"

    def test_sample_includes_both_replay_outcomes(self) -> None:
        sample = generate_metrics_sample()
        ns = MetricsConfig().namespace
        assert f'{ns}_replay_records_total{{outcome="ok"}}' in sample
        assert f'{ns}_replay_records_total{{outcome="schema_mismatch"}}' in sample


class TestMetricsConfigBucketField:
    """``MetricsConfig.vla_inference_seconds_buckets`` is a Pydantic-validated tuple."""

    def test_default_buckets_are_finite_and_ascending(self) -> None:
        cfg = MetricsConfig()
        buckets = cfg.vla_inference_seconds_buckets
        # Drop the +Inf sentinel before checking monotonicity.
        finite = [b for b in buckets if b != float("inf")]
        assert finite == sorted(finite), "Default buckets must be ascending"
        assert all(b > 0 for b in finite), "Default buckets must be positive"

    def test_custom_buckets_round_trip(self) -> None:
        cfg = MetricsConfig(vla_inference_seconds_buckets=(0.001, 0.01, 0.1))
        assert cfg.vla_inference_seconds_buckets == (0.001, 0.01, 0.1)


class TestHistogramBucketValidator:
    """The shared ``_validate_histogram_buckets`` Pydantic validator covers
    all four bucket fields: loop_latency_buckets_ms, llm_latency_buckets_ms,
    mcp_latency_buckets_ms, and vla_inference_seconds_buckets.

    Negative, zero, duplicate, and non-ascending values would silently corrupt
    histogram bucket accumulation, so the validator rejects them at schema-load.
    """

    @pytest.mark.parametrize(
        "field",
        [
            "loop_latency_buckets_ms",
            "llm_latency_buckets_ms",
            "mcp_latency_buckets_ms",
            "vla_inference_seconds_buckets",
        ],
    )
    def test_validator_rejects_descending_order(self, field: str) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="monotonically ascending"):
            MetricsConfig(**{field: (10.0, 5.0, 1.0)})

    @pytest.mark.parametrize(
        "field",
        [
            "loop_latency_buckets_ms",
            "llm_latency_buckets_ms",
            "mcp_latency_buckets_ms",
            "vla_inference_seconds_buckets",
        ],
    )
    def test_validator_rejects_negative_values(self, field: str) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="strictly positive"):
            MetricsConfig(**{field: (-1.0, 5.0, 10.0)})

    @pytest.mark.parametrize(
        "field",
        [
            "loop_latency_buckets_ms",
            "llm_latency_buckets_ms",
            "mcp_latency_buckets_ms",
            "vla_inference_seconds_buckets",
        ],
    )
    def test_validator_rejects_zero_value(self, field: str) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="strictly positive"):
            MetricsConfig(**{field: (0.0, 5.0, 10.0)})

    @pytest.mark.parametrize(
        "field",
        [
            "loop_latency_buckets_ms",
            "llm_latency_buckets_ms",
            "mcp_latency_buckets_ms",
            "vla_inference_seconds_buckets",
        ],
    )
    def test_validator_rejects_duplicates(self, field: str) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="unique"):
            MetricsConfig(**{field: (1.0, 5.0, 5.0, 10.0)})

    @pytest.mark.parametrize(
        "field",
        [
            "loop_latency_buckets_ms",
            "llm_latency_buckets_ms",
            "mcp_latency_buckets_ms",
            "vla_inference_seconds_buckets",
        ],
    )
    def test_validator_accepts_inf_sentinel(self, field: str) -> None:
        cfg = MetricsConfig(**{field: (1.0, 5.0, 10.0, float("inf"))})
        assert getattr(cfg, field) == (1.0, 5.0, 10.0, float("inf"))

    def test_validator_rejects_empty_tuple(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="non-empty"):
            MetricsConfig(vla_inference_seconds_buckets=())

    @pytest.mark.parametrize(
        "bad_tuple",
        [
            (float("inf"), 1.0, 5.0),  # inf at the head
            (1.0, float("inf"), 5.0, 10.0),  # inf in the middle
            (1.0, 5.0, float("inf"), 10.0),  # inf in the middle (before another finite)
        ],
    )
    def test_validator_rejects_inf_in_non_trailing_position(
        self, bad_tuple: tuple[float, ...]
    ) -> None:
        """``+inf`` is only meaningful as the trailing sentinel — anywhere else
        it would yield surprising bucket cardinality after the registry's
        runtime sort."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="trailing sentinel"):
            MetricsConfig(vla_inference_seconds_buckets=bad_tuple)


class TestLiteralTypeAliases:
    """The exported Literal aliases keep label values in sync with the schema."""

    def test_vla_backend_literal_matches_vla_config_field(self) -> None:
        """The :data:`VLABackendLiteral` alias must enumerate the same values
        as :class:`VLAConfig.backend`."""
        import typing as _typing

        from mousedroid.config.schema import VLABackendLiteral, VLAConfig

        alias_args = set(_typing.get_args(VLABackendLiteral))
        field_args = set(_typing.get_args(VLAConfig.model_fields["backend"].annotation))
        assert alias_args == field_args, (
            f"VLABackendLiteral drift: alias has {alias_args}, "
            f"VLAConfig.backend has {field_args}"
        )

    def test_replay_outcome_literal_has_expected_values(self) -> None:
        import typing as _typing

        from mousedroid.config.schema import ReplayOutcomeLiteral

        assert set(_typing.get_args(ReplayOutcomeLiteral)) == {"ok", "schema_mismatch"}

    def test_vla_active_backend_literal_is_strict_subset_of_full_backend(self) -> None:
        """``VLAActiveBackendLiteral`` must exclude ``"none"`` and otherwise
        match :data:`VLABackendLiteral` — used to type
        :meth:`MetricsRegistry.inc_vla_timeout` so a disabled backend cannot
        emit timeouts and pollute the metric with ``{mode="none"}`` series."""
        import typing as _typing

        from mousedroid.config.schema import VLAActiveBackendLiteral, VLABackendLiteral

        full = set(_typing.get_args(VLABackendLiteral))
        active = set(_typing.get_args(VLAActiveBackendLiteral))
        assert active < full, "VLAActiveBackendLiteral must be a strict subset"
        assert active == full - {"none"}, (
            f"VLAActiveBackendLiteral must equal VLABackendLiteral minus 'none'; "
            f"got active={active}, full={full}"
        )


class TestPrA2DefensiveGuards:
    """Counter monotonicity + histogram NaN/negative-value rejection guards
    enforced after PR-A2 review feedback.
    """

    def test_observe_vla_inference_seconds_drops_nan(self) -> None:
        """NaN samples must not enter the histogram sum (which would taint it)."""
        registry = _make_registry()
        registry.observe_vla_inference_seconds(float("nan"))
        registry.observe_vla_inference_seconds(0.05)  # valid sample

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f"{ns}_vla_inference_seconds_count 1" in text
        # The sum must not be NaN — render it and confirm no NaN substring.
        assert "NaN" not in text

    def test_observe_vla_inference_seconds_drops_negative(self) -> None:
        """Negative samples are dropped (regression-locked from original PR-A2 work)."""
        registry = _make_registry()
        registry.observe_vla_inference_seconds(-1.0)
        registry.observe_vla_inference_seconds(0.05)

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f"{ns}_vla_inference_seconds_count 1" in text

    def test_observe_world_model_observe_step_seconds_drops_nan(self) -> None:
        """Tier C3.1 helper must reject NaN to keep the histogram sum sane.

        Mirrors :meth:`test_observe_vla_inference_seconds_drops_nan` —
        a single NaN observation would render ``_sum NaN`` and propagate to
        every Grafana panel + alert that consumes the family.
        """
        registry = _make_registry()
        registry.observe_world_model_observe_step_seconds(float("nan"))
        registry.observe_world_model_observe_step_seconds(0.008)  # valid sample

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f"{ns}_world_model_observe_step_seconds_count 1" in text
        assert "NaN" not in text

    def test_observe_world_model_observe_step_seconds_drops_negative(self) -> None:
        """Negative wall-clock samples (clock skew, division-by-zero) are dropped."""
        registry = _make_registry()
        registry.observe_world_model_observe_step_seconds(-0.001)
        registry.observe_world_model_observe_step_seconds(0.008)

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f"{ns}_world_model_observe_step_seconds_count 1" in text

    def test_observe_vla_inference_seconds_drops_inf(self) -> None:
        """``+Inf`` samples MUST be rejected — they would taint ``_sum`` forever.

        ``_Histogram.observe`` routes ``+Inf`` into the ``le=+Inf`` bucket
        without complaint, but the rolling ``_sum`` accumulator would then
        equal ``+Inf`` from that point onward and break every
        ``histogram_quantile`` / ``rate`` computation downstream.
        """
        registry = _make_registry()
        registry.observe_vla_inference_seconds(float("inf"))
        registry.observe_vla_inference_seconds(0.05)

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f"{ns}_vla_inference_seconds_count 1" in text
        # The +Inf sample must not have leaked into the sum.
        assert "+Inf" not in text or "_sum +Inf" not in text

    def test_observe_world_model_observe_step_seconds_drops_inf(self) -> None:
        """``+Inf`` samples MUST be rejected on the world-model helper too.

        Mirrors :meth:`test_observe_vla_inference_seconds_drops_inf` — same
        rationale, same defensive guard. Pinning this on both helpers prevents
        a future refactor from accidentally re-enabling ``+Inf`` on one but
        not the other.
        """
        registry = _make_registry()
        registry.observe_world_model_observe_step_seconds(float("inf"))
        registry.observe_world_model_observe_step_seconds(0.008)

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f"{ns}_world_model_observe_step_seconds_count 1" in text
        assert "+Inf" not in text or "_sum +Inf" not in text

    _DROP_EVENT = "vla_inference_seconds_dropped"
    _WM_DROP_EVENT = "world_model_observe_step_seconds_dropped"

    def test_drop_emits_debug_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dropped samples must emit a DEBUG-level structured log.

        Asserted by intercepting the module-level ``_log.debug`` call rather
        than reconfiguring the global structlog state — avoids leaking
        test-only logger configuration into adjacent tests in the session.
        """
        from mousedroid.telemetry import metrics as metrics_module

        captured: list[tuple[str, dict[str, object]]] = []

        def _fake_debug(event: str, **kwargs: object) -> None:
            captured.append((event, dict(kwargs)))

        # Patch the module-level logger so the registry constructor's
        # ``metrics_registry_initialised`` debug log is also captured but
        # filtered below to the drop event only.
        registry = _make_registry()
        monkeypatch.setattr(metrics_module._log, "debug", _fake_debug)

        registry.observe_vla_inference_seconds(float("nan"))
        registry.observe_vla_inference_seconds(-0.001)
        registry.observe_vla_inference_seconds(float("inf"))
        registry.observe_vla_inference_seconds(0.05)  # accepted — no log

        drops = [(event, kwargs) for event, kwargs in captured if event == self._DROP_EVENT]
        assert len(drops) == 3, f"expected 3 drop logs, got {drops!r}"
        reasons = [kwargs.get("reason") for _, kwargs in drops]
        assert set(reasons) == {"nan", "negative", "inf"}

    def test_world_model_drop_emits_debug_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tier C3.1 helper must emit ``world_model_observe_step_seconds_dropped``
        on NaN + negative samples, mirroring the VLA-inference drop-log shape.

        Operator runbook: ``world_model_observe_step_seconds_dropped`` at
        DEBUG correlates a missing-histogram-observation with the upstream
        wall-clock root cause (clock skew, division-by-zero on a zero-budget
        timer, etc.). Asserting the structured-log shape here pins the
        contract so a future refactor that silently swallows the drop fails.
        """
        from mousedroid.telemetry import metrics as metrics_module

        captured: list[tuple[str, dict[str, object]]] = []

        def _fake_debug(event: str, **kwargs: object) -> None:
            captured.append((event, dict(kwargs)))

        registry = _make_registry()
        monkeypatch.setattr(metrics_module._log, "debug", _fake_debug)

        registry.observe_world_model_observe_step_seconds(float("nan"))
        registry.observe_world_model_observe_step_seconds(-0.001)
        registry.observe_world_model_observe_step_seconds(float("inf"))
        registry.observe_world_model_observe_step_seconds(0.008)  # accepted — no drop log

        drops = [(event, kwargs) for event, kwargs in captured if event == self._WM_DROP_EVENT]
        assert len(drops) == 3, f"expected 3 drop logs, got {drops!r}"
        reasons = [kwargs.get("reason") for _, kwargs in drops]
        assert set(reasons) == {"nan", "negative", "inf"}

    @pytest.mark.parametrize(
        "helper_name",
        [
            "inc_replay_record",
            "inc_vla_timeout",
            "inc_vlm_cache_hit",
            "inc_vlm_cache_miss",
        ],
    )
    def test_non_positive_amount_is_noop(self, helper_name: str) -> None:
        """All four counter helpers must guard counter monotonicity at amount <= 0."""
        registry = _make_registry()
        # Build a call signature appropriate for each helper.
        if helper_name == "inc_replay_record":
            registry.inc_replay_record("ok", amount=0)
            registry.inc_replay_record("ok", amount=-1)
        elif helper_name == "inc_vla_timeout":
            registry.inc_vla_timeout("mock", amount=0)
            registry.inc_vla_timeout("mock", amount=-1)
        elif helper_name == "inc_vlm_cache_hit":
            registry.inc_vlm_cache_hit(amount=0)
            registry.inc_vlm_cache_hit(amount=-1)
        else:  # inc_vlm_cache_miss
            registry.inc_vlm_cache_miss(amount=0)
            registry.inc_vlm_cache_miss(amount=-1)

        text = registry.render_prometheus()
        # No metric family should appear since no positive amount was applied.
        assert "replay_records_total" not in text
        assert "vla_timeouts_total" not in text
        assert "vlm_progress_cache_hits_total" not in text
        assert "vlm_progress_cache_misses_total" not in text

    def test_inc_replay_record_accepts_amount(self) -> None:
        """API consistency: ``amount`` kwarg works on inc_replay_record."""
        registry = _make_registry()
        registry.inc_replay_record("ok", amount=3)
        registry.inc_replay_record("schema_mismatch", amount=5)

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f'{ns}_replay_records_total{{outcome="ok"}} 3' in text
        assert f'{ns}_replay_records_total{{outcome="schema_mismatch"}} 5' in text

    def test_inc_vla_timeout_accepts_amount(self) -> None:
        """API consistency: ``amount`` kwarg works on inc_vla_timeout."""
        registry = _make_registry()
        registry.inc_vla_timeout("distilled_onnx", amount=4)

        text = registry.render_prometheus()
        ns = registry._cfg.namespace
        assert f'{ns}_vla_timeouts_total{{mode="distilled_onnx"}} 4' in text

    def test_inc_vla_timeout_type_excludes_none_mode(self) -> None:
        """The narrowed ``VLAActiveBackendLiteral`` excludes ``"none"`` so
        mypy --strict rejects ``inc_vla_timeout("none")`` at type-check time.
        We verify the *runtime* narrowing at the alias level — ``"none"`` is
        not a legal value of the parameter type, so no test calls it
        directly (would be a mypy error). This test pins the structural
        invariant in case the alias is widened in the future.
        """
        import typing as _typing

        from mousedroid.config.schema import VLAActiveBackendLiteral

        assert "none" not in _typing.get_args(VLAActiveBackendLiteral)
