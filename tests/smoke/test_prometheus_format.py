"""Smoke test for Prometheus text exposition format.

Validates that :class:`MetricsRegistry` produces well-formed Prometheus
text output with the expected metric families.  Self-contained -- no
external services required.

All metric names are derived from :attr:`MetricsConfig.namespace` so
the test suite adapts automatically if the namespace is changed.
"""

from __future__ import annotations

import re

import pytest

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import MetricsRegistry

pytestmark = pytest.mark.smoke

# ---------------------------------------------------------------------------
# Metric name derivation from config (NOT hardcoded)
# ---------------------------------------------------------------------------

_CFG = MetricsConfig()
_NS = _CFG.namespace  # default: "mousedroid"

# Gauge families (rendered with HELP/TYPE/value, no _total suffix)
_GAUGE_FAMILIES: tuple[str, ...] = (
    f"{_NS}_uptime_seconds",
    f"{_NS}_loop_time_ms",
    f"{_NS}_battery_voltage_v",
    f"{_NS}_ws_client_count",
    f"{_NS}_gpu_temp_celsius",
    f"{_NS}_publish_hz",
)

# Counter families (rendered with _total suffix in TYPE/HELP/sample lines)
_COUNTER_FAMILIES: tuple[str, ...] = (
    f"{_NS}_frame_drops",
    f"{_NS}_safety_violations",
)

# Histogram families (rendered with _bucket, _sum, _count suffixes)
_HISTOGRAM_FAMILIES: tuple[str, ...] = (f"{_NS}_loop_latency_ms",)

# Union of all families that must be present in output
_ALL_FAMILIES: tuple[str, ...] = _GAUGE_FAMILIES + _COUNTER_FAMILIES + _HISTOGRAM_FAMILIES

# Prometheus text format requires every metric family to have a HELP
# and TYPE declaration line.
_HELP_RE = re.compile(r"^# HELP (\S+) .+$", re.MULTILINE)
_TYPE_RE = re.compile(r"^# TYPE (\S+) (counter|gauge|histogram|summary|untyped)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_registry() -> MetricsRegistry:
    """Create a MetricsRegistry with all toggles enabled and populate values."""
    cfg = MetricsConfig()
    registry = MetricsRegistry(cfg)

    # Populate every metric so they all appear in the rendered output.
    registry.set_loop_time_ms(15.0)
    registry.set_battery_voltage(11.8)
    registry.set_ws_client_count(2)
    registry.set_gpu_temp_celsius(52.0)
    registry.set_publish_hz(10.0)
    registry.inc_frame_drops(3)
    registry.inc_safety_violation("law1")
    return registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_render_returns_nonempty_string() -> None:
    """Rendered output must be a non-empty string ending with a newline."""
    registry = _build_registry()
    text = registry.render_prometheus()
    assert isinstance(text, str)
    assert len(text) > 0
    assert text.endswith("\n"), "Prometheus text must end with a trailing newline"


def test_every_expected_family_present() -> None:
    """Every expected metric family appears in the rendered output.

    Gauge and histogram families appear by base name; counter families
    appear with ``_total`` suffix appended by the renderer.
    """
    registry = _build_registry()
    text = registry.render_prometheus()

    for family in _GAUGE_FAMILIES:
        assert family in text, f"Missing gauge family: {family}"

    for family in _COUNTER_FAMILIES:
        total_name = f"{family}_total"
        assert total_name in text, f"Missing counter family: {total_name}"

    for family in _HISTOGRAM_FAMILIES:
        assert family in text, f"Missing histogram family: {family}"


def test_help_lines_well_formed() -> None:
    """Every HELP line matches the Prometheus format spec."""
    registry = _build_registry()
    text = registry.render_prometheus()
    help_names = _HELP_RE.findall(text)
    assert len(help_names) >= len(
        _ALL_FAMILIES
    ), f"Expected at least {len(_ALL_FAMILIES)} HELP lines, got {len(help_names)}"


def test_type_lines_well_formed() -> None:
    """Every TYPE line uses a valid Prometheus metric type."""
    registry = _build_registry()
    text = registry.render_prometheus()
    type_pairs = _TYPE_RE.findall(text)
    assert len(type_pairs) >= len(
        _ALL_FAMILIES
    ), f"Expected at least {len(_ALL_FAMILIES)} TYPE lines, got {len(type_pairs)}"
    for name, kind in type_pairs:
        assert kind in {
            "counter",
            "gauge",
            "histogram",
            "summary",
            "untyped",
        }, f"Invalid type '{kind}' for metric '{name}'"


def test_counter_uses_total_suffix() -> None:
    """Counter metric names must end with ``_total`` per Prometheus convention."""
    registry = _build_registry()
    text = registry.render_prometheus()
    for line in text.splitlines():
        if line.startswith("# TYPE") and "counter" in line:
            parts = line.split()
            metric_name = parts[2]
            assert metric_name.endswith(
                "_total"
            ), f"Counter metric '{metric_name}' missing _total suffix"


def test_histogram_has_bucket_sum_count() -> None:
    """Histogram metrics must include _bucket, _sum, and _count lines."""
    registry = _build_registry()
    text = registry.render_prometheus()
    ns = _NS
    assert f"{ns}_loop_latency_ms_bucket" in text, "Missing histogram _bucket lines"
    assert f"{ns}_loop_latency_ms_sum" in text, "Missing histogram _sum line"
    assert f"{ns}_loop_latency_ms_count" in text, "Missing histogram _count line"


def test_histogram_buckets_include_inf() -> None:
    """Histogram must include a +Inf bucket as required by the spec."""
    registry = _build_registry()
    text = registry.render_prometheus()
    ns = _NS
    assert f'{ns}_loop_latency_ms_bucket{{le="+Inf"}}' in text, "Missing +Inf bucket"


def test_gauge_values_are_numeric() -> None:
    """Gauge sample lines must have a numeric value."""
    registry = _build_registry()
    text = registry.render_prometheus()
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # Sample line format: metric_name{labels} value
        parts = line.rsplit(" ", maxsplit=1)
        assert len(parts) == 2, f"Malformed sample line: {line}"
        value_str = parts[1]
        # Valid Prometheus values: number, +Inf, -Inf, NaN
        assert re.match(r"^[+-]?(\d+\.?\d*|\d*\.?\d+)([eE][+-]?\d+)?$", value_str) or (
            value_str in {"+Inf", "-Inf", "NaN"}
        ), f"Non-numeric value '{value_str}' in line: {line}"


def test_safety_violations_labeled_counter() -> None:
    """Safety violations counter includes the 'law' label."""
    registry = _build_registry()
    text = registry.render_prometheus()
    ns = _NS
    assert (
        f'{ns}_safety_violations_total{{law="law1"}}' in text
    ), "Safety violations counter missing law label"


def test_no_duplicate_type_declarations() -> None:
    """Each metric family must have exactly one TYPE declaration."""
    registry = _build_registry()
    text = registry.render_prometheus()
    type_names = _TYPE_RE.findall(text)
    seen: set[str] = set()
    for name, _ in type_names:
        assert name not in seen, f"Duplicate TYPE declaration for '{name}'"
        seen.add(name)


def test_namespace_is_configurable() -> None:
    """Metric names must derive from MetricsConfig.namespace, not be hardcoded."""
    custom_ns = "testbot"
    cfg = MetricsConfig(namespace=custom_ns)
    registry = MetricsRegistry(cfg)
    registry.set_loop_time_ms(10.0)
    registry.set_battery_voltage(12.0)
    registry.set_gpu_temp_celsius(45.0)
    registry.set_ws_client_count(1)
    registry.set_publish_hz(10.0)
    registry.inc_frame_drops(1)
    registry.inc_safety_violation("law2")

    text = registry.render_prometheus()

    # All metric names should use the custom namespace
    assert f"{custom_ns}_uptime_seconds" in text
    assert f"{custom_ns}_loop_time_ms" in text
    assert f"{custom_ns}_battery_voltage_v" in text
    assert f"{custom_ns}_frame_drops_total" in text
    assert f"{custom_ns}_safety_violations_total" in text
    assert f"{custom_ns}_loop_latency_ms" in text
    assert f"{custom_ns}_gpu_temp_celsius" in text
    assert f"{custom_ns}_ws_client_count" in text
    assert f"{custom_ns}_publish_hz" in text

    # Default namespace should NOT appear
    assert (
        "mousedroid_" not in text
    ), "Default namespace 'mousedroid_' should not appear when custom namespace is set"


def test_help_and_type_paired() -> None:
    """Every metric with a TYPE line also has a HELP line, and vice versa."""
    registry = _build_registry()
    text = registry.render_prometheus()
    help_names = set(_HELP_RE.findall(text))
    type_names = {name for name, _ in _TYPE_RE.findall(text)}
    assert help_names == type_names, (
        f"HELP/TYPE mismatch: help_only={help_names - type_names}, "
        f"type_only={type_names - help_names}"
    )
