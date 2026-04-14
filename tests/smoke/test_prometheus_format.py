"""Smoke test for Prometheus text exposition format.

Validates that :class:`MetricsRegistry` produces well-formed Prometheus
text output with the expected metric families.  Self-contained — no
external services required.
"""

from __future__ import annotations

import re

import pytest

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import MetricsRegistry

pytestmark = pytest.mark.smoke

# Metric families that MUST appear in the rendered output.
# These correspond to the pre-formatted names built from the
# ``mousedroid`` namespace in MetricsRegistry.__init__.
_EXPECTED_FAMILIES: tuple[str, ...] = (
    "mousedroid_uptime_seconds",
    "mousedroid_frame_drops_total",
    "mousedroid_loop_time_ms",
    "mousedroid_loop_latency_ms",
    "mousedroid_battery_voltage_v",
    "mousedroid_ws_client_count",
    "mousedroid_gpu_temp_celsius",
    "mousedroid_publish_hz",
)

# Prometheus text format requires every metric family to have a HELP
# and TYPE declaration line.
_HELP_RE = re.compile(r"^# HELP (\S+) .+$", re.MULTILINE)
_TYPE_RE = re.compile(r"^# TYPE (\S+) (counter|gauge|histogram|summary|untyped)$", re.MULTILINE)


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


def test_render_returns_nonempty_string() -> None:
    """Rendered output must be a non-empty string ending with a newline."""
    registry = _build_registry()
    text = registry.render_prometheus()
    assert isinstance(text, str)
    assert len(text) > 0
    assert text.endswith("\n"), "Prometheus text must end with a trailing newline"


def test_every_expected_family_present() -> None:
    """Every expected metric family appears in the rendered output."""
    registry = _build_registry()
    text = registry.render_prometheus()
    for family in _EXPECTED_FAMILIES:
        assert family in text, f"Missing metric family: {family}"


def test_help_lines_well_formed() -> None:
    """Every HELP line matches the Prometheus format spec."""
    registry = _build_registry()
    text = registry.render_prometheus()
    help_names = _HELP_RE.findall(text)
    assert len(help_names) >= len(
        _EXPECTED_FAMILIES
    ), f"Expected at least {len(_EXPECTED_FAMILIES)} HELP lines, got {len(help_names)}"


def test_type_lines_well_formed() -> None:
    """Every TYPE line uses a valid Prometheus metric type."""
    registry = _build_registry()
    text = registry.render_prometheus()
    type_pairs = _TYPE_RE.findall(text)
    assert len(type_pairs) >= len(
        _EXPECTED_FAMILIES
    ), f"Expected at least {len(_EXPECTED_FAMILIES)} TYPE lines, got {len(type_pairs)}"
    for name, kind in type_pairs:
        assert kind in {
            "counter",
            "gauge",
            "histogram",
            "summary",
            "untyped",
        }, f"Invalid type '{kind}' for metric '{name}'"


def test_counter_uses_total_suffix() -> None:
    """Counter metric names must end with ``_total``."""
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
    assert "mousedroid_loop_latency_ms_bucket" in text
    assert "mousedroid_loop_latency_ms_sum" in text
    assert "mousedroid_loop_latency_ms_count" in text


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
    assert 'mousedroid_safety_violations_total{law="law1"}' in text


def test_no_duplicate_type_declarations() -> None:
    """Each metric family must have exactly one TYPE declaration."""
    registry = _build_registry()
    text = registry.render_prometheus()
    type_names = _TYPE_RE.findall(text)
    seen: set[str] = set()
    for name, _ in type_names:
        assert name not in seen, f"Duplicate TYPE declaration for '{name}'"
        seen.add(name)
