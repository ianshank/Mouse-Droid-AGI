"""Byte-identical characterization of ``MetricsRegistry.render_prometheus``.

This is the safety net for Item 2 of the enterprise-hardening sprint: the
``render_prometheus`` decomposition must not reorder, respace, drop, or rename a
single metric family. No pre-existing test pinned the *exact* exposition bytes
(``test_llm_observability_backwards_compat`` only asserts ``name not in out``
for a handful of names), so a naive refactor could silently reshape the output.

The golden fixture (``fixtures/render_prometheus_golden.txt``) was captured from
the pre-refactor implementation with every metric family populated. The only
clock-derived line (the uptime gauge) is masked; every other value is a pure
function of the deterministic populate calls in ``build_maximal_registry``.

If this test fails after an intentional metric change, regenerate the fixture:

    python -c "from tests.regression._render_prometheus_populate import \\
        build_maximal_registry, mask_uptime; \\
        open('tests/regression/fixtures/render_prometheus_golden.txt','w')\\
        .write(mask_uptime(build_maximal_registry().render_prometheus()))"
"""

from __future__ import annotations

from pathlib import Path

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import MetricsRegistry
from tests.regression._render_prometheus_populate import (
    build_maximal_registry,
    mask_uptime,
)

_GOLDEN_PATH = Path(__file__).parent / "fixtures" / "render_prometheus_golden.txt"


def test_render_prometheus_matches_golden() -> None:
    """Full exposition (every family populated) is byte-identical to the golden."""
    rendered = mask_uptime(build_maximal_registry().render_prometheus())
    golden = _GOLDEN_PATH.read_text()
    assert rendered == golden, (
        "render_prometheus output drifted from the golden fixture. If this is an "
        "intentional metric change, regenerate the fixture (see module docstring); "
        "otherwise the render_prometheus refactor changed the exposition bytes."
    )


def test_render_prometheus_family_count() -> None:
    """Guards the fixture itself against silent truncation (62 populated families)."""
    golden = _GOLDEN_PATH.read_text()
    help_lines = [line for line in golden.splitlines() if line.startswith("# HELP")]
    assert len(help_lines) == 63


# The 22 families a fresh (unpopulated) default-config registry emits: the
# always-on gauges/counters that render at their zero value, plus the two
# unconditional families (uptime, publish_hz). Labeled/snapshot families are
# pure-add and stay absent until first write. Characterized from pre-refactor
# behaviour — the decomposition must not change this set.
_DEFAULT_REGISTRY_FAMILIES = [
    "mousedroid_battery_voltage_v",
    "mousedroid_cloud_experience_hwm_lag",
    "mousedroid_cloud_experience_queue_depth",
    "mousedroid_curiosity_intrinsic_reward",
    "mousedroid_frame_drops_total",
    "mousedroid_gpu_temp_celsius",
    "mousedroid_lidar_min_distance_m",
    "mousedroid_lidar_scan_points",
    "mousedroid_llm_latency_ms",
    "mousedroid_llm_requests_total",
    "mousedroid_llm_translation_latency_ms",
    "mousedroid_loop_latency_ms",
    "mousedroid_loop_time_ms",
    "mousedroid_mcp_requests_total",
    "mousedroid_memory_episodic_size",
    "mousedroid_memory_semantic_size",
    "mousedroid_memory_working_size",
    "mousedroid_publish_hz",
    "mousedroid_sensor_recoveries_total",
    "mousedroid_sensor_recovery_failures_total",
    "mousedroid_uptime_seconds",
    "mousedroid_ws_client_count",
]


def test_render_prometheus_default_registry_family_set() -> None:
    """A fresh default-config registry emits exactly the always-on family set.

    Guards the other half of the pure-add contract: labeled/snapshot families
    stay absent until first write, while the always-on gauges/counters render
    at zero. The refactor must not change which families appear here.
    """
    rendered = MetricsRegistry(MetricsConfig.model_validate({})).render_prometheus()
    help_names = sorted(
        line.split()[2] for line in rendered.splitlines() if line.startswith("# HELP")
    )
    assert help_names == _DEFAULT_REGISTRY_FAMILIES
    assert rendered.endswith("\n")
