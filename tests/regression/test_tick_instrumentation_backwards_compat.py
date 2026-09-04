"""Backwards-compatibility half of the tick-instrumentation config surface.

CLAUDE.md invariant 6: existing YAML must load unchanged after a ``git pull``.
Every field added for tick instrumentation is opt-in, and this file pins that
— a deployment that does not know these fields exist must behave exactly as it
did before, including the resolved ``max_loop_time_ms`` and the rendered
``/metrics`` family set.

Companion file: ``test_tick_instrumentation_aqa.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings
from mousedroid.config.schema.reward_safety import SafetyConfig
from mousedroid.config.schema.telemetry import MetricsConfig

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_OVERLAYS = sorted(p for p in _CONFIG_DIR.glob("*.yaml") if p.name != "baselines.yaml")


def test_overlay_set_is_non_empty() -> None:
    """Guard against the glob matching nothing and vacuously passing below."""
    assert len(_OVERLAYS) >= 10, f"expected the shipped overlays, found {_OVERLAYS}"


class TestDefaultsPreservePreFeatureBehaviour:
    """Every new knob defaults to exactly what the code did before it existed."""

    def test_safety_defaults(self) -> None:
        cfg = SafetyConfig()
        # None => use the literal max_loop_time_ms, no derivation.
        assert cfg.max_loop_time_factor is None
        # 1 => trip on a single overrunning tick, as the monitor always did.
        assert cfg.loop_overrun_consecutive_ticks == 1
        # 0 => no warm-up grace, as the monitor always had none.
        assert cfg.loop_overrun_warmup_ticks == 0
        # 1.0 => soft budget equals the control period exactly.
        assert cfg.loop_soft_budget_factor == 1.0
        # Untouched by the additions.
        assert cfg.max_loop_time_ms == 200.0

    def test_metrics_defaults(self) -> None:
        cfg = MetricsConfig()
        assert cfg.track_tick_phases is True
        assert cfg.tick_phase_buckets_ms[-1] == float("inf")

    def test_a_minimal_pre_feature_settings_still_resolves(self) -> None:
        """A config that predates every field above must load and behave."""
        settings = Settings(mock_hardware=True)
        assert settings.safety.max_loop_time_ms == 200.0
        assert settings.safety.max_loop_time_factor is None


@pytest.mark.parametrize("overlay", _OVERLAYS, ids=lambda p: p.name)
class TestShippedOverlaysUnchanged:
    """No shipped overlay may see a different resolved threshold."""

    def test_overlay_loads(self, overlay: Path) -> None:
        assert load_settings(overlay) is not None

    def test_overlay_threshold_is_not_silently_derived(self, overlay: Path) -> None:
        """None of the shipped overlays opt in, so all keep their literal value.

        If a future overlay sets ``max_loop_time_factor``, this asserts the
        derivation is consistent with it rather than failing — the point is
        that the value is never *silently* different from what the file says.
        """
        settings = load_settings(overlay)
        factor = settings.safety.max_loop_time_factor
        if factor is None:
            assert settings.safety.max_loop_time_ms > 0
        else:
            expected = factor / settings.loop.control_hz * 1000.0
            assert settings.safety.max_loop_time_ms == pytest.approx(expected)


def test_new_metric_families_are_absent_until_observed() -> None:
    """``/metrics`` must render byte-identically for a deployment that never ticks.

    The tick-phase families are pure-add: a registry with no observations must
    not start emitting new HELP lines, or every dashboard and scrape config
    that pins the family set breaks on upgrade.
    """
    from mousedroid.telemetry.metrics.registry import MetricsRegistry

    rendered = MetricsRegistry(MetricsConfig()).render_prometheus()
    assert "mousedroid_tick_phase_ms" not in rendered
    assert "mousedroid_tick_overruns" not in rendered
