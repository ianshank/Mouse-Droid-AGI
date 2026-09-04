"""Backwards-compatibility half of the safety-threshold validators.

CLAUDE.md invariant 6: existing YAML must load unchanged after a ``git pull``.
Adding cross-field validators to ``SafetyConfig`` and bounds to
``UltrasonicConfig`` is exactly the kind of change that can silently break a
deployment's config, so this file pins that every shipped overlay still
resolves and that the defaults are untouched.

Companion file: ``test_safety_threshold_ordering_aqa.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.loader import load_settings
from mousedroid.config.schema.hardware import UltrasonicConfig
from mousedroid.config.schema.reward_safety import SafetyConfig

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

# Overlay YAML files that resolve standalone. ``baselines.yaml`` is a fragment
# merged by the loader rather than an overlay, and the Grafana/Prometheus
# subdirectories are not mousedroid configs.
_OVERLAYS = sorted(p for p in _CONFIG_DIR.glob("*.yaml") if p.name not in {"baselines.yaml"})


def test_overlay_set_is_non_empty() -> None:
    """Guard against the glob silently matching nothing.

    Without this, every parametrised case below would vacuously pass if the
    config directory moved — the failure mode this suite exists to catch.
    """
    assert len(_OVERLAYS) >= 10, f"expected the shipped overlays, found {_OVERLAYS}"


@pytest.mark.parametrize("overlay", _OVERLAYS, ids=lambda p: p.name)
def test_every_shipped_overlay_still_loads(overlay: Path) -> None:
    """Invariant 6: the new validators must reject none of the shipped configs."""
    settings = load_settings(overlay)
    assert settings.safety.gpu_warn_temp_c < settings.safety.gpu_critical_temp_c


def test_safety_defaults_are_unchanged() -> None:
    """The validators must not have moved any default value."""
    cfg = SafetyConfig()
    assert cfg.gpu_warn_temp_c == 75.0
    assert cfg.gpu_critical_temp_c == 90.0
    assert cfg.battery_warn_v == 10.5
    assert cfg.battery_critical_v == 9.5
    assert cfg.battery_implausible_below_v == 1.0
    assert cfg.max_loop_time_ms == 200.0


def test_default_battery_thresholds_satisfy_their_own_ordering() -> None:
    """The shipped defaults must be a valid example of the rule they enforce."""
    cfg = SafetyConfig()
    assert cfg.battery_implausible_below_v < cfg.battery_critical_v < cfg.battery_warn_v


def test_ultrasonic_pin_fields_remain_required() -> None:
    """Bounding the pins must not have given them a default.

    A defaulted GPIO pin would be worse than an unbounded one: it would let a
    config omit the wiring entirely and still construct.
    """
    assert UltrasonicConfig.model_fields["trigger_pin"].is_required()
    assert UltrasonicConfig.model_fields["echo_pin"].is_required()


def test_new_validators_carry_descriptions_on_every_bounded_field() -> None:
    """House rule: every ``Field`` carries a non-empty description."""
    for name in ("trigger_pin", "echo_pin"):
        description = UltrasonicConfig.model_fields[name].description
        assert description, f"UltrasonicConfig.{name} must document its bounds"
        assert "BCM" in description
