"""Backwards-compat pin for the emergency-stop latch (peer review D-5).

The latch ships **off**. Latching is strictly fail-safer, so invariant 6
would permit defaulting it on — it is off on operational grounds only: a
default-on latch on a fleet without the re-arm CLI and runbook deployed turns
the first transient sensor dropout into a rover that will not move and an
operator with no documented way to fix it. The ratchet to on is a separate
change, the same shape as the #135 soak gate.

This file pins that "off" is genuinely inert, not merely defaulted — without
it, a future flip of the default would be invisible.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mousedroid.config.schema import EmergencyLatchConfig, SafetyConfig, Settings

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_NOT_AN_OVERLAY = {"baselines.yaml"}


def test_absent_key_resolves_to_the_pre_latch_default() -> None:
    assert Settings.model_validate({"mock_hardware": True}).safety.emergency_latch.enabled is False


def test_legacy_safety_block_loads_unchanged() -> None:
    """A YAML written before the latch existed must load, untouched."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "safety": {"min_valid_sensors": 1, "min_forward_clearance_m": 0.25},
        }
    )
    assert cfg.safety.emergency_latch.enabled is False
    assert cfg.safety.min_valid_sensors == 1
    assert cfg.safety.min_forward_clearance_m == pytest.approx(0.25)


@pytest.mark.parametrize(
    "overlay",
    sorted(p.name for p in _CONFIG_DIR.glob("*.yaml") if p.name not in _NOT_AN_OVERLAY),
)
def test_every_shipped_overlay_still_loads_with_the_latch_off(overlay: str) -> None:
    """Fleet-scale: no shipped config gains the latch by accident."""
    raw = yaml.safe_load((_CONFIG_DIR / overlay).read_text(encoding="utf-8")) or {}
    raw["mock_hardware"] = True
    cfg = Settings.model_validate(raw)
    assert cfg.safety.emergency_latch.enabled is False


def test_the_toggle_round_trips_when_present() -> None:
    cfg = Settings.model_validate(
        {"mock_hardware": True, "safety": {"emergency_latch": {"enabled": True}}}
    )
    assert cfg.safety.emergency_latch.enabled is True


def test_defaults_are_the_documented_ones() -> None:
    latch = EmergencyLatchConfig()
    assert latch.enabled is False
    assert latch.state_dir == "estop_latch"
    assert latch.fsync is True


def test_the_nested_field_uses_a_default_factory() -> None:
    """A shared mutable default would leak state between Settings instances."""
    field = SafetyConfig.model_fields["emergency_latch"]
    assert field.default_factory is not None
    assert field.default_factory().enabled is False  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", ["/abs", "../escape", "a/../../b", "", "   ", "C:\\x", "..\\x"])
def test_state_dir_cannot_escape_the_experience_root(bad: str) -> None:
    """Windows separator forms are the non-obvious half the validator handles."""
    with pytest.raises(ValidationError, match="state_dir"):
        EmergencyLatchConfig(state_dir=bad)


def test_state_dir_accepts_a_sane_relative_path() -> None:
    """Proves the validator is not simply always-raise."""
    assert EmergencyLatchConfig(state_dir="estop_latch").state_dir == "estop_latch"


def test_the_error_message_names_the_right_key() -> None:
    """The shared validator defaults to ``slot_dir``; this field is not that.

    Catches a lazy reuse that would send an operator hunting for a key they
    do not have.
    """
    with pytest.raises(ValidationError) as excinfo:
        EmergencyLatchConfig(state_dir="/abs")
    assert "slot_dir" not in str(excinfo.value)
    assert "safety.emergency_latch.state_dir" in str(excinfo.value)
