"""Unit tests for ``OnDeviceLearningConfig`` (Phase 6 WS1).

Covers every field default and every constraint boundary so a future refactor
that loosens a bound (e.g. ``gt`` -> ``ge``) fails fast. The model is the
foundation of the on-device incremental-learning subsystem: default-OFF,
backwards-compatible, NO hardcoded absolute paths.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import OnDeviceLearningConfig


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
def test_defaults_are_safe_and_off() -> None:
    cfg = OnDeviceLearningConfig()
    assert cfg.enabled is False
    assert cfg.trigger_min_new_records == 500
    assert cfg.update_steps == 50
    assert cfg.regression_tolerance == 0.05
    assert cfg.held_out_fraction == 0.1
    assert cfg.ewc_lambda == 1.0
    assert cfg.learning_rate == 1e-4
    assert cfg.slot_dir == "on_device_slot"


def test_slot_dir_is_not_an_absolute_host_path() -> None:
    """The default slot_dir must be experience-root-relative — resolved under it."""
    cfg = OnDeviceLearningConfig()
    assert not cfg.slot_dir.startswith("/")
    assert "/home/jetson" not in cfg.slot_dir


@pytest.mark.parametrize(
    "bad",
    [
        "/abs/path",  # absolute path escapes the experience root
        "../escape",  # parent traversal escapes the experience root
        "",  # empty
        "   ",  # whitespace-only
        "nested/../escape",  # parent traversal in a deeper component
    ],
)
def test_slot_dir_rejects_unsafe_values(bad: str) -> None:
    """Absolute, parent-traversal, and empty/whitespace slot_dir are rejected."""
    with pytest.raises(ValidationError):
        OnDeviceLearningConfig(slot_dir=bad)


def test_slot_dir_accepts_safe_relative_values() -> None:
    """A non-empty relative leaf (default + a nested relative path) is accepted."""
    assert OnDeviceLearningConfig(slot_dir="on_device_slot").slot_dir == "on_device_slot"
    assert OnDeviceLearningConfig(slot_dir="weights/on_device").slot_dir == "weights/on_device"


# --------------------------------------------------------------------------- #
# Constraint validation (boundaries)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0, -1])
def test_trigger_min_new_records_must_be_positive(bad: int) -> None:
    with pytest.raises(ValidationError):
        OnDeviceLearningConfig(trigger_min_new_records=bad)


@pytest.mark.parametrize("bad", [0, -5])
def test_update_steps_must_be_positive(bad: int) -> None:
    with pytest.raises(ValidationError):
        OnDeviceLearningConfig(update_steps=bad)


def test_regression_tolerance_allows_zero_but_not_negative() -> None:
    assert OnDeviceLearningConfig(regression_tolerance=0.0).regression_tolerance == 0.0
    with pytest.raises(ValidationError):
        OnDeviceLearningConfig(regression_tolerance=-0.01)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_held_out_fraction_in_open_zero_to_one(bad: float) -> None:
    with pytest.raises(ValidationError):
        OnDeviceLearningConfig(held_out_fraction=bad)


def test_held_out_fraction_allows_one() -> None:
    assert OnDeviceLearningConfig(held_out_fraction=1.0).held_out_fraction == 1.0


def test_ewc_lambda_allows_zero_but_not_negative() -> None:
    assert OnDeviceLearningConfig(ewc_lambda=0.0).ewc_lambda == 0.0
    with pytest.raises(ValidationError):
        OnDeviceLearningConfig(ewc_lambda=-1.0)


@pytest.mark.parametrize("bad", [0.0, -1e-5])
def test_learning_rate_must_be_positive(bad: float) -> None:
    with pytest.raises(ValidationError):
        OnDeviceLearningConfig(learning_rate=bad)


# --------------------------------------------------------------------------- #
# WS-E0 enablement fields (additive, all defaulted)
# --------------------------------------------------------------------------- #
def test_ws_e0_fields_default_to_off_and_safe() -> None:
    """The enablement fields default to the byte-identical #134 behaviour."""
    cfg = OnDeviceLearningConfig()
    assert cfg.enable_hot_swap is False
    assert cfg.refine_sequence_length == 16
    assert cfg.refine_batch_episodes == 4


def test_enable_hot_swap_requires_enabled() -> None:
    """Hot-swap without the master switch is an operator-actionable config error."""
    with pytest.raises(ValidationError):
        OnDeviceLearningConfig(enable_hot_swap=True, enabled=False)


def test_enable_hot_swap_with_enabled_is_accepted() -> None:
    """Hot-swap is permitted once the master switch is on."""
    cfg = OnDeviceLearningConfig(enable_hot_swap=True, enabled=True)
    assert cfg.enable_hot_swap is True
    assert cfg.enabled is True


@pytest.mark.parametrize("bad", [0, -1])
def test_refine_sequence_length_must_be_positive(bad: int) -> None:
    with pytest.raises(ValidationError):
        OnDeviceLearningConfig(refine_sequence_length=bad)


@pytest.mark.parametrize("bad", [0, -2])
def test_refine_batch_episodes_must_be_positive(bad: int) -> None:
    with pytest.raises(ValidationError):
        OnDeviceLearningConfig(refine_batch_episodes=bad)


# --------------------------------------------------------------------------- #
# Field hygiene
# --------------------------------------------------------------------------- #
def test_every_field_has_a_description() -> None:
    for name, field in OnDeviceLearningConfig.model_fields.items():
        assert field.description, f"{name} must carry an operator description"
