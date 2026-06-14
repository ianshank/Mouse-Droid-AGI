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
    """The default slot_dir must be repo-relative — resolved under experience root."""
    cfg = OnDeviceLearningConfig()
    assert not cfg.slot_dir.startswith("/")
    assert "/home/jetson" not in cfg.slot_dir


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
# Field hygiene
# --------------------------------------------------------------------------- #
def test_every_field_has_a_description() -> None:
    for name, field in OnDeviceLearningConfig.model_fields.items():
        assert field.description, f"{name} must carry an operator description"
