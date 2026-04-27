"""Backwards-compatibility regression tests for Phase 1 domain randomization.

These guarantees protect existing users from silent regressions when the
Phase 1 domain-randomization feature lands:

* Every committed YAML file loads without validation error after the new
  ``domain_randomization`` section + ``RangeF`` model were introduced.
* YAMLs that omit ``domain_randomization`` continue to load (defaults apply).
* The ``DomainRandomizationConfig`` defaults match the documented values, so
  hash-pinned downstream artefacts do not silently shift.
* The synthetic data generator's disabled-DR code path is byte-identical to
  the legacy implementation: ``_apply_episode_randomization`` returns the
  input dict by identity when given empty :class:`EpisodeParams`.
* The ``Settings.domain_randomization`` field is wired through the root
  Pydantic Settings model and round-trips via ``model_dump`` /
  ``model_validate``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from training.data_generator import _apply_episode_randomization

from mousedroid.config.schema import (
    DomainRandomizationConfig,
    RangeF,
    Settings,
)
from mousedroid.training.domain_randomization import (
    DomainRandomizer,
    EpisodeParams,
)

_CONFIG_DIR = Path("config")

# Every committed mouse-droid YAML must continue to load unchanged after the
# DomainRandomizationConfig field was added to Settings.
_MOUSE_DROID_YAMLS = [
    "default.yaml",
    "mock_hardware.yaml",
    "jetson_production.yaml",
    "jetson_hailo.yaml",
    "jetson_secure_metrics.yaml",
    "local_training.yaml",
]


# ---------------------------------------------------------------------------
# YAML load hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", _MOUSE_DROID_YAMLS)
def test_yaml_loads_with_dr_field(filename: str) -> None:
    """Every committed mouse-droid YAML loads with the new DR field present."""
    path = _CONFIG_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not present in this checkout")
    data = yaml.safe_load(path.read_text())
    settings = Settings.model_validate(data)
    assert settings.domain_randomization is not None
    assert isinstance(settings.domain_randomization, DomainRandomizationConfig)


def test_yaml_without_domain_randomization_section_still_loads() -> None:
    """YAML files that omit ``domain_randomization`` retain default behaviour."""
    settings = Settings.model_validate({"mock_hardware": True})
    assert settings.domain_randomization.enabled is True  # Schema default


# ---------------------------------------------------------------------------
# DomainRandomizationConfig — pinned defaults (changing these is a breaking
# change and must update this test deliberately).
# ---------------------------------------------------------------------------


def test_dr_config_default_enabled() -> None:
    """Schema default has DR enabled — opt-out via YAML or env var."""
    cfg = DomainRandomizationConfig()
    assert cfg.enabled is True


def test_dr_config_default_visual_ranges_pinned() -> None:
    """Visual range defaults must remain stable across versions."""
    cfg = DomainRandomizationConfig()
    assert cfg.brightness.low == pytest.approx(0.6)
    assert cfg.brightness.high == pytest.approx(1.4)
    assert cfg.contrast.low == pytest.approx(0.7)
    assert cfg.contrast.high == pytest.approx(1.3)
    assert cfg.gaussian_noise_std.low == pytest.approx(0.0)
    assert cfg.gaussian_noise_std.high == pytest.approx(0.04)


def test_dr_config_default_chassis_ranges_pinned() -> None:
    """Chassis dynamics ranges must remain stable across versions."""
    cfg = DomainRandomizationConfig()
    assert cfg.wheel_friction.low == pytest.approx(0.7)
    assert cfg.wheel_friction.high == pytest.approx(1.3)
    assert cfg.motor_gain.low == pytest.approx(0.85)
    assert cfg.motor_gain.high == pytest.approx(1.15)


def test_dr_config_default_sensor_ranges_pinned() -> None:
    """Range-sensor noise defaults must remain stable across versions."""
    cfg = DomainRandomizationConfig()
    assert cfg.ultrasonic_noise_m.low == pytest.approx(0.0)
    assert cfg.ultrasonic_noise_m.high == pytest.approx(0.03)
    assert cfg.ultrasonic_dropout_prob.high == pytest.approx(0.05)


def test_dr_config_default_push_event_prob_pinned() -> None:
    cfg = DomainRandomizationConfig()
    assert cfg.push_event_prob == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# RangeF validation invariants
# ---------------------------------------------------------------------------


def test_range_f_rejects_inverted_bounds() -> None:
    """RangeF must reject ``low > high`` to fail-fast in config validation."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RangeF(low=2.0, high=1.0)


def test_range_f_allows_degenerate_range() -> None:
    """RangeF must accept ``low == high`` (sampler returns the constant)."""
    r = RangeF(low=0.5, high=0.5)
    assert r.low == r.high == 0.5


# ---------------------------------------------------------------------------
# Disabled-DR code path is byte-identical to legacy
# ---------------------------------------------------------------------------


def test_disabled_dr_returns_empty_episode_params() -> None:
    """``DomainRandomizer.sample`` with ``enabled=False`` returns the empty sentinel."""
    dr = DomainRandomizer(DomainRandomizationConfig(enabled=False))
    params = dr.sample(np.random.default_rng(seed=0))
    assert params.is_empty


def test_apply_episode_randomization_identity_for_empty_params() -> None:
    """Empty ``EpisodeParams`` round-trips the observation dict by identity.

    This is the legacy-behaviour contract: a future maintainer who inadvertently
    breaks this guarantee will see this test fail loudly with a clear message.
    """
    obs = {
        "vision": torch.randn(256, dtype=torch.float32),
        "ultrasonic": torch.tensor([1.0], dtype=torch.float32),
        "motor_state": torch.zeros(4, dtype=torch.float32),
        "valid_mask": torch.ones(4, dtype=torch.float32),
        "lidar": torch.zeros(0, dtype=torch.float32),
    }
    out = _apply_episode_randomization(obs, EpisodeParams(), np.random.default_rng(seed=0))
    assert out is obs, (
        "_apply_episode_randomization MUST return the input dict by identity "
        "when ep_params is empty (legacy byte-identical contract)"
    )


# ---------------------------------------------------------------------------
# Settings round-trip
# ---------------------------------------------------------------------------


def test_settings_dr_round_trip_preserves_overrides() -> None:
    """``Settings`` round-trips via ``model_dump`` / ``model_validate``."""
    overrides = {
        "mock_hardware": True,
        "domain_randomization": {
            "enabled": False,
            "brightness": {"low": 0.8, "high": 1.2},
        },
    }
    settings = Settings.model_validate(overrides)
    dumped = settings.model_dump()
    restored = Settings.model_validate(dumped)
    assert restored.domain_randomization.enabled is False
    assert restored.domain_randomization.brightness.low == pytest.approx(0.8)
    assert restored.domain_randomization.brightness.high == pytest.approx(1.2)


def test_settings_env_var_disables_dr(monkeypatch: pytest.MonkeyPatch) -> None:
    """``MOUSEDROID_DOMAIN_RANDOMIZATION__ENABLED=false`` disables DR."""
    monkeypatch.setenv("MOUSEDROID_DOMAIN_RANDOMIZATION__ENABLED", "false")
    settings = Settings(mock_hardware=True)
    assert settings.domain_randomization.enabled is False
