from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError
from training.data_generator import (
    _apply_episode_randomization,
)

from mousedroid.config.schema import DomainRandomizationConfig, RangeF, Settings
from mousedroid.training.domain_randomization import DomainRandomizer, EpisodeParams

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "config_name",
    [
        "default.yaml",
        "mock_hardware.yaml",
        "jetson_production.yaml",
        "jetson_hailo.yaml",
        "jetson_secure_metrics.yaml",
        "local_training.yaml",
    ],
)
def test_yaml_loads_with_dr_field(config_name: str) -> None:
    config_path = _REPO_ROOT / "config" / config_name
    if not config_path.exists():
        pytest.skip(f"missing config fixture: {config_name}")

    settings = Settings.model_validate(yaml.safe_load(config_path.read_text()) or {})

    assert isinstance(settings.domain_randomization, DomainRandomizationConfig)


def test_yaml_without_domain_randomization_section_still_loads() -> None:
    settings = Settings.model_validate({"mock_hardware": True})
    assert settings.domain_randomization.enabled is True


def test_dr_config_default_enabled() -> None:
    assert DomainRandomizationConfig().enabled is True


def test_dr_config_default_visual_ranges_pinned() -> None:
    cfg = DomainRandomizationConfig()
    assert cfg.brightness.low == 0.6
    assert cfg.brightness.high == 1.4
    assert cfg.contrast.low == 0.7
    assert cfg.contrast.high == 1.3
    assert cfg.gaussian_noise_std.low == 0.0
    assert cfg.gaussian_noise_std.high == 0.04


def test_dr_config_default_chassis_ranges_pinned() -> None:
    cfg = DomainRandomizationConfig()
    assert cfg.wheel_friction.low == 0.7
    assert cfg.wheel_friction.high == 1.3
    assert cfg.motor_gain.low == 0.85
    assert cfg.motor_gain.high == 1.15


def test_dr_config_default_comms_ranges_pinned() -> None:
    cfg = DomainRandomizationConfig()
    assert cfg.uart_latency_ms.low == 2.0
    assert cfg.uart_latency_ms.high == 18.0
    assert cfg.encoder_dropout_prob.high == 0.02


def test_dr_config_default_push_event_prob_pinned() -> None:
    assert DomainRandomizationConfig().push_event_prob == 0.05


def test_range_f_rejects_inverted_bounds() -> None:
    with pytest.raises(ValidationError):
        RangeF(low=2.0, high=1.0)


def test_range_f_allows_degenerate_range() -> None:
    cfg = RangeF(low=0.5, high=0.5)
    assert cfg.low == 0.5
    assert cfg.high == 0.5


def test_disabled_dr_returns_empty_episode_params() -> None:
    params = DomainRandomizer(DomainRandomizationConfig(enabled=False)).sample(
        np.random.default_rng(0)
    )
    assert params.is_empty is True


def test_apply_episode_randomization_identity_for_empty_params() -> None:
    obs = {
        "vision": torch.ones(8, dtype=torch.float32),
        "ultrasonic": torch.tensor([1.0], dtype=torch.float32),
        "motor_state": torch.ones(4, dtype=torch.float32),
        "valid_mask": torch.ones(5, dtype=torch.float32),
        "lidar": torch.ones(4, dtype=torch.float32),
    }
    randomized = _apply_episode_randomization(obs, EpisodeParams(), np.random.default_rng(0))
    assert randomized is obs, "identity is the legacy contract for empty episode params"


def test_settings_dr_round_trip_preserves_overrides() -> None:
    settings = Settings.model_validate(
        {
            "mock_hardware": True,
            "domain_randomization": {
                "enabled": False,
                "brightness": {"low": 0.8, "high": 1.2},
            },
        }
    )

    round_tripped = Settings.model_validate(settings.model_dump())
    assert round_tripped.domain_randomization.enabled is False
    assert round_tripped.domain_randomization.brightness.low == 0.8
    assert round_tripped.domain_randomization.brightness.high == 1.2


def test_settings_env_var_disables_dr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOUSEDROID_DOMAIN_RANDOMIZATION__ENABLED", "false")
    settings = Settings(mock_hardware=True)
    assert settings.domain_randomization.enabled is False
