from __future__ import annotations

import hashlib

import numpy as np
import torch

from mousedroid.config.schema import Settings
from mousedroid.training.domain_randomization import DomainRandomizer, EpisodeParams
from training.data_generator import _apply_episode_randomization


def _hash_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def _baseline_obs() -> dict[str, torch.Tensor]:
    return {
        "vision": torch.linspace(0.0, 1.0, 32, dtype=torch.float32),
        "ultrasonic": torch.tensor([1.25], dtype=torch.float32),
        "motor_state": torch.tensor([0.1, 0.2, 0.3, 12.0], dtype=torch.float32),
        "valid_mask": torch.ones(5, dtype=torch.float32),
        "lidar": torch.linspace(0.0, 1.0, 16, dtype=torch.float32),
    }


class TestEmptyParamsAreIdentity:
    def test_empty_returns_input_dict(self) -> None:
        baseline_obs = _baseline_obs()
        randomized = _apply_episode_randomization(
            baseline_obs,
            EpisodeParams(),
            np.random.default_rng(0),
        )
        assert randomized is baseline_obs

    def test_baseline_hashes_match(self) -> None:
        baseline_obs = _baseline_obs()
        randomized = _apply_episode_randomization(
            baseline_obs,
            EpisodeParams(),
            np.random.default_rng(0),
        )
        for key in ("vision", "ultrasonic", "motor_state", "valid_mask", "lidar"):
            assert _hash_tensor(randomized[key]) == _hash_tensor(baseline_obs[key])


class TestPopulatedParamsAlterObservation:
    def test_feature_noise_changes_vision(self) -> None:
        baseline_obs = _baseline_obs()
        randomized = _apply_episode_randomization(
            baseline_obs,
            EpisodeParams(feature={"noise_std": 0.1}),
            np.random.default_rng(0),
        )
        assert _hash_tensor(randomized["vision"]) != _hash_tensor(baseline_obs["vision"])
        assert _hash_tensor(randomized["motor_state"]) == _hash_tensor(baseline_obs["motor_state"])


class TestDisabledFlowMatchesLegacy:
    def test_disabled_settings_yield_empty_episode_params(self) -> None:
        cfg = Settings.model_validate(
            {
                "mock_hardware": True,
                "domain_randomization": {"enabled": False},
            }
        )
        params = DomainRandomizer(cfg.domain_randomization).sample(np.random.default_rng(0))
        assert cfg.domain_randomization.enabled is False
        assert params.is_empty is True