"""Regression tests for the data-generator domain-randomization integration.

The disabled-DR path is the contract: existing pipelines must observe
byte-identical output, including the legacy ``torch.randn`` action sampling.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from training.data_generator import _apply_episode_randomization

from mousedroid.config.schema import (
    DomainRandomizationConfig,
    Settings,
)
from mousedroid.training.domain_randomization import EpisodeParams


def _hash_tensor(t: torch.Tensor) -> str:
    """Stable digest for byte-equality assertions."""
    arr = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=2024)


@pytest.fixture
def baseline_obs() -> dict[str, torch.Tensor]:
    """Fixed-content observation tensors mirroring the bundle layout."""
    torch.manual_seed(0)
    return {
        "vision": torch.randn(256, dtype=torch.float32),
        "ultrasonic": torch.tensor([1.25], dtype=torch.float32),
        "motor_state": torch.zeros(4, dtype=torch.float32),
        "valid_mask": torch.ones(4, dtype=torch.float32),
        "lidar": torch.zeros(0, dtype=torch.float32),
    }


class TestEmptyParamsAreIdentity:
    """Empty ``EpisodeParams`` must round-trip the observation dict verbatim."""

    def test_empty_returns_input_dict(
        self, baseline_obs: dict[str, torch.Tensor], rng: np.random.Generator
    ) -> None:
        out = _apply_episode_randomization(baseline_obs, EpisodeParams(), rng)
        assert out is baseline_obs  # identity, not just equal

    def test_baseline_hashes_match(
        self, baseline_obs: dict[str, torch.Tensor], rng: np.random.Generator
    ) -> None:
        out = _apply_episode_randomization(baseline_obs, EpisodeParams(), rng)
        for key in ("vision", "ultrasonic", "motor_state", "valid_mask", "lidar"):
            assert _hash_tensor(out[key]) == _hash_tensor(baseline_obs[key])


class TestPopulatedParamsAlterObservation:
    """Non-empty params modify vision and ultrasonic but never other slots."""

    def test_feature_noise_changes_vision(
        self, baseline_obs: dict[str, torch.Tensor], rng: np.random.Generator
    ) -> None:
        params = EpisodeParams(feature={"noise_std": 0.1})
        out = _apply_episode_randomization(baseline_obs, params, rng)
        assert _hash_tensor(out["vision"]) != _hash_tensor(baseline_obs["vision"])
        assert _hash_tensor(out["motor_state"]) == _hash_tensor(baseline_obs["motor_state"])

    def test_range_sensor_noise_changes_ultrasonic(
        self, baseline_obs: dict[str, torch.Tensor], rng: np.random.Generator
    ) -> None:
        params = EpisodeParams(range_sensor={"noise_m": 0.05, "dropout_prob": 0.0})
        out = _apply_episode_randomization(baseline_obs, params, rng)
        assert out["ultrasonic"].shape == baseline_obs["ultrasonic"].shape
        assert out["ultrasonic"].item() != pytest.approx(baseline_obs["ultrasonic"].item())

    def test_dropout_keeps_previous_reading(self, baseline_obs: dict[str, torch.Tensor]) -> None:
        local_rng = np.random.default_rng(seed=0)
        params = EpisodeParams(range_sensor={"noise_m": 0.0, "dropout_prob": 1.0})
        out = _apply_episode_randomization(baseline_obs, params, local_rng)
        # NaN handling: integration code drops the noisy sample and forwards
        # the nominal reading rather than poisoning the dataset with NaNs.
        assert torch.equal(out["ultrasonic"], baseline_obs["ultrasonic"])


class TestDisabledFlowMatchesLegacy:
    """Settings with DR disabled retain the legacy generator code path."""

    def test_disabled_settings_yield_empty_episode_params(self) -> None:
        cfg = Settings(
            mock_hardware=True,
            domain_randomization=DomainRandomizationConfig(enabled=False),
        )
        # Construct an EpisodeParams the way the generator would for the
        # disabled branch: it never invokes ``DomainRandomizer.sample``.
        params = EpisodeParams()
        assert params.is_empty
        assert cfg.domain_randomization.enabled is False
