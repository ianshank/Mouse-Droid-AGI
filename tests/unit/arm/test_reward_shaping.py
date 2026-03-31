"""Tests for reward shaping functions."""

from __future__ import annotations

import numpy as np

from mousedroid.arm.environments.reward_shaping import RewardShaper
from mousedroid.config.schema import ArmTrainingConfig


def _make_shaper() -> RewardShaper:
    """Create reward shaper with default config."""
    return RewardShaper(ArmTrainingConfig())


class TestRewardShaper:
    """Test RewardShaper reward computation."""

    def test_success_gives_positive_reward(self) -> None:
        shaper = _make_shaper()
        reward = shaper.compute(np.zeros(3), np.zeros(3), {"is_success": True})
        assert reward > 0.0

    def test_collision_gives_negative_reward(self) -> None:
        shaper = _make_shaper()
        reward = shaper.compute(np.zeros(3), np.zeros(3), {"collision": True})
        assert reward < 0.0

    def test_grasp_gives_positive_reward(self) -> None:
        shaper = _make_shaper()
        reward = shaper.compute(np.zeros(3), np.zeros(3), {"grasp_success": True})
        assert reward > 0.0

    def test_place_gives_positive_reward(self) -> None:
        shaper = _make_shaper()
        reward = shaper.compute(np.zeros(3), np.zeros(3), {"place_correct": True})
        assert reward > 0.0

    def test_wrong_disk_gives_negative_reward(self) -> None:
        shaper = _make_shaper()
        reward = shaper.compute(np.zeros(3), np.zeros(3), {"wrong_disk": True})
        assert reward < 0.0

    def test_empty_info_gives_zero_reward(self) -> None:
        shaper = _make_shaper()
        reward = shaper.compute(np.zeros(3), np.zeros(3), {})
        # Only distance component (zero since same goal)
        assert abs(reward) < 0.01

    def test_distance_penalty_increases_with_distance(self) -> None:
        shaper = _make_shaper()
        close = shaper.compute(np.array([0.1, 0, 0]), np.zeros(3), {})
        far = shaper.compute(np.array([1.0, 0, 0]), np.zeros(3), {})
        assert far < close

    def test_batch_computation(self) -> None:
        shaper = _make_shaper()
        achieved = np.zeros((3, 2))
        desired = np.ones((3, 2))
        infos = [{"is_success": True}, {"collision": True}, {}]
        rewards = shaper.compute_batch(achieved, desired, infos)
        assert rewards.shape == (3,)
        assert rewards[0] > rewards[1]  # Success > collision
