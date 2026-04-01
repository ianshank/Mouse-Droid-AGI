"""Tests for Tower of Hanoi Gymnasium environment."""

from __future__ import annotations

import numpy as np

from mousedroid.arm.environments.tower_of_hanoi import TowerOfHanoiEnv
from mousedroid.config.schema import ArmTaskConfig, ArmTrainingConfig


def _make_env(num_disks: int = 3, max_steps: int = 50) -> TowerOfHanoiEnv:
    """Create Tower of Hanoi env with test defaults."""
    task_cfg = ArmTaskConfig(num_disks=num_disks, max_episode_steps=max_steps)
    training_cfg = ArmTrainingConfig()
    return TowerOfHanoiEnv(task_cfg, training_cfg, dof=6)


class TestTowerOfHanoiEnv:
    """Test TowerOfHanoiEnv Gymnasium interface."""

    def test_reset_returns_observation_dict(self) -> None:
        env = _make_env()
        obs, info = env.reset()
        assert "observation" in obs
        assert "achieved_goal" in obs
        assert "desired_goal" in obs
        assert isinstance(info, dict)

    def test_observation_shape(self) -> None:
        env = _make_env(num_disks=3)
        obs, _ = env.reset()
        # obs = 6 joints + 3 disk pegs = 9
        assert obs["observation"].shape == (9,)
        assert obs["achieved_goal"].shape == (3,)
        assert obs["desired_goal"].shape == (3,)

    def test_desired_goal_is_last_peg(self) -> None:
        env = _make_env(num_disks=3)
        obs, _ = env.reset()
        # All disks should target peg 2 (last of 3 pegs)
        np.testing.assert_array_equal(obs["desired_goal"], [2.0, 2.0, 2.0])

    def test_initial_achieved_goal_is_first_peg(self) -> None:
        env = _make_env()
        obs, _ = env.reset()
        # All disks start on peg 0
        np.testing.assert_array_equal(obs["achieved_goal"], [0.0, 0.0, 0.0])

    def test_step_returns_correct_tuple(self) -> None:
        env = _make_env()
        env.reset()
        action = np.zeros(6, dtype=np.float64)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_truncation_after_max_steps(self) -> None:
        env = _make_env(max_steps=5)
        env.reset()
        for _ in range(5):
            _, _, _, truncated, _ = env.step(np.zeros(6))
        assert truncated

    def test_reset_with_seed_deterministic(self) -> None:
        env = _make_env()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1["observation"], obs2["observation"])

    def test_close_no_error(self) -> None:
        env = _make_env()
        env.reset()
        env.close()

    def test_render_returns_none_headless(self) -> None:
        env = _make_env()
        env.reset()
        assert env.render() is None
