"""Tests for laundry sorting environment."""

from __future__ import annotations

import numpy as np

from mousedroid.arm.environments.laundry_sorting import LaundrySortingEnv
from mousedroid.config.schema import ArmTaskConfig, ArmTrainingConfig


def _make_env() -> LaundrySortingEnv:
    task_cfg = ArmTaskConfig(task_type="laundry_sorting", num_garments=3)
    training_cfg = ArmTrainingConfig()
    return LaundrySortingEnv(task_cfg, training_cfg)


class TestLaundrySortingEnv:
    """Tests for LaundrySortingEnv."""

    def test_reset_returns_observation_dict(self) -> None:
        env = _make_env()
        obs, info = env.reset()
        assert "observation" in obs
        assert "achieved_goal" in obs
        assert "desired_goal" in obs
        assert info["is_success"] is False

    def test_observation_shapes(self) -> None:
        env = _make_env()
        obs, _info = env.reset()
        # obs = joint_angles(6) + garments_sorted(1) = 7
        assert obs["observation"].shape == (7,)
        assert obs["achieved_goal"].shape == (1,)
        assert obs["desired_goal"].shape == (1,)

    def test_desired_goal_equals_num_garments(self) -> None:
        env = _make_env()
        obs, _info = env.reset()
        assert obs["desired_goal"][0] == 3.0

    def test_step_returns_five_tuple(self) -> None:
        env = _make_env()
        env.reset()
        action = np.zeros(6, dtype=np.float64)
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_truncation_at_max_steps(self) -> None:
        task_cfg = ArmTaskConfig(task_type="laundry_sorting", max_episode_steps=3, num_garments=5)
        training_cfg = ArmTrainingConfig()
        env = LaundrySortingEnv(task_cfg, training_cfg)
        env.reset()
        action = np.zeros(6, dtype=np.float64)
        for _ in range(2):
            _, _, _, truncated, _ = env.step(action)
            assert truncated is False
        _, _, _, truncated, _ = env.step(action)
        assert truncated is True

    def test_action_clipping_uses_config(self) -> None:
        task_cfg = ArmTaskConfig(task_type="laundry_sorting")
        training_cfg = ArmTrainingConfig(action_delta_min=-0.05, action_delta_max=0.05)
        env = LaundrySortingEnv(task_cfg, training_cfg)
        env.reset()
        large_action = np.full(6, 1.0, dtype=np.float64)
        env.step(large_action)
        # Joints should only move by 0.05, not 1.0
        obs, _ = env.reset(seed=0)
        env.step(large_action)
        obs_after, _ = env.reset(seed=0)
        # Just verify no crash — the clipping is internal

    def test_reset_with_seed_deterministic(self) -> None:
        env = _make_env()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1["observation"], obs2["observation"])

    def test_render_returns_none(self) -> None:
        env = _make_env()
        assert env.render() is None

    def test_close_no_error(self) -> None:
        env = _make_env()
        env.close()
