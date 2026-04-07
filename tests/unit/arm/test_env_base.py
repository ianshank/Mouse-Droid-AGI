"""Tests for ArmEnvironmentBase shared behaviour."""

from __future__ import annotations

import numpy as np

from mousedroid.arm.environments.laundry_sorting import LaundrySortingEnv
from mousedroid.arm.environments.tower_of_hanoi import TowerOfHanoiEnv
from mousedroid.config.schema import ArmTaskConfig, ArmTrainingConfig


def _make_hanoi() -> TowerOfHanoiEnv:
    task_cfg = ArmTaskConfig(num_disks=2, max_episode_steps=10)
    training_cfg = ArmTrainingConfig()
    return TowerOfHanoiEnv(task_cfg, training_cfg)


def _make_laundry() -> LaundrySortingEnv:
    task_cfg = ArmTaskConfig(task_type="laundry_sorting", max_episode_steps=10, num_garments=3)
    training_cfg = ArmTrainingConfig()
    return LaundrySortingEnv(task_cfg, training_cfg)


class TestBaseClassSharedBehaviour:
    """Verify that base class provides consistent behaviour across envs."""

    def test_both_envs_reset_returns_obs_dict(self) -> None:
        for env in [_make_hanoi(), _make_laundry()]:
            obs, info = env.reset()
            assert "observation" in obs
            assert "achieved_goal" in obs
            assert "desired_goal" in obs
            assert "is_success" in info

    def test_both_envs_step_returns_five_tuple(self) -> None:
        for env in [_make_hanoi(), _make_laundry()]:
            env.reset()
            action = np.zeros(6, dtype=np.float64)
            result = env.step(action)
            assert len(result) == 5

    def test_truncation_consistent(self) -> None:
        for env in [_make_hanoi(), _make_laundry()]:
            env.reset()
            action = np.zeros(6, dtype=np.float64)
            for _ in range(9):
                _, _, _, truncated, _ = env.step(action)
                assert truncated is False
            _, _, _, truncated, _ = env.step(action)
            assert truncated is True

    def test_render_returns_none(self) -> None:
        for env in [_make_hanoi(), _make_laundry()]:
            assert env.render() is None

    def test_close_no_error(self) -> None:
        for env in [_make_hanoi(), _make_laundry()]:
            env.close()

    def test_action_clipping_uses_config(self) -> None:
        training_cfg = ArmTrainingConfig(action_delta_min=-0.01, action_delta_max=0.01)
        task_cfg = ArmTaskConfig(num_disks=2, max_episode_steps=100)
        env = TowerOfHanoiEnv(task_cfg, training_cfg)
        env.reset()
        large_action = np.ones(6, dtype=np.float64)
        env.step(large_action)
        # Joint angles should be at most 0.01 after one step from zero
        _obs, _ = env.reset()
        env.step(large_action)
        # Joints should have moved by exactly action_delta_max
        np.testing.assert_allclose(env._joint_angles, np.full(6, 0.01), atol=1e-10)

    def test_seed_deterministic(self) -> None:
        env = _make_hanoi()
        obs1, _ = env.reset(seed=99)
        obs2, _ = env.reset(seed=99)
        np.testing.assert_array_equal(obs1["observation"], obs2["observation"])
