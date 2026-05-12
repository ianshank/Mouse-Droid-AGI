"""Tests for the mock rover environment.

These tests are CI-safe (no GPU / Isaac / MuJoCo deps) and pin the
behaviour the future Isaac Lab backend must replicate.
"""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.config.schema import (
    RoverActionConfig,
    RoverConfig,
    RoverObservationConfig,
    RoverSimConfig,
    Settings,
)
from mousedroid.factory import build_rover_env
from mousedroid.sim.mock_rover_env import MockRoverEnv
from mousedroid.sim.protocols import RoverEnvProtocol


def _make_env(**rover_overrides):
    cfg = RoverConfig(**rover_overrides)
    return MockRoverEnv(cfg, wheel_radius_m=0.042, track_width_m=0.20)


def test_mock_env_implements_protocol():
    env = _make_env()
    assert isinstance(env, RoverEnvProtocol)


def test_reset_returns_dict_and_info():
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    assert info["step_idx"] == 0


def test_default_observation_keys():
    env = _make_env()
    obs, _ = env.reset(seed=0)
    assert set(obs.keys()) == {"imu", "chassis_pose", "wheel_vel", "lidar"}
    assert obs["imu"].shape == (6,)
    assert obs["chassis_pose"].shape == (4,)
    assert obs["wheel_vel"].shape == (4,)
    assert obs["lidar"].shape == (16,)


def test_observation_toggles_drop_keys():
    cfg = RoverConfig(
        observation=RoverObservationConfig(
            include_imu=False,
            include_wheel_encoders=False,
            include_chassis_pose=True,
            include_lidar_sectors=False,
        )
    )
    env = MockRoverEnv(cfg, wheel_radius_m=0.042, track_width_m=0.20)
    obs, _ = env.reset(seed=0)
    assert set(obs.keys()) == {"chassis_pose"}


def test_action_dim_is_two_for_differential():
    env = _make_env()
    assert env.action_dim == 2


def test_step_returns_five_tuple():
    env = _make_env()
    env.reset(seed=0)
    action = np.zeros(2, dtype=np.float32)
    out = env.step(action)
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "distance_to_goal_m" in info


def test_step_action_shape_validation():
    env = _make_env()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="action shape must be"):
        env.step(np.zeros(3, dtype=np.float32))


def test_seed_determinism():
    env1 = _make_env()
    env2 = _make_env()
    _obs1, _ = env1.reset(seed=42)
    _obs2, _ = env2.reset(seed=42)
    action = np.asarray([5.0, 5.0], dtype=np.float32)
    for _ in range(10):
        o1, r1, *_rest1 = env1.step(action)
        o2, r2, *_rest2 = env2.step(action)
        assert np.allclose(o1["chassis_pose"], o2["chassis_pose"])
        assert r1 == r2


def test_forward_command_advances_x():
    env = _make_env()
    env.reset(seed=0)
    action = np.asarray([10.0, 10.0], dtype=np.float32)  # equal wheels -> forward
    for _ in range(30):
        obs, *_rest = env.step(action)
    assert obs["chassis_pose"][0] > 0.0  # x advanced
    assert abs(float(obs["chassis_pose"][1])) < 1e-3  # y stayed put


def test_in_place_turn_advances_theta():
    env = _make_env()
    env.reset(seed=0)
    action = np.asarray([-5.0, 5.0], dtype=np.float32)  # opposite wheels -> spin
    for _ in range(30):
        obs, *_rest = env.step(action)
    cos_t, sin_t = float(obs["chassis_pose"][2]), float(obs["chassis_pose"][3])
    assert abs(cos_t - 1.0) > 1e-2 or abs(sin_t) > 1e-2  # heading changed


def test_wheel_velocity_clamped_to_max():
    env = _make_env()
    env.reset(seed=0)
    huge = np.asarray([1e6, -1e6], dtype=np.float32)
    obs, *_rest = env.step(huge)
    assert float(obs["wheel_vel"].max()) <= 25.0 + 1e-6
    assert float(obs["wheel_vel"].min()) >= -25.0 - 1e-6


def test_truncates_at_episode_length():
    cfg = RoverConfig(sim=RoverSimConfig(episode_length_s=0.1))  # ~3 steps at 30 Hz
    env = MockRoverEnv(cfg, wheel_radius_m=0.042, track_width_m=0.20)
    env.reset(seed=0)
    action = np.zeros(2, dtype=np.float32)
    truncated = False
    for _ in range(20):
        _obs, _r, _term, truncated, _info = env.step(action)
        if truncated:
            break
    assert truncated


def test_body_velocity_mode():
    cfg = RoverConfig(action=RoverActionConfig(mode="body_velocity"))
    env = MockRoverEnv(cfg, wheel_radius_m=0.042, track_width_m=0.20)
    env.reset(seed=0)
    action = np.asarray([0.3, 0.0], dtype=np.float32)  # 0.3 m/s forward
    _obs, _r, _t, _tr, info = env.step(action)
    assert info["vx_body_mps"] == pytest.approx(0.3)
    assert info["omega_rads"] == 0.0


def test_build_rover_env_uses_mock_by_default():
    cfg = Settings(rover=RoverConfig())
    env = build_rover_env(cfg)
    assert isinstance(env, MockRoverEnv)


def test_build_rover_env_requires_rover_block():
    cfg = Settings(rover=None)
    with pytest.raises(ValueError, match="rover config required"):
        build_rover_env(cfg)


def test_build_rover_env_mujoco_not_implemented():
    cfg = Settings(rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")))
    with pytest.raises(NotImplementedError, match="MuJoCo rover backend"):
        build_rover_env(cfg)
