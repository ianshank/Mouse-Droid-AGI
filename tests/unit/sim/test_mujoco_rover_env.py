"""RoverMuJoCoEnv conforms to the protocol and matches the mock obs contract."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mousedroid.config.schema import RoverConfig
from mousedroid.sim.mock_rover_env import MockRoverEnv
from mousedroid.sim.mujoco_rover_env import RoverMuJoCoEnv
from mousedroid.sim.protocols import (
    ROVER_CHASSIS_POSE_DIM,
    ROVER_IMU_DIM,
    ROVER_NUM_WHEELS,
    RoverEnvProtocol,
)

_WHEEL_RADIUS_M = 0.042
_TRACK_WIDTH_M = 0.20


def _mj() -> RoverMuJoCoEnv:
    return RoverMuJoCoEnv(
        RoverConfig(), wheel_radius_m=_WHEEL_RADIUS_M, track_width_m=_TRACK_WIDTH_M
    )


def _mock() -> MockRoverEnv:
    return MockRoverEnv(RoverConfig(), wheel_radius_m=_WHEEL_RADIUS_M, track_width_m=_TRACK_WIDTH_M)


def test_satisfies_protocol() -> None:
    assert isinstance(_mj(), RoverEnvProtocol)


def test_observation_keys_match_mock() -> None:
    assert _mj().observation_keys == _mock().observation_keys


def test_reset_obs_shapes_match_contract() -> None:
    obs, _info = _mj().reset(seed=0)
    assert obs["imu"].shape == (ROVER_IMU_DIM,)
    assert obs["chassis_pose"].shape == (ROVER_CHASSIS_POSE_DIM,)
    assert obs["wheel_vel"].shape == (ROVER_NUM_WHEELS,)
    assert obs["lidar"].shape == (16,)


def test_step_advances_and_is_finite() -> None:
    env = _mj()
    env.reset(seed=0)
    action = np.full((env.action_dim,), 8.0, dtype=np.float32)
    obs, reward, _term, _trunc, info = env.step(action)
    assert np.isfinite(reward)
    assert "vx_body_mps" in info
    for v in obs.values():
        assert np.isfinite(v).all()


def test_lidar_normalised_to_unit_interval() -> None:
    obs, _ = _mj().reset(seed=0)
    lidar = obs["lidar"]
    assert float(lidar.min()) >= 0.0
    assert float(lidar.max()) <= 1.0


def test_spin_in_place_changes_heading() -> None:
    """Skid-steer: opposite wheel commands rotate the chassis (physics sanity)."""
    env = _mj()
    env.reset(seed=0)
    obs: dict[str, np.ndarray] = {}
    for _ in range(80):
        obs = env.step(np.asarray([-8.0, 8.0], dtype=np.float32))[0]
    pose = obs["chassis_pose"]
    heading = float(np.arctan2(pose[3], pose[2]))
    assert abs(heading) > 0.1


def test_invalid_action_shape_raises() -> None:
    env = _mj()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="action shape"):
        env.step(np.zeros(5, dtype=np.float32))


def test_close_is_idempotent() -> None:
    env = _mj()
    env.close()
    env.close()
