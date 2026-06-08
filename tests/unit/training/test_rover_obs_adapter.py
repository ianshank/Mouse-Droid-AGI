"""RoverObsAdapter maps rover obs dict + info -> RSSM encoder tensors."""

from __future__ import annotations

import numpy as np

from mousedroid.constants import SENSOR_SLOT_MAP
from mousedroid.training.rover_obs_adapter import RoverObsAdapter


def _obs() -> dict[str, np.ndarray]:
    return {
        "imu": np.zeros(6, dtype=np.float32),
        "chassis_pose": np.asarray([0.1, 0.2, 1.0, 0.0], dtype=np.float32),
        "wheel_vel": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "lidar": np.full(16, 0.5, dtype=np.float32),
    }


def test_motor_state_is_vx_vy_omega_battery() -> None:
    adp = RoverObsAdapter(battery_v=12.0)
    out = adp.adapt(_obs(), info={"vx_body_mps": 0.3, "omega_rads": 0.05})
    assert out["motor"].shape == (4,)
    assert out["motor"][1] == 0.0  # vy == 0 (skid-steer)
    assert out["motor"][3] == 12.0  # battery const


def test_vision_omitted_mask_has_vision_slot_zero() -> None:
    adp = RoverObsAdapter(battery_v=12.0)
    out = adp.adapt(_obs(), info={"vx_body_mps": 0.0, "omega_rads": 0.0})
    mask = out["valid_mask"]
    assert mask[SENSOR_SLOT_MAP["vision"]] == 0.0
    assert mask[SENSOR_SLOT_MAP["motor"]] == 1.0
    assert "vision" not in out


def test_lidar_and_range_passed_through() -> None:
    adp = RoverObsAdapter(battery_v=12.0)
    out = adp.adapt(_obs(), info={"vx_body_mps": 0.0, "omega_rads": 0.0})
    assert out["lidar"].shape == (16,)
    assert out["ultrasonic"].shape == (1,)  # min-forward range scalar


def test_vision_features_set_slot_and_key() -> None:
    adp = RoverObsAdapter(battery_v=12.0)
    vf = np.full(256, 0.1, dtype=np.float32)
    out = adp.adapt(_obs(), info={"vx_body_mps": 0.0, "omega_rads": 0.0}, vision_features=vf)
    assert "vision" in out
    assert out["vision"].shape == (256,)
    assert out["valid_mask"][SENSOR_SLOT_MAP["vision"]] == 1.0


def test_no_lidar_yields_full_range_and_no_lidar_key() -> None:
    adp = RoverObsAdapter(battery_v=12.0)
    obs = {"chassis_pose": np.zeros(4, dtype=np.float32)}
    out = adp.adapt(obs, info={})
    assert "lidar" not in out
    assert out["ultrasonic"][0] == 1.0  # no lidar -> full-range default
