"""AQA — F-044 Isaac sensor + training-seam hygiene."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from mousedroid.constants import SENSOR_SLOT_MAP
from mousedroid.sim.isaaclab.rover_env import RoverIsaacLabEnv
from mousedroid.sim.isaaclab.sensors import read_rover_imu, read_rover_lidar, read_rover_pose
from mousedroid.training.rover_obs_adapter import RoverObsAdapter


def test_readers_are_public_and_documented() -> None:
    for fn in (read_rover_imu, read_rover_pose, read_rover_lidar):
        doc = inspect.getdoc(fn)
        assert doc
        assert "LidarConfig" not in doc


def test_env_exposes_to_body_action() -> None:
    assert callable(RoverIsaacLabEnv.to_body_action)
    doc = inspect.getdoc(RoverIsaacLabEnv.to_body_action)
    assert doc
    assert "imu_dim" not in doc.lower() or "not" in doc.lower()


def test_env_exposes_inject_sensor() -> None:
    assert callable(RoverIsaacLabEnv.inject_sensor)
    doc = inspect.getdoc(RoverIsaacLabEnv.inject_sensor)
    assert doc
    assert "zeros" in doc.lower()


def test_adapter_still_zeros_rssm_imu_slot() -> None:
    adapter = RoverObsAdapter(battery_v=12.0)
    out = adapter.adapt(
        {"lidar": np.zeros(16, dtype=np.float32)},
        {"vx_body_mps": 0.1, "omega_rads": 0.0},
    )
    assert out["valid_mask"][SENSOR_SLOT_MAP["imu"]] == 0.0
    assert "imu" not in out


def test_constants_do_not_claim_imusensor_attach() -> None:
    text = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mousedroid"
        / "sim"
        / "isaaclab"
        / "constants.py"
    ).read_text(encoding="utf-8")
    assert "IMUSensorCfg" not in text
    assert "CameraSensorCfg" not in text
    assert "duck-typed readers" in text
