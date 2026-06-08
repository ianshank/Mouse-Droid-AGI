"""The skid-steer MJCF loads, exposes the expected sensors, and is stable at rest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MJCF = _REPO_ROOT / "assets" / "rover" / "mse6_4wd.xml"


def test_model_loads() -> None:
    model = mujoco.MjModel.from_xml_path(str(_MJCF))
    assert model.nu == 4  # 4 wheel velocity actuators
    # The base MJCF ships only accel + gyro; RoverMuJoCoEnv injects the lidar ring.
    assert model.nsensor == 2


def test_camera_present() -> None:
    model = mujoco.MjModel.from_xml_path(str(_MJCF))
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rover_cam")
    assert cam_id >= 0  # the forward-facing rover camera exists


def test_rest_state_is_finite() -> None:
    model = mujoco.MjModel.from_xml_path(str(_MJCF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert np.isfinite(data.qacc).all()
    # the chassis should not be free-falling through the floor at rest
    assert abs(float(data.qacc[2])) < 50.0
