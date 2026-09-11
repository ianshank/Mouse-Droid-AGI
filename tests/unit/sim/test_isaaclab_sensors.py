"""Duck-typed IMU / pose / LiDAR readers (always-on; no isaaclab)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from mousedroid.sim.isaaclab.constants import ROVER_SENSOR_LINK_NAMES
from mousedroid.sim.isaaclab.sensors import (
    identity_chassis_pose,
    read_rover_imu,
    read_rover_lidar,
    read_rover_pose,
    to_numpy,
)
from mousedroid.sim.protocols import ROVER_CHASSIS_POSE_DIM, ROVER_IMU_DIM


def test_identity_pose_is_unit_heading() -> None:
    pose = identity_chassis_pose()
    assert pose.shape == (ROVER_CHASSIS_POSE_DIM,)
    assert pose[0] == pytest.approx(0.0)
    assert pose[1] == pytest.approx(0.0)
    assert pose[2] ** 2 + pose[3] ** 2 == pytest.approx(1.0)


def test_read_pose_from_articulation_root() -> None:
    yaw = np.pi / 2
    quat = np.array([[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]], dtype=np.float32)
    art = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=np.array([[1.5, -0.25, 0.0]], dtype=np.float32),
            root_quat_w=quat,
        )
    )
    pose = read_rover_pose(sensors={}, articulation=art)
    assert pose[0] == pytest.approx(1.5)
    assert pose[1] == pytest.approx(-0.25)
    assert pose[2] == pytest.approx(np.cos(yaw), abs=1e-5)
    assert pose[3] == pytest.approx(np.sin(yaw), abs=1e-5)


def test_read_pose_missing_is_identity() -> None:
    pose = read_rover_pose(sensors={}, articulation=None)
    np.testing.assert_array_equal(pose, identity_chassis_pose())


def test_read_imu_concatenates_lin_and_ang() -> None:
    handle = SimpleNamespace(
        data=SimpleNamespace(
            lin_acc_b=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            ang_vel_b=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        )
    )
    imu = read_rover_imu(
        sensors={ROVER_SENSOR_LINK_NAMES[0]: handle},
        articulation=None,
    )
    assert imu.shape == (ROVER_IMU_DIM,)
    np.testing.assert_allclose(imu, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])


def test_read_imu_zeros_when_missing() -> None:
    imu = read_rover_imu(sensors={}, articulation=None)
    np.testing.assert_array_equal(imu, np.zeros(ROVER_IMU_DIM, dtype=np.float32))


def test_read_lidar_resamples_and_normalises() -> None:
    raw = np.array([0.0, 2.0, 4.0, 8.0], dtype=np.float32)
    handle = SimpleNamespace(data=SimpleNamespace(ray_distance=raw))
    lidar = read_rover_lidar(
        sensors={ROVER_SENSOR_LINK_NAMES[1]: handle},
        n_sectors=2,
        max_range_m=4.0,
    )
    assert lidar.shape == (2,)
    assert float(lidar.max()) <= 1.0 + 1e-6
    assert float(lidar.min()) >= 0.0


def test_read_lidar_missing_is_zeros() -> None:
    lidar = read_rover_lidar(sensors={}, n_sectors=16, max_range_m=4.0)
    np.testing.assert_array_equal(lidar, np.zeros(16, dtype=np.float32))


def test_read_lidar_from_hit_points() -> None:
    hits = np.array([[[4.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=np.float32)
    origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    handle = SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits, pos_w=origin))
    lidar = read_rover_lidar(
        sensors={ROVER_SENSOR_LINK_NAMES[1]: handle},
        n_sectors=2,
        max_range_m=4.0,
    )
    assert lidar.shape == (2,)
    assert lidar[0] == pytest.approx(1.0)
    assert lidar[1] == pytest.approx(0.5)


def test_to_numpy_accepts_detach_cpu_numpy_chain() -> None:
    class _FakeTensor:
        def detach(self) -> _FakeTensor:
            return self

        def cpu(self) -> _FakeTensor:
            return self

        def numpy(self) -> NDArray[np.float32]:
            return np.array([1.0, 2.0], dtype=np.float32)

    arr = to_numpy(_FakeTensor())
    assert arr is not None
    np.testing.assert_array_equal(arr, np.array([1.0, 2.0], dtype=np.float32))
