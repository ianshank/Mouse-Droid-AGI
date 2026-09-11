"""Shared skid-steer kinematics used by MuJoCo and Isaac backends."""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.sim.kinematics import (
    body_velocity_to_wheels,
    left_right_wheel_means,
    policy_action_to_body,
    wheels_array_to_body_velocity,
    wheels_to_body_velocity,
    yaw_from_wxyz,
)


def test_straight_differential_has_zero_yaw_rate() -> None:
    vx, omega = wheels_to_body_velocity(10.0, 10.0, wheel_radius_m=0.042, track_width_m=0.20)
    assert omega == pytest.approx(0.0)
    assert vx == pytest.approx(0.042 * 10.0)


def test_opposite_wheels_are_pure_rotation() -> None:
    vx, omega = wheels_to_body_velocity(-5.0, 5.0, wheel_radius_m=0.05, track_width_m=0.2)
    assert vx == pytest.approx(0.0)
    assert omega == pytest.approx(0.05 * 10.0 / 0.2)


def test_body_to_wheels_inverts() -> None:
    left, right = 3.0, 7.0
    vx, omega = wheels_to_body_velocity(left, right, wheel_radius_m=0.042, track_width_m=0.20)
    left2, right2 = body_velocity_to_wheels(vx, omega, wheel_radius_m=0.042, track_width_m=0.20)
    assert left2 == pytest.approx(left)
    assert right2 == pytest.approx(right)


def test_policy_action_differential_pads_vy() -> None:
    action = np.array([4.0, 6.0], dtype=np.float32)
    body = policy_action_to_body(
        action, mode="differential", wheel_radius_m=0.042, track_width_m=0.20
    )
    assert body.shape == (3,)
    assert body[1] == pytest.approx(0.0)


def test_left_right_means_alternating_layout() -> None:
    wheels = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)
    left, right = left_right_wheel_means(wheels)
    assert left == pytest.approx(1.0)
    assert right == pytest.approx(2.0)


def test_wheels_array_to_body_matches_pair() -> None:
    wheels = np.array([2.0, 4.0, 2.0, 4.0], dtype=np.float32)
    vx, omega = wheels_array_to_body_velocity(wheels, wheel_radius_m=0.042, track_width_m=0.20)
    vx2, omega2 = wheels_to_body_velocity(2.0, 4.0, wheel_radius_m=0.042, track_width_m=0.20)
    assert vx == pytest.approx(vx2)
    assert omega == pytest.approx(omega2)


def test_yaw_from_wxyz_identity_and_quarter_turn() -> None:
    assert yaw_from_wxyz(1.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)
    yaw = np.pi / 2
    w = float(np.cos(yaw / 2))
    z = float(np.sin(yaw / 2))
    assert yaw_from_wxyz(w, 0.0, 0.0, z) == pytest.approx(yaw)
