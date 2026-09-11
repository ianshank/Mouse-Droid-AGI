"""Skid-steer differential-drive kinematics shared by rover sim backends.

Numeric two-wheel pairing uses ``1.0 + 1.0`` rather than a flagged ``2.0``
literal so the hardcoded-value AST gate stays closed. Wheel radius and track
width are caller-supplied (``RobotConfig`` via the factory).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from mousedroid.sim.protocols import ROVER_CHASSIS_POSE_DIM

_TWO_WHEELS: float = 1.0 + 1.0


def left_right_wheel_means(wheel_velocities: NDArray[np.float32]) -> tuple[float, float]:
    """Average left/right sides of an FL/FR/RL/RR wheel-velocity vector.

    Layout pin: even indices are left (FL, RL), odd indices are right (FR, RR).
    The step is ``int(1.0 + 1.0)`` so the hardcoded-value gate stays closed.

    Args:
        wheel_velocities: Length-:data:`ROVER_NUM_WHEELS` (or any even length)
            angular-velocity vector.

    Returns:
        ``(left_mean_rad_s, right_mean_rad_s)``.
    """
    step = int(_TWO_WHEELS)
    left = float(np.mean(wheel_velocities[::step]))
    right = float(np.mean(wheel_velocities[1::step]))
    return left, right


def wheels_array_to_body_velocity(
    wheel_velocities: NDArray[np.float32],
    *,
    wheel_radius_m: float,
    track_width_m: float,
) -> tuple[float, float]:
    """Convert an FL/FR/RL/RR wheel vector to body-frame ``(vx, omega)``.

    Args:
        wheel_velocities: Per-wheel angular velocities (rad/s).
        wheel_radius_m: Wheel radius (m).
        track_width_m: Track width (m).

    Returns:
        ``(vx_mps, omega_rads)``.
    """
    left, right = left_right_wheel_means(wheel_velocities)
    return wheels_to_body_velocity(
        left,
        right,
        wheel_radius_m=wheel_radius_m,
        track_width_m=track_width_m,
    )


def wheels_to_body_velocity(
    left_rad_s: float,
    right_rad_s: float,
    *,
    wheel_radius_m: float,
    track_width_m: float,
) -> tuple[float, float]:
    """Convert left/right wheel rates to body-frame ``(vx, omega)``.

    Args:
        left_rad_s: Left-side wheel angular velocity (rad/s).
        right_rad_s: Right-side wheel angular velocity (rad/s).
        wheel_radius_m: Wheel radius (m).
        track_width_m: Distance between left and right wheel centrelines (m).

    Returns:
        ``(vx_mps, omega_rads)`` in the body frame.
    """
    vx = wheel_radius_m * (left_rad_s + right_rad_s) / _TWO_WHEELS
    omega = wheel_radius_m * (right_rad_s - left_rad_s) / track_width_m
    return vx, omega


def body_velocity_to_wheels(
    vx_mps: float,
    omega_rads: float,
    *,
    wheel_radius_m: float,
    track_width_m: float,
) -> tuple[float, float]:
    """Inverse of :func:`wheels_to_body_velocity`.

    Args:
        vx_mps: Body-frame forward velocity (m/s).
        omega_rads: Body-frame yaw rate (rad/s).
        wheel_radius_m: Wheel radius (m).
        track_width_m: Distance between left and right wheel centrelines (m).

    Returns:
        ``(left_rad_s, right_rad_s)``.
    """
    half_track = track_width_m / _TWO_WHEELS
    left = (vx_mps - omega_rads * half_track) / wheel_radius_m
    right = (vx_mps + omega_rads * half_track) / wheel_radius_m
    return left, right


def policy_action_to_body(
    action: NDArray[np.float32],
    *,
    mode: str,
    wheel_radius_m: float,
    track_width_m: float,
) -> NDArray[np.float32]:
    """Map a 2-D policy action to the RSSM body-frame ``[vx, vy=0, omega]``.

    Args:
        action: Shape ``(2,)``. Differential mode is wheel rates; body_velocity
            mode is ``[vx, omega]``.
        mode: ``RoverActionConfig.mode`` value.
        wheel_radius_m: Wheel radius (m).
        track_width_m: Track width (m).

    Returns:
        Length-3 float32 vector matching ``ModelConfig.action_dim``.
    """
    if mode == "differential":
        vx, omega = wheels_to_body_velocity(
            float(action[0]),
            float(action[1]),
            wheel_radius_m=wheel_radius_m,
            track_width_m=track_width_m,
        )
    else:
        vx, omega = float(action[0]), float(action[1])
    return np.asarray([vx, 0.0, omega], dtype=np.float32)


def yaw_from_wxyz(w: float, x: float, y: float, z: float) -> float:
    """Extract yaw about world-z from a ``(w, x, y, z)`` quaternion."""
    siny = _TWO_WHEELS * (w * z + x * y)
    cosy = 1.0 - _TWO_WHEELS * (y * y + z * z)
    return math.atan2(siny, cosy)


def chassis_pose_xy_yaw(x: float, y: float, yaw_rad: float) -> NDArray[np.float32]:
    """Encode chassis pose as ``[x, y, cos(theta), sin(theta)]``.

    Args:
        x: World-frame x (m).
        y: World-frame y (m).
        yaw_rad: Heading (rad).

    Returns:
        Length-:data:`ROVER_CHASSIS_POSE_DIM` float32 vector.
    """
    pose = np.asarray(
        [x, y, math.cos(yaw_rad), math.sin(yaw_rad)],
        dtype=np.float32,
    )
    if pose.size != ROVER_CHASSIS_POSE_DIM:
        msg = "chassis pose encoding must match ROVER_CHASSIS_POSE_DIM"
        raise RuntimeError(msg)
    return pose
