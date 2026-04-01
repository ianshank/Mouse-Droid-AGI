"""Trajectory generation and smoothing for robot arm.

Generates smooth joint trajectories between waypoints with
joint limit enforcement and velocity constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmConfig

_log = get_logger(__name__)


class TrajectoryGenerator:
    """Generate smooth trajectories between joint configurations.

    Uses cubic interpolation with velocity constraints and
    joint limit enforcement.

    Args:
        arm_cfg: Arm hardware configuration with joint limits.
    """

    def __init__(self, arm_cfg: ArmConfig) -> None:
        """Initialise trajectory generator.

        Args:
            arm_cfg: Arm config with DOF, velocity limits, joint bounds.
        """
        self._cfg = arm_cfg
        self._max_velocity = arm_cfg.max_joint_velocity_rads
        _log.info("trajectory_generator_init", dof=arm_cfg.dof)

    def interpolate(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        n_steps: int,
    ) -> NDArray[np.float64]:
        """Generate linearly interpolated trajectory between two joint configs.

        Args:
            start: Start joint angles, shape ``(dof,)``.
            end: End joint angles, shape ``(dof,)``.
            n_steps: Number of interpolation steps.

        Returns:
            Trajectory array, shape ``(n_steps, dof)``.
        """
        t = np.linspace(0.0, 1.0, n_steps).reshape(-1, 1)
        trajectory = start + t * (end - start)
        return trajectory

    def smooth(
        self,
        waypoints: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Apply velocity-constrained smoothing to waypoint trajectory.

        Ensures no joint velocity exceeds the configured maximum between
        consecutive waypoints.

        Args:
            waypoints: Joint waypoints, shape ``(N, dof)``.
            dt: Time between waypoints (seconds).

        Returns:
            Smoothed trajectory (may have more points than input).
        """
        if len(waypoints) < 2:
            return waypoints

        smoothed: list[NDArray[np.float64]] = [waypoints[0]]

        for i in range(1, len(waypoints)):
            diff = waypoints[i] - waypoints[i - 1]
            max_joint_delta = float(np.max(np.abs(diff)))
            max_allowed = self._max_velocity * dt

            if max_joint_delta > max_allowed:
                # Need to subdivide this segment
                n_subdivisions = int(np.ceil(max_joint_delta / max_allowed))
                sub_traj = self.interpolate(waypoints[i - 1], waypoints[i], n_subdivisions + 1)
                # Skip first point (already in smoothed)
                for j in range(1, len(sub_traj)):
                    smoothed.append(sub_traj[j])
            else:
                smoothed.append(waypoints[i])

        result = np.stack(smoothed)
        _log.debug(
            "trajectory_smoothed",
            input_points=len(waypoints),
            output_points=len(result),
        )
        return result

    def enforce_limits(
        self,
        trajectory: NDArray[np.float64],
        joint_limits: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Clamp trajectory to joint limits.

        Args:
            trajectory: Joint trajectory, shape ``(N, dof)``.
            joint_limits: Joint limits, shape ``(dof, 2)`` [min, max].
                If None, uses [-pi, pi] for all joints.

        Returns:
            Clamped trajectory.
        """
        if joint_limits is None:
            joint_limits = np.array([[-np.pi, np.pi]] * self._cfg.dof, dtype=np.float64)

        clamped = np.clip(
            trajectory,
            joint_limits[:, 0],
            joint_limits[:, 1],
        )

        violations = int(np.sum(trajectory != clamped))
        if violations > 0:
            _log.warning("joint_limit_violations", count=violations)

        return clamped
