"""Tests for trajectory generation and smoothing."""

from __future__ import annotations

import numpy as np

from mousedroid.arm.control.trajectory import TrajectoryGenerator
from mousedroid.config.schema import ArmConfig


def _make_generator(dof: int = 6, max_vel: float = 2.0) -> TrajectoryGenerator:
    """Create trajectory generator with test defaults."""
    cfg = ArmConfig(
        dof=dof,
        max_joint_velocity_rads=max_vel,
        home_position=[0.0] * dof,
    )
    return TrajectoryGenerator(cfg)


class TestTrajectoryGenerator:
    """Test TrajectoryGenerator methods."""

    def test_interpolate_shape(self) -> None:
        gen = _make_generator()
        start = np.zeros(6)
        end = np.ones(6)
        traj = gen.interpolate(start, end, n_steps=10)
        assert traj.shape == (10, 6)

    def test_interpolate_starts_at_start(self) -> None:
        gen = _make_generator()
        start = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        end = np.ones(6)
        traj = gen.interpolate(start, end, n_steps=10)
        np.testing.assert_allclose(traj[0], start)

    def test_interpolate_ends_at_end(self) -> None:
        gen = _make_generator()
        start = np.zeros(6)
        end = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        traj = gen.interpolate(start, end, n_steps=10)
        np.testing.assert_allclose(traj[-1], end)

    def test_smooth_subdivides_fast_segments(self) -> None:
        gen = _make_generator(max_vel=1.0)
        # Large jump that exceeds velocity limit at dt=0.01
        waypoints = np.array([[0.0] * 6, [1.0] * 6])
        smoothed = gen.smooth(waypoints, dt=0.01)
        assert len(smoothed) > 2  # Must subdivide

    def test_smooth_preserves_slow_segments(self) -> None:
        gen = _make_generator(max_vel=100.0)
        waypoints = np.array([[0.0] * 6, [0.01] * 6])
        smoothed = gen.smooth(waypoints, dt=1.0)
        assert len(smoothed) == 2  # No subdivision needed

    def test_smooth_single_waypoint(self) -> None:
        gen = _make_generator()
        waypoints = np.array([[0.0] * 6])
        smoothed = gen.smooth(waypoints, dt=0.01)
        assert len(smoothed) == 1

    def test_enforce_limits_clamps(self) -> None:
        gen = _make_generator()
        traj = np.array([[10.0] * 6])  # Way beyond pi
        limits = np.array([[-np.pi, np.pi]] * 6)
        clamped = gen.enforce_limits(traj, limits)
        assert np.all(clamped <= np.pi)
        assert np.all(clamped >= -np.pi)

    def test_enforce_limits_default_pi(self) -> None:
        gen = _make_generator()
        traj = np.array([[5.0] * 6])
        clamped = gen.enforce_limits(traj)
        assert np.all(clamped <= np.pi)

    def test_enforce_limits_no_change_within_bounds(self) -> None:
        gen = _make_generator()
        traj = np.array([[0.5, -0.5, 1.0, -1.0, 0.0, 0.1]])
        clamped = gen.enforce_limits(traj)
        np.testing.assert_array_equal(traj, clamped)
