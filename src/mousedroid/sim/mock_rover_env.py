"""NumPy-only mock rover environment.

This backend integrates a kinematic differential-drive (or body-velocity)
model with no physics simulator. It exists to:

1. Provide a CI-safe target for ``build_rover_env`` so unit tests pass
   without GPU / Isaac Lab / MuJoCo installed.
2. Pin the observation / action contract that the Isaac Lab and (future)
   MuJoCo backends must replicate.

Roll dynamics, slip, and contact are **not** modelled. The goal is
behavioural fidelity of the env interface, not physics fidelity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.config.schema import RoverConfig
from mousedroid.logging.setup import get_logger
from mousedroid.sim.protocols import (
    ROVER_CHASSIS_POSE_DIM,
    ROVER_IMU_DIM,
    ROVER_NUM_WHEELS,
)

_log = get_logger(__name__)

# Heading wraparound period; pulled out so the wrap math is self-documenting.
_HEADING_WRAP_PERIOD: float = 2.0 * np.pi


class MockRoverEnv:
    """Kinematic-integrator rover env conforming to :class:`RoverEnvProtocol`.

    Differential-drive mode: action is
    ``[left_wheel_rad_s, right_wheel_rad_s]`` and is converted to body
    velocities using the robot's wheel radius and track width
    (:class:`RobotConfig`). Body-velocity mode: action is
    ``[vx_mps, omega_rads]`` and is applied directly.

    Reward is a placeholder ``-||pose - goal_xy_m||`` driven by
    :class:`RoverTaskConfig`. The Phase C reward shaper
    (``mousedroid.training.rover_reward``) replaces it.
    """

    def __init__(self, cfg: RoverConfig, wheel_radius_m: float, track_width_m: float) -> None:
        """Initialise the mock env.

        Args:
            cfg: Rover configuration block from :class:`Settings`.
            wheel_radius_m: Wheel radius from :class:`RobotConfig`.
            track_width_m: Track width from :class:`RobotConfig`.
        """
        self._cfg = cfg
        self._wheel_radius = wheel_radius_m
        self._track_width = track_width_m

        self._control_dt_s = cfg.sim.sim_dt_s * cfg.sim.decimation
        self._max_steps = max(1, int(cfg.sim.episode_length_s / self._control_dt_s))

        self._action_dim = cfg.action.action_dim
        self._goal_xy: NDArray[np.float32] = np.asarray(cfg.task.goal_xy_m, dtype=np.float32)
        self._goal_reach_radius_m = cfg.task.goal_reach_radius_m

        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._wheel_vel: NDArray[np.float32] = np.zeros(ROVER_NUM_WHEELS, dtype=np.float32)
        self._step_idx = 0

        self._obs_keys: tuple[str, ...] = cfg.observation.enabled_keys()
        _log.info(
            "mock_rover_env_initialised",
            mode=cfg.action.mode,
            action_dim=self._action_dim,
            control_dt_s=self._control_dt_s,
            max_steps=self._max_steps,
            goal_xy_m=tuple(float(v) for v in self._goal_xy),
            goal_reach_radius_m=self._goal_reach_radius_m,
        )

    # ----- protocol surface -------------------------------------------------

    @property
    def action_dim(self) -> int:
        """Return the action-space dimensionality (per ``RoverActionConfig.mode``)."""
        return self._action_dim

    @property
    def observation_keys(self) -> tuple[str, ...]:
        """Return the keys present in observation dicts."""
        return self._obs_keys

    def reset(
        self,
        *,
        seed: int | None = None,  # kept for protocol parity (Phase B noise)
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        """Reset the env to ``(0, 0, 0)`` with zero wheel velocities.

        Args:
            seed: Accepted for Gymnasium-protocol parity. The kinematic
                integrator is deterministic, so this argument is a no-op
                until Phase B wires per-episode sensor / dynamics noise.

        Returns:
            ``(observation, info)``.
        """
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._wheel_vel = np.zeros(ROVER_NUM_WHEELS, dtype=np.float32)
        self._step_idx = 0
        return self._observe(), {"step_idx": self._step_idx}

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[
        dict[str, NDArray[np.float32]],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Integrate one control step under ``action``.

        Args:
            action: Shape ``(2,)``; meaning depends on
                :attr:`RoverActionConfig.mode`.

        Returns:
            ``(obs, reward, terminated, truncated, info)``.
        """
        if action.shape != (self._action_dim,):
            msg = f"action shape must be ({self._action_dim},), got {action.shape}"
            raise ValueError(msg)

        vx_body, omega = self._action_to_body_velocity(action)
        self._x += float(vx_body) * np.cos(self._theta) * self._control_dt_s
        self._y += float(vx_body) * np.sin(self._theta) * self._control_dt_s
        new_theta = self._theta + omega * self._control_dt_s
        self._theta = float((new_theta + np.pi) % _HEADING_WRAP_PERIOD - np.pi)

        dx = float(self._goal_xy[0]) - self._x
        dy = float(self._goal_xy[1]) - self._y
        distance = float(np.hypot(dx, dy))
        reward = -distance

        self._step_idx += 1
        truncated = self._step_idx >= self._max_steps
        terminated = distance < self._goal_reach_radius_m

        info: dict[str, Any] = {
            "step_idx": self._step_idx,
            "distance_to_goal_m": distance,
            "vx_body_mps": float(vx_body),
            "omega_rads": float(omega),
        }
        return self._observe(), reward, terminated, truncated, info

    def close(self) -> None:
        """No-op for the mock backend."""

    # ----- internals --------------------------------------------------------

    def _action_to_body_velocity(self, action: NDArray[np.float32]) -> tuple[float, float]:
        """Convert the policy action vector to body-frame ``(vx, omega)``."""
        if self._cfg.action.mode == "differential":
            cap = self._cfg.action.max_wheel_rad_s
            left = float(np.clip(action[0], -cap, cap))
            right = float(np.clip(action[1], -cap, cap))
            # Wheel order: FL, FR, RL, RR (URDF + RoverIsaacLabEnv parity).
            self._wheel_vel = np.asarray([left, right, left, right], dtype=np.float32)
            vx_body = self._wheel_radius * (left + right) / 2.0
            omega = self._wheel_radius * (right - left) / self._track_width
            return vx_body, omega

        # body_velocity mode
        max_v = self._cfg.action.max_wheel_rad_s * self._wheel_radius
        vx_body = float(np.clip(action[0], -max_v, max_v))
        omega = float(action[1])
        # Synthesize wheel velocities for observation parity.
        left = (vx_body - 0.5 * omega * self._track_width) / self._wheel_radius
        right = (vx_body + 0.5 * omega * self._track_width) / self._wheel_radius
        self._wheel_vel = np.asarray([left, right, left, right], dtype=np.float32)
        return vx_body, omega

    def _observe(self) -> dict[str, NDArray[np.float32]]:
        """Produce an observation dict in the configured key order."""
        obs_cfg = self._cfg.observation
        obs: dict[str, NDArray[np.float32]] = {}
        if obs_cfg.include_imu:
            # Kinematic model has no accel/gyro signal — return a zero
            # vector of the IMU contract dimensionality so the obs shape
            # stays stable across backends.
            obs["imu"] = np.zeros(ROVER_IMU_DIM, dtype=np.float32)
        if obs_cfg.include_chassis_pose:
            pose = np.zeros(ROVER_CHASSIS_POSE_DIM, dtype=np.float32)
            pose[0] = self._x
            pose[1] = self._y
            pose[2] = float(np.cos(self._theta))
            pose[3] = float(np.sin(self._theta))
            obs["chassis_pose"] = pose
        if obs_cfg.include_wheel_encoders:
            obs["wheel_vel"] = self._wheel_vel.copy()
        if obs_cfg.include_lidar_sectors:
            obs["lidar"] = np.zeros(obs_cfg.lidar_num_sectors, dtype=np.float32)
        return obs
