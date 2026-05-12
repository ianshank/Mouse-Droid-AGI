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

_log = get_logger(__name__)


class MockRoverEnv:
    """Kinematic-integrator rover env conforming to :class:`RoverEnvProtocol`.

    Differential-drive mode: action is
    ``[left_wheel_rad_s, right_wheel_rad_s]`` and is converted to body
    velocities using the robot's wheel radius and track width
    (:class:`RobotConfig`). Body-velocity mode: action is
    ``[vx_mps, omega_rads]`` and is applied directly.

    Reward is a placeholder ``-||pose - goal_pose||`` with a fixed goal
    at ``(2.0, 0.0)``. The Phase C reward shaper will replace this.
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

        self._action_dim = 2
        self._goal_xy: NDArray[np.float32] = np.asarray([2.0, 0.0], dtype=np.float32)

        # State: x, y, theta, vx_body, omega, wheel_velocities (4,)
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._wheel_vel: NDArray[np.float32] = np.zeros(4, dtype=np.float32)
        self._step_idx = 0
        self._rng: np.random.Generator = np.random.default_rng()

        self._obs_keys: tuple[str, ...] = self._compute_observation_keys()
        _log.info(
            "mock_rover_env_initialised",
            mode=cfg.action.mode,
            control_dt_s=self._control_dt_s,
            max_steps=self._max_steps,
        )

    # ----- protocol surface -------------------------------------------------

    @property
    def action_dim(self) -> int:
        """Return the action-space dimensionality (always 2)."""
        return self._action_dim

    @property
    def observation_keys(self) -> tuple[str, ...]:
        """Return the keys present in observation dicts."""
        return self._obs_keys

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        """Reset the env to ``(0, 0, 0)`` with zero wheel velocities.

        Args:
            seed: Optional RNG seed; identical seeds yield identical
                trajectories under identical action sequences.

        Returns:
            ``(observation, info)``.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._wheel_vel = np.zeros(4, dtype=np.float32)
        self._step_idx = 0
        return self._observe(), {"step_idx": 0}

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
        self._theta = float((new_theta + np.pi) % (2 * np.pi) - np.pi)

        dx = self._goal_xy[0] - self._x
        dy = self._goal_xy[1] - self._y
        distance = float(np.hypot(dx, dy))
        reward = -distance

        self._step_idx += 1
        truncated = self._step_idx >= self._max_steps
        terminated = distance < 0.10

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

    def _compute_observation_keys(self) -> tuple[str, ...]:
        """Enumerate the obs keys based on observation toggles in config."""
        obs_cfg = self._cfg.observation
        keys: list[str] = []
        if obs_cfg.include_imu:
            keys.append("imu")
        if obs_cfg.include_chassis_pose:
            keys.append("chassis_pose")
        if obs_cfg.include_wheel_encoders:
            keys.append("wheel_vel")
        if obs_cfg.include_lidar_sectors:
            keys.append("lidar")
        return tuple(keys)

    def _observe(self) -> dict[str, NDArray[np.float32]]:
        """Produce an observation dict in the configured key order."""
        obs_cfg = self._cfg.observation
        obs: dict[str, NDArray[np.float32]] = {}
        if obs_cfg.include_imu:
            # Kinematic model: zero linear accel + a single omega component.
            obs["imu"] = np.zeros(6, dtype=np.float32)
        if obs_cfg.include_chassis_pose:
            obs["chassis_pose"] = np.asarray(
                [self._x, self._y, np.cos(self._theta), np.sin(self._theta)],
                dtype=np.float32,
            )
        if obs_cfg.include_wheel_encoders:
            obs["wheel_vel"] = self._wheel_vel.copy()
        if obs_cfg.include_lidar_sectors:
            obs["lidar"] = np.zeros(obs_cfg.lidar_num_sectors, dtype=np.float32)
        return obs
