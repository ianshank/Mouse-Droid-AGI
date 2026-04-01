"""Laundry sorting Gymnasium environment.

Extends the arm manipulation framework for garment sorting
into multiple baskets based on colour and fabric type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.arm.environments.reward_shaping import RewardShaper
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmTaskConfig, ArmTrainingConfig

_log = get_logger(__name__)


class LaundrySortingEnv:
    """Gymnasium-compatible laundry sorting environment.

    Simulates picking garments from a pile and placing them
    into the correct sorting basket.

    Args:
        task_cfg: Task configuration (num_baskets, positions).
        training_cfg: Training configuration (reward weights).
        dof: Robot arm degrees of freedom.
    """

    def __init__(
        self,
        task_cfg: ArmTaskConfig,
        training_cfg: ArmTrainingConfig,
        dof: int = 6,
    ) -> None:
        """Initialise laundry sorting environment.

        Args:
            task_cfg: Task config with basket parameters.
            training_cfg: Training config with reward weights.
            dof: Arm degrees of freedom.
        """
        self._task_cfg = task_cfg
        self._training_cfg = training_cfg
        self._dof = dof
        self._num_baskets = task_cfg.num_baskets
        self._max_steps = task_cfg.max_episode_steps
        self._action_delta_min = training_cfg.action_delta_min
        self._action_delta_max = training_cfg.action_delta_max
        self._reward_shaper = RewardShaper(training_cfg)

        self._joint_angles: NDArray[np.float64] = np.zeros(dof, dtype=np.float64)
        self._garments_sorted = 0
        self._total_garments = task_cfg.num_garments
        self._step_count = 0
        self._rng = np.random.default_rng(training_cfg.seed)

        _log.info(
            "laundry_sorting_env_init",
            num_baskets=self._num_baskets,
            max_steps=self._max_steps,
        )

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        """Reset environment with new garment pile.

        Args:
            seed: Optional random seed.

        Returns:
            Tuple of (observation_dict, info_dict).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._joint_angles = np.zeros(self._dof, dtype=np.float64)
        self._garments_sorted = 0
        self._step_count = 0

        obs = self._get_observation()
        info: dict[str, Any] = {"is_success": False, "garments_sorted": 0}
        return obs, info

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[dict[str, NDArray[np.float64]], float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Joint angle deltas, shape ``(dof,)``.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        self._step_count += 1

        action = np.clip(action, self._action_delta_min, self._action_delta_max)
        self._joint_angles = self._joint_angles + action
        self._joint_angles = np.clip(self._joint_angles, -np.pi, np.pi)

        info: dict[str, Any] = {
            "grasp_success": False,
            "place_correct": False,
            "collision": False,
            "wrong_disk": False,
            "garments_sorted": self._garments_sorted,
        }

        is_success = self._garments_sorted >= self._total_garments
        info["is_success"] = is_success
        terminated = is_success
        truncated = self._step_count >= self._max_steps

        obs = self._get_observation()
        reward = self._reward_shaper.compute(obs["achieved_goal"], obs["desired_goal"], info)

        return obs, reward, terminated, truncated, info

    def render(self) -> NDArray[np.uint8] | None:
        """Render current state.

        Returns:
            None (headless mode).
        """
        return None

    def close(self) -> None:
        """Clean up resources."""
        _log.debug("laundry_env_closed")

    def _get_observation(self) -> dict[str, NDArray[np.float64]]:
        """Build observation dict.

        Returns:
            Dict with observation, achieved_goal, desired_goal.
        """
        obs = np.concatenate(
            [
                self._joint_angles,
                np.array([float(self._garments_sorted)], dtype=np.float64),
            ]
        )

        achieved = np.array([float(self._garments_sorted)], dtype=np.float64)
        desired = np.array([float(self._total_garments)], dtype=np.float64)

        return {
            "observation": obs,
            "achieved_goal": achieved,
            "desired_goal": desired,
        }
