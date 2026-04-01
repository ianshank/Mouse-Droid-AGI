"""Base class for arm manipulation Gymnasium environments.

Extracts shared logic (joint state management, action clipping,
step counting, truncation, reward computation) to reduce duplication
between ``TowerOfHanoiEnv`` and ``LaundrySortingEnv``.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.arm.environments.reward_shaping import RewardShaper
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmTaskConfig, ArmTrainingConfig

_log = get_logger(__name__)


class ArmEnvironmentBase(abc.ABC):
    """Abstract base for goal-conditioned arm environments.

    Provides joint state management, action clipping, step
    counting, truncation, and reward computation. Subclasses
    implement goal-specific observation and termination logic.

    Args:
        task_cfg: Task configuration.
        training_cfg: Training configuration with reward weights.
        dof: Robot arm degrees of freedom.
    """

    def __init__(
        self,
        task_cfg: ArmTaskConfig,
        training_cfg: ArmTrainingConfig,
        dof: int = 6,
    ) -> None:
        """Initialise base environment.

        Args:
            task_cfg: Task config with episode limits.
            training_cfg: Training config with reward and action bounds.
            dof: Arm degrees of freedom.
        """
        self._task_cfg = task_cfg
        self._training_cfg = training_cfg
        self._dof = dof
        self._max_steps = task_cfg.max_episode_steps
        self._action_delta_min = training_cfg.action_delta_min
        self._action_delta_max = training_cfg.action_delta_max
        self._reward_shaper = RewardShaper(training_cfg)

        self._joint_angles: NDArray[np.float64] = np.zeros(dof, dtype=np.float64)
        self._step_count = 0
        self._rng = np.random.default_rng(training_cfg.seed)

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Optional random seed.

        Returns:
            Tuple of (observation_dict, info_dict).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._joint_angles = np.zeros(self._dof, dtype=np.float64)
        self._step_count = 0
        self._reset_task_state()

        obs = self._get_observation()
        info: dict[str, Any] = {"is_success": False}
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

        info = self._get_step_info()

        is_success = self._check_goal()
        info["is_success"] = is_success
        terminated = is_success
        truncated = self._step_count >= self._max_steps

        obs = self._get_observation()
        reward = self._reward_shaper.compute(obs["achieved_goal"], obs["desired_goal"], info)

        return obs, reward, terminated, truncated, info

    def render(self) -> NDArray[np.uint8] | None:
        """Render current state (stub — override with MuJoCo rendering).

        Returns:
            None (headless mode).
        """
        return None

    def close(self) -> None:
        """Clean up resources."""
        _log.debug("env_closed")

    @abc.abstractmethod
    def _reset_task_state(self) -> None:
        """Reset task-specific state (called from ``reset()``)."""

    @abc.abstractmethod
    def _get_observation(self) -> dict[str, NDArray[np.float64]]:
        """Build goal-conditioned observation dict.

        Returns:
            Dict with observation, achieved_goal, desired_goal arrays.
        """

    @abc.abstractmethod
    def _check_goal(self) -> bool:
        """Check if task goal is achieved.

        Returns:
            True if goal is met.
        """

    @abc.abstractmethod
    def _get_step_info(self) -> dict[str, Any]:
        """Build step info dict with event flags.

        Returns:
            Info dict with task-specific event flags.
        """
