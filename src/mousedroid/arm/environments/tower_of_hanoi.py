"""Tower of Hanoi Gymnasium environment.

Provides a Gymnasium-compatible environment for Tower of Hanoi
with configurable disk count, goal-conditioned observations,
and shaped rewards. Can run headless (training) or with MuJoCo
rendering (evaluation/debugging).
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


class TowerOfHanoiEnv:
    """Gymnasium-compatible Tower of Hanoi environment.

    Simulates the Tower of Hanoi puzzle with configurable disk count.
    Provides goal-conditioned observations for HER compatibility.

    Observation space (dict):
        - observation: joint angles + disk positions, shape ``(obs_dim,)``
        - achieved_goal: current disk configuration, shape ``(goal_dim,)``
        - desired_goal: target disk configuration, shape ``(goal_dim,)``

    Action space:
        Continuous joint angle deltas, shape ``(dof,)``

    Args:
        task_cfg: Task configuration (num_disks, peg_positions, etc.).
        training_cfg: Training configuration (reward weights).
        dof: Robot arm degrees of freedom.
    """

    def __init__(
        self,
        task_cfg: ArmTaskConfig,
        training_cfg: ArmTrainingConfig,
        dof: int = 6,
    ) -> None:
        """Initialise Tower of Hanoi environment.

        Args:
            task_cfg: Task config with disk/peg parameters.
            training_cfg: Training config with reward weights.
            dof: Arm degrees of freedom.
        """
        self._task_cfg = task_cfg
        self._training_cfg = training_cfg
        self._dof = dof
        self._num_disks = task_cfg.num_disks
        self._num_pegs = task_cfg.num_pegs
        self._max_steps = task_cfg.max_episode_steps
        self._action_delta_min = training_cfg.action_delta_min
        self._action_delta_max = training_cfg.action_delta_max
        self._reward_shaper = RewardShaper(training_cfg)

        # State: which peg each disk is on (0-indexed)
        self._disk_pegs: NDArray[np.int64] = np.zeros(self._num_disks, dtype=np.int64)
        self._joint_angles: NDArray[np.float64] = np.zeros(dof, dtype=np.float64)
        self._step_count = 0
        self._rng = np.random.default_rng(training_cfg.seed)

        _log.info(
            "tower_of_hanoi_env_init",
            num_disks=self._num_disks,
            num_pegs=self._num_pegs,
            max_steps=self._max_steps,
        )

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        """Reset environment to initial state.

        All disks start on peg 0 (smallest on top).

        Args:
            seed: Optional random seed.

        Returns:
            Tuple of (observation_dict, info_dict).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # All disks on first peg
        self._disk_pegs = np.zeros(self._num_disks, dtype=np.int64)
        self._joint_angles = np.zeros(self._dof, dtype=np.float64)
        self._step_count = 0

        obs = self._get_observation()
        info: dict[str, Any] = {"is_success": False}
        return obs, info

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[dict[str, NDArray[np.float64]], float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Interprets action as joint angle deltas, updates state,
        and checks for disk movement (symbolic level).

        Args:
            action: Joint angle deltas, shape ``(dof,)``.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        self._step_count += 1

        # Apply action (clipped joint deltas)
        action = np.clip(action, self._action_delta_min, self._action_delta_max)
        self._joint_angles = self._joint_angles + action
        self._joint_angles = np.clip(self._joint_angles, -np.pi, np.pi)

        # Check for symbolic disk movement (simplified — real env uses contact detection)
        info = self._check_disk_movement()

        # Check termination
        is_success = self._check_goal()
        info["is_success"] = is_success
        terminated = is_success
        truncated = self._step_count >= self._max_steps

        # Compute reward
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

    def _get_observation(self) -> dict[str, NDArray[np.float64]]:
        """Build goal-conditioned observation dict.

        Returns:
            Dict with observation, achieved_goal, desired_goal arrays.
        """
        # Observation: joint angles + disk peg assignments
        obs = np.concatenate(
            [
                self._joint_angles,
                self._disk_pegs.astype(np.float64),
            ]
        )

        # Achieved goal: current disk peg assignments
        achieved = self._disk_pegs.astype(np.float64)

        # Desired goal: all disks on last peg
        desired = np.full(self._num_disks, self._num_pegs - 1, dtype=np.float64)

        return {
            "observation": obs,
            "achieved_goal": achieved,
            "desired_goal": desired,
        }

    def _check_goal(self) -> bool:
        """Check if all disks are on the target peg.

        Returns:
            True if puzzle is solved.
        """
        target_peg = self._num_pegs - 1
        return bool(np.all(self._disk_pegs == target_peg))

    def _check_disk_movement(self) -> dict[str, Any]:
        """Check for symbolic disk movement based on joint state.

        This is a simplified simulation — in the full MuJoCo env,
        disk movement is determined by physics contact detection.

        Returns:
            Info dict with event flags.
        """
        return {
            "grasp_success": False,
            "place_correct": False,
            "collision": False,
            "wrong_disk": False,
        }
