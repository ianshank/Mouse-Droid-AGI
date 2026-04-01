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

from mousedroid.arm.environments.base import ArmEnvironmentBase
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmTaskConfig, ArmTrainingConfig

_log = get_logger(__name__)


class TowerOfHanoiEnv(ArmEnvironmentBase):
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
        super().__init__(task_cfg, training_cfg, dof)
        self._num_disks = task_cfg.num_disks
        self._num_pegs = task_cfg.num_pegs
        self._disk_pegs: NDArray[np.int64] = np.zeros(self._num_disks, dtype=np.int64)

        _log.info(
            "tower_of_hanoi_env_init",
            num_disks=self._num_disks,
            num_pegs=self._num_pegs,
            max_steps=self._max_steps,
        )

    def _reset_task_state(self) -> None:
        """Reset all disks to peg 0."""
        self._disk_pegs = np.zeros(self._num_disks, dtype=np.int64)

    def _get_observation(self) -> dict[str, NDArray[np.float64]]:
        """Build goal-conditioned observation dict.

        Returns:
            Dict with observation, achieved_goal, desired_goal arrays.
        """
        obs = np.concatenate(
            [
                self._joint_angles,
                self._disk_pegs.astype(np.float64),
            ]
        )
        achieved = self._disk_pegs.astype(np.float64)
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

    def _get_step_info(self) -> dict[str, Any]:
        """Build step info with disk movement event flags.

        Returns:
            Info dict with grasp/place/collision flags.
        """
        return {
            "grasp_success": False,
            "place_correct": False,
            "collision": False,
            "wrong_disk": False,
        }
