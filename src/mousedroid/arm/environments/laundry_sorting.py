"""Laundry sorting Gymnasium environment.

Extends the arm manipulation framework for garment sorting
into multiple baskets based on colour and fabric type.
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


class LaundrySortingEnv(ArmEnvironmentBase):
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
        super().__init__(task_cfg, training_cfg, dof)
        self._num_baskets = task_cfg.num_baskets
        self._total_garments = task_cfg.num_garments
        self._garments_sorted = 0

        _log.info(
            "laundry_sorting_env_init",
            num_baskets=self._num_baskets,
            max_steps=self._max_steps,
        )

    def _reset_task_state(self) -> None:
        """Reset garment sorting state."""
        self._garments_sorted = 0

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

    def _check_goal(self) -> bool:
        """Check if all garments are sorted.

        Returns:
            True if all garments sorted into baskets.
        """
        return self._garments_sorted >= self._total_garments

    def _get_step_info(self) -> dict[str, Any]:
        """Build step info with sorting event flags.

        Returns:
            Info dict with grasp/place/collision flags + sort count.
        """
        return {
            "grasp_success": False,
            "place_correct": False,
            "collision": False,
            "wrong_disk": False,
            "garments_sorted": self._garments_sorted,
        }
