"""Shaped reward functions for arm manipulation tasks.

All reward weights are configurable via ArmTrainingConfig — no
hardcoded reward values in this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmTrainingConfig

_log = get_logger(__name__)


class RewardShaper:
    """Configurable reward function for arm manipulation.

    Computes shaped rewards for grasp, place, completion, and
    collision events using weights from training config.

    Args:
        training_cfg: Training config with reward/penalty weights.
    """

    def __init__(self, training_cfg: ArmTrainingConfig) -> None:
        """Initialise reward shaper.

        Args:
            training_cfg: Config containing reward weights.
        """
        self._cfg = training_cfg
        _log.info(
            "reward_shaper_init",
            grasp=training_cfg.reward_grasp,
            place=training_cfg.reward_place,
            complete=training_cfg.reward_complete,
            collision=training_cfg.penalty_collision,
        )

    def compute(
        self,
        achieved_goal: NDArray[np.float64],
        desired_goal: NDArray[np.float64],
        info: dict[str, Any],
    ) -> float:
        """Compute shaped reward for a transition.

        Args:
            achieved_goal: State achieved after action.
            desired_goal: Target goal state.
            info: Step info dict with event flags.

        Returns:
            Scalar reward value.
        """
        reward = 0.0

        # Task completion bonus
        if info.get("is_success", False):
            reward += self._cfg.reward_complete

        # Grasp reward
        if info.get("grasp_success", False):
            reward += self._cfg.reward_grasp

        # Correct placement reward
        if info.get("place_correct", False):
            reward += self._cfg.reward_place

        # Collision penalty
        if info.get("collision", False):
            reward += self._cfg.penalty_collision

        # Wrong disk penalty (Tower of Hanoi constraint violation)
        if info.get("wrong_disk", False):
            reward += self._cfg.penalty_wrong_disk

        # Distance-based shaping (dense component)
        if achieved_goal.shape == desired_goal.shape:
            distance = float(np.linalg.norm(achieved_goal - desired_goal))
            reward -= 0.01 * distance  # Small distance penalty for guidance

        return reward

    def compute_batch(
        self,
        achieved_goals: NDArray[np.float64],
        desired_goals: NDArray[np.float64],
        infos: list[dict[str, Any]],
    ) -> NDArray[np.float64]:
        """Compute rewards for a batch of transitions (HER compatibility).

        Args:
            achieved_goals: Batch of achieved goals, shape ``(N, goal_dim)``.
            desired_goals: Batch of desired goals, shape ``(N, goal_dim)``.
            infos: List of info dicts, length N.

        Returns:
            Reward array, shape ``(N,)``.
        """
        rewards = np.array(
            [
                self.compute(achieved_goals[i], desired_goals[i], infos[i])
                for i in range(len(infos))
            ],
            dtype=np.float64,
        )
        return rewards
