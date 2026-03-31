"""Curriculum learning manager for progressive task difficulty.

Automatically advances difficulty stages (e.g., 1 -> 3 -> 5 disks)
based on evaluation success rate thresholds.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmCurriculumConfig

_log = get_logger(__name__)


class CurriculumManager:
    """Manage progressive difficulty for arm training.

    Tracks evaluation success rates and promotes to harder stages
    when the promotion threshold is met.

    Args:
        cfg: Curriculum configuration with stages and thresholds.
    """

    def __init__(self, cfg: ArmCurriculumConfig) -> None:
        """Initialise curriculum manager.

        Args:
            cfg: Curriculum config with stages, threshold, eval episodes.
        """
        self._cfg = cfg
        self._stages = list(cfg.stages)
        self._current_stage_idx = 0
        self._eval_results: deque[bool] = deque(maxlen=cfg.promotion_eval_episodes)
        _log.info(
            "curriculum_init",
            stages=self._stages,
            threshold=cfg.promotion_threshold,
        )

    @property
    def current_difficulty(self) -> int:
        """Current difficulty level (e.g., number of disks)."""
        return self._stages[self._current_stage_idx]

    @property
    def current_stage_index(self) -> int:
        """Current stage index (0-based)."""
        return self._current_stage_idx

    @property
    def is_final_stage(self) -> bool:
        """Whether current stage is the last one."""
        return self._current_stage_idx >= len(self._stages) - 1

    @property
    def success_rate(self) -> float:
        """Current evaluation success rate."""
        if not self._eval_results:
            return 0.0
        return sum(self._eval_results) / len(self._eval_results)

    def record_episode(self, success: bool) -> bool:
        """Record an evaluation episode result.

        Args:
            success: Whether the episode was successful.

        Returns:
            True if a stage promotion occurred.
        """
        self._eval_results.append(success)

        if len(self._eval_results) < self._cfg.promotion_eval_episodes:
            return False

        rate = self.success_rate
        if rate >= self._cfg.promotion_threshold and not self.is_final_stage:
            self._promote()
            return True

        return False

    def _promote(self) -> None:
        """Advance to the next curriculum stage."""
        old_stage = self.current_difficulty
        self._current_stage_idx += 1
        self._eval_results.clear()
        _log.info(
            "curriculum_promoted",
            from_stage=old_stage,
            to_stage=self.current_difficulty,
            stage_index=self._current_stage_idx,
        )

    def reset(self) -> None:
        """Reset curriculum to first stage."""
        self._current_stage_idx = 0
        self._eval_results.clear()
        _log.info("curriculum_reset")
