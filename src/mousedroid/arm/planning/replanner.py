"""LLM-based adaptive replanner for failure recovery.

When symbolic plan execution fails (e.g., unexpected state), this
module uses an LLM to analyse the error and generate a recovery plan.
Falls back to the symbolic planner if LLM replanning is disabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.arm.protocols import PlanStep, SymbolicState
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmPlanningConfig, ArmTaskConfig

_log = get_logger(__name__)


class Replanner:
    """Adaptive replanner with LLM-based error analysis.

    When execution fails, analyses the current state and error to
    generate a recovery plan. Can use either LLM reasoning or
    fall back to pure symbolic replanning.

    Args:
        planning_cfg: Planning configuration.
        task_cfg: Task configuration.
    """

    def __init__(self, planning_cfg: ArmPlanningConfig, task_cfg: ArmTaskConfig) -> None:
        """Initialise replanner.

        Args:
            planning_cfg: Planning config with replanner settings.
            task_cfg: Task config for problem generation.
        """
        self._planning_cfg = planning_cfg
        self._task_cfg = task_cfg
        self._attempt_count = 0
        _log.info(
            "replanner_init",
            llm_enabled=planning_cfg.llm_replanner_enabled,
            max_attempts=planning_cfg.max_replan_attempts,
        )

    def replan(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> list[PlanStep]:
        """Generate recovery plan after execution failure.

        Args:
            current_state: Current (unexpected) symbolic state.
            goal_state: Original target state.
            error: Description of the execution failure.

        Returns:
            Recovery plan steps.

        Raises:
            ReplanningExhaustedError: If max replan attempts exceeded.
        """
        self._attempt_count += 1

        if self._attempt_count > self._planning_cfg.max_replan_attempts:
            msg = f"Max replan attempts ({self._planning_cfg.max_replan_attempts}) exceeded"
            _log.error("replanning_exhausted", attempts=self._attempt_count)
            raise ReplanningExhaustedError(msg)

        _log.warning(
            "replanning_attempt",
            attempt=self._attempt_count,
            error=error,
        )

        if self._planning_cfg.llm_replanner_enabled:
            return self._llm_replan(current_state, goal_state, error)

        # Fall back to symbolic replanning from current state
        from mousedroid.arm.planning.symbolic_planner import SymbolicPlanner

        planner = SymbolicPlanner(self._planning_cfg, self._task_cfg)
        return planner.plan(current_state, goal_state)

    def _llm_replan(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> list[PlanStep]:
        """Use LLM to analyse error and generate recovery plan.

        Args:
            current_state: Current symbolic state.
            goal_state: Target symbolic state.
            error: Failure description.

        Returns:
            Recovery plan from LLM analysis.
        """
        _log.info("llm_replan_start", error=error)

        # LLM integration placeholder — will use Claude API
        # For now, fall back to symbolic replanning
        _log.warning("llm_replan_not_implemented", fallback="symbolic")
        from mousedroid.arm.planning.symbolic_planner import SymbolicPlanner

        planner = SymbolicPlanner(self._planning_cfg, self._task_cfg)
        return planner.plan(current_state, goal_state)

    def reset(self) -> None:
        """Reset attempt counter for a new execution cycle."""
        self._attempt_count = 0


class ReplanningExhaustedError(Exception):
    """Raised when maximum replanning attempts are exceeded."""
