"""LLM-based adaptive replanner for failure recovery.

When symbolic plan execution fails (e.g., unexpected state), this
module uses an LLM to analyse the error and generate a recovery plan.
Falls back to the symbolic planner if LLM replanning is disabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.arm.protocols import ArmPlannerProtocol, PlanStep, SymbolicState
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.arm.planning.llm_replanners.base import LLMReplannerProtocol
    from mousedroid.config.schema import ArmPlanningConfig

_log = get_logger(__name__)


class Replanner:
    """Adaptive replanner with LLM-based error analysis.

    When execution fails, analyses the current state and error to
    generate a recovery plan. Can use either LLM reasoning or
    fall back to pure symbolic replanning.

    Args:
        planning_cfg: Planning configuration.
        planner: Symbolic planner (injected via protocol).
    """

    def __init__(
        self,
        planning_cfg: ArmPlanningConfig,
        planner: ArmPlannerProtocol,
        *,
        llm_replanner: LLMReplannerProtocol | None = None,
    ) -> None:
        """Initialise replanner.

        Args:
            planning_cfg: Planning config with replanner settings.
            planner: Injected planner implementing ArmPlannerProtocol.
            llm_replanner: Optional LLM-backed replanner. When ``None``
                (the default) and the legacy ``llm_replanner_enabled``
                flag is True, a :class:`NullLLMReplanner` is wired so the
                outer fall-back to symbolic planning is preserved.
        """
        self._planning_cfg = planning_cfg
        self._planner = planner
        self._attempt_count = 0
        if llm_replanner is None:
            from mousedroid.arm.planning.llm_replanners.null_backend import (
                NullLLMReplanner,
            )

            llm_replanner = NullLLMReplanner()
        self._llm_replanner = llm_replanner
        _log.info(
            "replanner_init",
            llm_enabled=planning_cfg.llm_replanner_enabled,
            max_attempts=planning_cfg.max_replan_attempts,
            llm_backend=getattr(self._llm_replanner, "name", "unknown"),
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

        if self._is_llm_replan_enabled():
            return self._llm_replan(current_state, goal_state, error)

        # Fall back to symbolic replanning from current state
        return self._planner.plan(current_state, goal_state)

    def _is_llm_replan_enabled(self) -> bool:
        """True when LLM replanning is requested by either config surface.

        The new :class:`LLMReplannerConfig` sub-model takes precedence: if
        ``planning_cfg.llm_replanner.enabled`` is True the LLM path runs
        regardless of the legacy ``llm_replanner_enabled`` flag, so a
        deployment configured solely through the new sub-model works
        without also setting the legacy boolean. Conversely, the legacy
        flag remains an opt-in for backwards compatibility.
        """
        rp_cfg = getattr(self._planning_cfg, "llm_replanner", None)
        if rp_cfg is not None and rp_cfg.enabled:
            return True
        return bool(self._planning_cfg.llm_replanner_enabled)

    def _llm_replan(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> list[PlanStep]:
        """Use the injected ``LLMReplannerProtocol`` to generate a recovery plan.

        When the backend returns an empty list (the default for
        :class:`NullLLMReplanner`, or any backend that fails / is
        disabled), fall back to symbolic replanning. This keeps existing
        callers working when the LLM extra is not installed.
        """
        _log.info(
            "llm_replan_start",
            error=error,
            backend=getattr(self._llm_replanner, "name", "unknown"),
        )
        steps = self._llm_replanner.replan(current_state, goal_state, error)
        if steps:
            return steps
        _log.warning("llm_replan_empty_falling_back", fallback="symbolic")
        return self._planner.plan(current_state, goal_state)

    def reset(self) -> None:
        """Reset attempt counter for a new execution cycle."""
        self._attempt_count = 0


class ReplanningExhaustedError(Exception):
    """Raised when maximum replanning attempts are exceeded."""
