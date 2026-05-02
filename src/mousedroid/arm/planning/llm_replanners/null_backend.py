"""No-op LLM replanner — returns ``[]`` so the outer ``Replanner`` falls back."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.arm.planning.llm_replanners.base import LLMReplannerProtocol
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.arm.protocols import PlanStep, SymbolicState

_log = get_logger(__name__)


class NullLLMReplanner:
    """Default replanner: explicitly disabled, always falls back."""

    name = "null"

    def replan(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> list[PlanStep]:
        _log.debug(
            "null_llm_replan_called",
            error=error,
        )
        return []


_PROTOCOL_CHECK: LLMReplannerProtocol = NullLLMReplanner()
del _PROTOCOL_CHECK


__all__ = ["NullLLMReplanner"]
