"""Protocol for LLM-backed arm replanners."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from mousedroid.arm.protocols import PlanStep, SymbolicState


@runtime_checkable
class LLMReplannerProtocol(Protocol):
    """Generates a recovery plan from a (state, goal, error) triple.

    Implementations may call any LLM provider (Anthropic, the local
    ``LLMGateway``, an in-process stub for tests, …). Returning an empty
    list is the explicit signal for the outer :class:`Replanner` to fall
    back to its symbolic planner — no other "no plan" sentinel is used.
    """

    @property
    def name(self) -> str: ...

    def replan(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> list[PlanStep]: ...


__all__ = ["LLMReplannerProtocol"]
