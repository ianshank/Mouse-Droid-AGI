"""Reusable acceptance-predicate combinators for harness tasks.

Each combinator implements :class:`AcceptancePredicateProtocol`. They are
deliberately tiny, deterministic, and side-effect free so the task tracker
can call them inside the 30 Hz tick loop without cost concerns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mousedroid.harness.protocol import (
    AcceptancePredicateProtocol,
    TaskState,
    TickContext,
)

if TYPE_CHECKING:  # pragma: no cover - import-only
    pass


@dataclass(frozen=True)
class AlwaysTrue:
    """Predicate that always returns True (useful as a sentinel/test)."""

    def __call__(self, task_state: TaskState, ctx: TickContext) -> bool:
        return True


@dataclass(frozen=True)
class AlwaysFalse:
    """Predicate that never accepts."""

    def __call__(self, task_state: TaskState, ctx: TickContext) -> bool:
        return False


@dataclass(frozen=True)
class AllOf:
    """Boolean AND of any number of acceptance predicates."""

    predicates: tuple[AcceptancePredicateProtocol, ...]

    def __call__(self, task_state: TaskState, ctx: TickContext) -> bool:
        return all(p(task_state, ctx) for p in self.predicates)


@dataclass(frozen=True)
class AnyOf:
    """Boolean OR of any number of acceptance predicates."""

    predicates: tuple[AcceptancePredicateProtocol, ...]

    def __call__(self, task_state: TaskState, ctx: TickContext) -> bool:
        return any(p(task_state, ctx) for p in self.predicates)


@dataclass(frozen=True)
class Negate:
    """Boolean NOT of a single acceptance predicate."""

    predicate: AcceptancePredicateProtocol

    def __call__(self, task_state: TaskState, ctx: TickContext) -> bool:
        return not self.predicate(task_state, ctx)


@dataclass(frozen=True)
class TickCountReached:
    """Accept after the orchestrator has executed at least ``n`` ticks.

    Useful in tests and as a building block for time-bounded acceptance.
    Reads ``task_state.started_at_s`` indirectly by deriving the start
    tick from ``ctx.tick_index`` at submission time captured in metadata.
    """

    n: int

    def __call__(self, task_state: TaskState, ctx: TickContext) -> bool:
        start_tick = task_state.spec.metadata.get("submitted_at_tick")
        if not isinstance(start_tick, int):
            return False
        return (ctx.tick_index - start_tick) >= self.n


@dataclass(frozen=True)
class CallablePredicate:
    """Wrap an arbitrary ``(state, ctx) -> bool`` callable as a predicate.

    Provided so callers can express ad-hoc acceptance criteria (e.g. lambda
    closures, bound methods) without subclassing.
    """

    fn: Callable[[TaskState, TickContext], bool]

    def __call__(self, task_state: TaskState, ctx: TickContext) -> bool:
        return bool(self.fn(task_state, ctx))


@dataclass(frozen=True)
class ObservationFieldEquals:
    """Accept when ``ctx.observation.<field>`` equals ``expected``.

    Falls back to ``False`` when the observation is missing or the field
    is not present, so the predicate is safe to apply against partial
    observation snapshots.
    """

    field_name: str
    expected: Any

    def __call__(self, task_state: TaskState, ctx: TickContext) -> bool:
        obs = ctx.observation
        if obs is None:
            return False
        if isinstance(obs, dict):
            return obs.get(self.field_name) == self.expected
        return getattr(obs, self.field_name, object()) == self.expected


__all__ = [
    "AllOf",
    "AlwaysFalse",
    "AlwaysTrue",
    "AnyOf",
    "CallablePredicate",
    "Negate",
    "ObservationFieldEquals",
    "TickCountReached",
]
