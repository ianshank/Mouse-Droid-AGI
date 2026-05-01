"""Protocols and dataclasses for the agent harness.

All public interfaces live here as ``@runtime_checkable Protocol`` so
concrete implementations remain swappable and concrete types are only
imported inside :mod:`mousedroid.factory`. Dataclasses are frozen where
they describe immutable specifications and mutable where they hold
runtime state.
"""

from __future__ import annotations

import enum
import sys
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, enum.Enum):
        """Backport of ``enum.StrEnum`` for Python 3.10."""


class TaskStatus(StrEnum):
    """Lifecycle status of a harness task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


_TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.TIMED_OUT,
        TaskStatus.CANCELLED,
    }
)


def is_terminal(status: TaskStatus) -> bool:
    """Return True when ``status`` is a terminal lifecycle state."""
    return status in _TERMINAL_STATUSES


class HookPhase(StrEnum):
    """Tick-loop hook phases observed by the orchestrator."""

    PRE_TICK = "pre_tick"
    PRE_ACTION = "pre_action"
    POST_ACTION = "post_action"
    POST_TICK = "post_tick"
    ON_ERROR = "on_error"


# ---------------------------------------------------------------------------
# Acceptance predicates
# ---------------------------------------------------------------------------


@runtime_checkable
class AcceptancePredicateProtocol(Protocol):
    """Returns True when the bound task should be considered complete.

    A predicate is a sync callable with the signature
    ``(task_state, tick_context) -> bool`` so it can be cheaply evaluated
    inside the hot tick loop. Predicates must be pure: never raise on the
    happy path and never mutate ``task_state``.
    """

    def __call__(self, task_state: TaskState, ctx: TickContext) -> bool:
        """Evaluate the predicate against the task state and tick context."""
        ...


# ---------------------------------------------------------------------------
# Task spec & state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """Immutable specification of a harness task.

    Attributes:
        id: Stable identifier (caller-provided or generated).
        goal: Free-form goal description (NL or structured payload).
        constraints: Free-form constraints (limits, preferences).
        acceptance_predicate: Callable evaluated by the tracker each tick.
        timeout_s: Wall-clock timeout. ``None`` defers to the tracker
            config's ``default_timeout_s``.
        metadata: Arbitrary tags (skill name, parent agent, source, ...).
        parent_id: Optional id of a parent task for hierarchical delegation.
    """

    id: str
    goal: str
    acceptance_predicate: AcceptancePredicateProtocol
    constraints: tuple[str, ...] = ()
    timeout_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None


@dataclass
class TaskState:
    """Mutable runtime state of a submitted task."""

    spec: TaskSpec
    status: TaskStatus = TaskStatus.PENDING
    started_at_s: float | None = None
    finished_at_s: float | None = None
    last_error: str | None = None
    last_evaluated_at_s: float | None = None
    # Tick counter captured the first time the tracker evaluates this task.
    # ``TickCountReached`` reads it so callers don't have to manually inject
    # ``submitted_at_tick`` into ``TaskSpec.metadata``.
    started_at_tick: int | None = None

    @property
    def id(self) -> str:
        """Convenience accessor for ``spec.id``."""
        return self.spec.id

    @property
    def is_terminal(self) -> bool:
        """True when this task has reached a terminal status."""
        return is_terminal(self.status)


# ---------------------------------------------------------------------------
# Tick context (passed to hooks and predicates)
# ---------------------------------------------------------------------------


@dataclass
class TickContext:
    """Snapshot of one orchestrator tick passed to hooks and predicates.

    The orchestrator updates ``proposed_action``, ``executed_action``, and
    ``error`` as the tick progresses so hooks at later phases observe the
    latest values. Hooks SHOULD treat the context as read-only — only the
    orchestrator mutates it.
    """

    tick_index: int
    timestamp_s: float
    observation: Any = None
    safety_ctx: Any = None
    prev_action: Any = None
    proposed_action: Any = None
    executed_action: Any = None
    active_tasks: tuple[str, ...] = ()
    timings: dict[str, float] = field(default_factory=dict)
    error: BaseException | None = None
    loop_time_ms: float | None = None


# ---------------------------------------------------------------------------
# Task tracker protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TaskTrackerProtocol(Protocol):
    """Tracks the lifecycle of submitted harness tasks."""

    def submit(self, spec: TaskSpec) -> TaskState:
        """Register ``spec`` and return its initial state."""
        ...

    def get(self, task_id: str) -> TaskState | None:
        """Return the state of ``task_id`` or ``None`` if unknown."""
        ...

    def update(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        error: str | None = None,
    ) -> TaskState:
        """Force ``task_id`` to ``status``; sets ``last_error`` if given."""
        ...

    def evaluate(self, state: TaskState, ctx: TickContext) -> TaskStatus:
        """Evaluate ``state``'s predicate / timeout against ``ctx``.

        Returns the (possibly updated) status. Implementations MUST handle
        timeout enforcement and predicate exceptions gracefully.
        """
        ...

    async def evaluate_active(self, ctx: TickContext) -> tuple[TaskState, ...]:
        """Evaluate every active (non-terminal) task against ``ctx``.

        Returns a snapshot tuple of task states at evaluation time.
        """
        ...

    def active(self) -> Iterable[TaskState]:
        """Iterate currently-active (non-terminal) tasks."""
        ...

    def history(self) -> Iterable[TaskState]:
        """Iterate completed tasks in submission order (bounded)."""
        ...

    def cancel(self, task_id: str, *, reason: str | None = None) -> TaskState:
        """Force the task to ``CANCELLED``; idempotent for terminal tasks."""
        ...


# ---------------------------------------------------------------------------
# Hook protocol & registry
# ---------------------------------------------------------------------------


HookHandler = Callable[[TickContext], Awaitable[None]]
"""Async callable invoked for a registered hook phase."""


@dataclass(frozen=True)
class HookSpec:
    """Specification for a tick-loop hook.

    Attributes:
        name: Unique identifier; later registrations replace earlier ones.
        phase: When the hook should fire.
        handler: Async callable taking the ``TickContext``.
        error_policy: ``raise`` (re-raise after logging), ``warn``
            (log and continue), or ``swallow`` (silent).
    """

    name: str
    phase: HookPhase
    handler: HookHandler
    error_policy: str = "warn"


@runtime_checkable
class HookRegistryProtocol(Protocol):
    """Phase-keyed registry of tick-loop hooks."""

    def register(self, spec: HookSpec) -> None:
        """Register or replace a hook (replacement is logged)."""
        ...

    def unregister(self, name: str) -> bool:
        """Remove ``name`` from every phase. Returns True if removed."""
        ...

    def for_phase(self, phase: HookPhase) -> tuple[HookSpec, ...]:
        """Return registered hooks for ``phase`` in registration order."""
        ...

    async def run_phase(self, phase: HookPhase, ctx: TickContext) -> None:
        """Invoke every hook bound to ``phase`` with ``ctx``.

        Errors are handled per ``HookSpec.error_policy``. The orchestrator
        passes a no-op registry by default, so this is a single dict
        lookup + early-return when no hooks are registered.
        """
        ...


__all__ = [
    "AcceptancePredicateProtocol",
    "HookHandler",
    "HookPhase",
    "HookRegistryProtocol",
    "HookSpec",
    "TaskSpec",
    "TaskState",
    "TaskStatus",
    "TaskTrackerProtocol",
    "TickContext",
    "is_terminal",
]
