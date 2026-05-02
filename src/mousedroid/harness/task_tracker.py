"""In-memory implementation of :class:`TaskTrackerProtocol`.

The tracker is the deterministic part of the agent harness that knows
which tasks are active, evaluates their acceptance predicates each tick,
and enforces timeouts. It is intentionally synchronous in the hot path
(predicate evaluation) and asynchronous only at the iteration boundary
so it can be awaited from the orchestrator's tick loop without surprise.

All tunables come from :class:`mousedroid.config.schema.HarnessTrackerConfig`;
nothing is hardcoded.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from mousedroid.harness.protocol import (
    TaskSpec,
    TaskState,
    TaskStatus,
    TickContext,
    is_terminal,
)
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import HarnessTrackerConfig

_log = get_logger(__name__)


class TaskTrackerError(RuntimeError):
    """Raised when a tracker operation cannot be satisfied."""


class InMemoryTaskTracker:
    """Stores task lifecycle in memory; no persistence.

    The tracker is fully synchronous for read/write APIs and exposes a
    single async iterator (``evaluate_active``) for orchestrator
    integration. History is bounded by the configured ``history_size``
    using ``deque(maxlen=...)`` per project invariant.
    """

    def __init__(
        self,
        cfg: HarnessTrackerConfig,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Build the in-memory task tracker.

        Args:
            cfg: Config sub-model controlling history, timeout, and capacity.
            clock: Optional ``() -> seconds`` clock for deterministic tests.
                Defaults to :func:`time.monotonic`.
        """
        self._cfg = cfg
        self._clock = clock if clock is not None else time.monotonic
        self._active: dict[str, TaskState] = {}
        self._history: deque[TaskState] = deque(maxlen=cfg.history_size)
        _log.info(
            "task_tracker_initialised",
            history_size=cfg.history_size,
            default_timeout_s=cfg.default_timeout_s,
            max_active=cfg.max_active,
            enabled=cfg.enabled,
        )

    # ---------------------------------------------------------------- API
    def submit(self, spec: TaskSpec) -> TaskState:
        """Register ``spec`` and return its initial state.

        Raises:
            TaskTrackerError: If the active set is at ``max_active`` or a
                task with the same id already exists in either the active
                set or the bounded history (so ids stay unique across the
                tracker's full state, not just ``_active``).
        """
        if self.get(spec.id) is not None:
            msg = f"Task ID already exists: {spec.id!r}"
            raise TaskTrackerError(msg)
        if len(self._active) >= self._cfg.max_active:
            msg = f"Active task cap reached ({self._cfg.max_active}); cannot submit {spec.id!r}"
            raise TaskTrackerError(msg)
        state = TaskState(
            spec=spec,
            status=TaskStatus.RUNNING,
            started_at_s=self._clock(),
        )
        self._active[spec.id] = state
        _log.info(
            "task_submitted",
            task_id=spec.id,
            goal=spec.goal,
            timeout_s=self._effective_timeout(spec),
        )
        return state

    def get(self, task_id: str) -> TaskState | None:
        if task_id in self._active:
            return self._active[task_id]
        for state in self._history:
            if state.id == task_id:
                return state
        return None

    def update(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        error: str | None = None,
    ) -> TaskState:
        state = self._active.get(task_id)
        if state is None:
            # Already terminal — search history and return without mutating.
            historical = self.get(task_id)
            if historical is None:
                msg = f"Unknown task: {task_id!r}"
                raise TaskTrackerError(msg)
            return historical
        old = state.status
        state.status = status
        if error is not None:
            state.last_error = error
        if is_terminal(status):
            state.finished_at_s = self._clock()
            self._active.pop(task_id, None)
            self._history.append(state)
        _log.info(
            "task_state_changed",
            task_id=task_id,
            from_status=old.value,
            to_status=status.value,
            error=error,
        )
        return state

    def evaluate(self, state: TaskState, ctx: TickContext) -> TaskStatus:
        """Apply the predicate + timeout to ``state``.

        Predicate exceptions are caught and recorded as ``FAILED``. The
        first evaluation also captures ``ctx.tick_index`` into
        ``state.started_at_tick`` so tick-based predicates (e.g.
        :class:`TickCountReached`) work without callers having to thread
        the start tick through ``TaskSpec.metadata`` themselves.
        """
        if state.is_terminal:
            return state.status
        state.last_evaluated_at_s = self._clock()
        if state.started_at_tick is None:
            state.started_at_tick = ctx.tick_index

        # Timeout takes precedence — the predicate may never be queried
        # again once the task has expired.
        timeout_s = self._effective_timeout(state.spec)
        if state.started_at_s is not None and timeout_s > 0.0:
            elapsed = state.last_evaluated_at_s - state.started_at_s
            if elapsed >= timeout_s:
                self.update(
                    state.id,
                    TaskStatus.TIMED_OUT,
                    error=f"timeout after {elapsed:.3f}s",
                )
                return TaskStatus.TIMED_OUT

        try:
            accepted = bool(state.spec.acceptance_predicate(state, ctx))
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning(
                "task_predicate_error",
                task_id=state.id,
                error=str(exc),
                exc_info=True,
            )
            self.update(state.id, TaskStatus.FAILED, error=str(exc))
            return TaskStatus.FAILED

        if accepted:
            self.update(state.id, TaskStatus.COMPLETED)
            _log.info("task_terminal", task_id=state.id, status=TaskStatus.COMPLETED.value)
            return TaskStatus.COMPLETED
        return TaskStatus(state.status)

    async def evaluate_active(self, ctx: TickContext) -> tuple[TaskState, ...]:
        """Evaluate every currently-active task; returns the same snapshot."""
        # Copy keys defensively — ``evaluate`` may move tasks to history.
        snapshot = tuple(self._active.values())
        for state in snapshot:
            self.evaluate(state, ctx)
        # ``snapshot`` already references the (now possibly mutated) states.
        return snapshot

    def active(self) -> Iterable[TaskState]:
        return tuple(self._active.values())

    def history(self) -> Iterable[TaskState]:
        return tuple(self._history)

    def cancel(self, task_id: str, *, reason: str | None = None) -> TaskState:
        existing = self.get(task_id)
        if existing is None:
            msg = f"Unknown task: {task_id!r}"
            raise TaskTrackerError(msg)
        if existing.is_terminal:
            return existing
        return self.update(task_id, TaskStatus.CANCELLED, error=reason)

    # ------------------------------------------------------------ helpers
    def _effective_timeout(self, spec: TaskSpec) -> float:
        if spec.timeout_s is not None and spec.timeout_s > 0.0:
            return spec.timeout_s
        return self._cfg.default_timeout_s


__all__ = ["InMemoryTaskTracker", "TaskTrackerError"]
