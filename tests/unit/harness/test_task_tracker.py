"""Tests for ``mousedroid.harness.task_tracker.InMemoryTaskTracker``."""

from __future__ import annotations

import itertools

import pytest

from mousedroid.config.schema import HarnessTrackerConfig
from mousedroid.harness.predicates import (
    AlwaysFalse,
    AlwaysTrue,
    CallablePredicate,
)
from mousedroid.harness.protocol import (
    TaskSpec,
    TaskStatus,
    TaskTrackerProtocol,
    TickContext,
)
from mousedroid.harness.task_tracker import (
    InMemoryTaskTracker,
    TaskTrackerError,
)


class _ManualClock:
    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


def _ctx(tick: int = 0) -> TickContext:
    return TickContext(tick_index=tick, timestamp_s=float(tick))


_DEFAULT_PREDICATE = AlwaysFalse()


def _spec(
    idx: str = "t",
    *,
    predicate=_DEFAULT_PREDICATE,
    timeout_s: float | None = None,
) -> TaskSpec:
    return TaskSpec(id=idx, goal="g", acceptance_predicate=predicate, timeout_s=timeout_s)


@pytest.fixture
def clock() -> _ManualClock:
    return _ManualClock()


@pytest.fixture
def tracker(clock: _ManualClock) -> InMemoryTaskTracker:
    cfg = HarnessTrackerConfig(enabled=True, history_size=4, default_timeout_s=10.0, max_active=3)
    return InMemoryTaskTracker(cfg, clock=clock)


# ---------------------------------------------------------------------------
# Construction & invariants
# ---------------------------------------------------------------------------


def test_implements_protocol(tracker: InMemoryTaskTracker) -> None:
    assert isinstance(tracker, TaskTrackerProtocol)


def test_submit_then_get(tracker: InMemoryTaskTracker) -> None:
    spec = _spec("a")
    state = tracker.submit(spec)
    assert state.status == TaskStatus.RUNNING
    assert tracker.get("a") is state


def test_submit_duplicate_raises(tracker: InMemoryTaskTracker) -> None:
    tracker.submit(_spec("a"))
    with pytest.raises(TaskTrackerError):
        tracker.submit(_spec("a"))


def test_submit_capped_by_max_active(tracker: InMemoryTaskTracker) -> None:
    tracker.submit(_spec("a"))
    tracker.submit(_spec("b"))
    tracker.submit(_spec("c"))
    with pytest.raises(TaskTrackerError):
        tracker.submit(_spec("d"))


# ---------------------------------------------------------------------------
# evaluate / lifecycle
# ---------------------------------------------------------------------------


def test_evaluate_completes_on_predicate_true(tracker: InMemoryTaskTracker) -> None:
    state = tracker.submit(_spec("a", predicate=AlwaysTrue()))
    new_status = tracker.evaluate(state, _ctx())
    assert new_status == TaskStatus.COMPLETED
    assert state.is_terminal


def test_evaluate_keeps_running_when_false(tracker: InMemoryTaskTracker) -> None:
    state = tracker.submit(_spec("a", predicate=AlwaysFalse()))
    assert tracker.evaluate(state, _ctx()) == TaskStatus.RUNNING
    assert not state.is_terminal


def test_evaluate_terminal_is_idempotent(tracker: InMemoryTaskTracker) -> None:
    state = tracker.submit(_spec("a", predicate=AlwaysTrue()))
    tracker.evaluate(state, _ctx())
    again = tracker.evaluate(state, _ctx())
    assert again == TaskStatus.COMPLETED


def test_evaluate_timeout(clock: _ManualClock, tracker: InMemoryTaskTracker) -> None:
    state = tracker.submit(_spec("a", predicate=AlwaysFalse(), timeout_s=2.0))
    clock.advance(1.0)
    assert tracker.evaluate(state, _ctx()) == TaskStatus.RUNNING
    clock.advance(2.0)
    assert tracker.evaluate(state, _ctx()) == TaskStatus.TIMED_OUT
    assert state.last_error is not None
    assert "timeout" in state.last_error


def test_evaluate_default_timeout_used_when_spec_omits_it(
    clock: _ManualClock,
    tracker: InMemoryTaskTracker,
) -> None:
    state = tracker.submit(_spec("a", predicate=AlwaysFalse()))  # uses default 10s
    clock.advance(11.0)
    assert tracker.evaluate(state, _ctx()) == TaskStatus.TIMED_OUT


def test_predicate_exception_marks_failed(tracker: InMemoryTaskTracker) -> None:
    def boom(state, ctx):  # type: ignore[no-untyped-def]
        raise RuntimeError("kapow")

    state = tracker.submit(_spec("a", predicate=CallablePredicate(boom)))
    assert tracker.evaluate(state, _ctx()) == TaskStatus.FAILED
    assert state.last_error == "kapow"


# ---------------------------------------------------------------------------
# active / history / cancel
# ---------------------------------------------------------------------------


def test_active_only_lists_running(tracker: InMemoryTaskTracker) -> None:
    a = tracker.submit(_spec("a", predicate=AlwaysTrue()))
    tracker.submit(_spec("b", predicate=AlwaysFalse()))
    tracker.evaluate(a, _ctx())
    ids = {s.id for s in tracker.active()}
    assert ids == {"b"}


def test_history_is_bounded(tracker: InMemoryTaskTracker) -> None:
    for i in range(8):
        s = tracker.submit(_spec(f"t{i}", predicate=AlwaysTrue()))
        tracker.evaluate(s, _ctx())
    history = list(tracker.history())
    assert len(history) == tracker._cfg.history_size  # 4
    # Newest tasks are retained.
    assert history[-1].id == "t7"


def test_cancel_terminates(tracker: InMemoryTaskTracker) -> None:
    tracker.submit(_spec("a", predicate=AlwaysFalse()))
    out = tracker.cancel("a", reason="stopped by user")
    assert out.status == TaskStatus.CANCELLED
    assert out.last_error == "stopped by user"
    assert "a" not in {s.id for s in tracker.active()}


def test_cancel_unknown_raises(tracker: InMemoryTaskTracker) -> None:
    with pytest.raises(TaskTrackerError):
        tracker.cancel("missing")


def test_cancel_terminal_is_idempotent(tracker: InMemoryTaskTracker) -> None:
    state = tracker.submit(_spec("a", predicate=AlwaysTrue()))
    tracker.evaluate(state, _ctx())
    again = tracker.cancel("a")
    assert again.status == TaskStatus.COMPLETED  # not overwritten


def test_update_unknown_raises(tracker: InMemoryTaskTracker) -> None:
    with pytest.raises(TaskTrackerError):
        tracker.update("missing", TaskStatus.COMPLETED)


# ---------------------------------------------------------------------------
# evaluate_active (async snapshot)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_active_returns_snapshot(tracker: InMemoryTaskTracker) -> None:
    counter = itertools.count()

    def alternating(state, ctx):  # type: ignore[no-untyped-def]
        return next(counter) % 2 == 0  # accept first call

    a = tracker.submit(_spec("a", predicate=CallablePredicate(alternating)))
    b = tracker.submit(_spec("b", predicate=AlwaysFalse()))
    snapshot = await tracker.evaluate_active(_ctx())
    ids = {s.id for s in snapshot}
    assert ids == {"a", "b"}
    # 'a' should have completed on the first predicate call.
    assert a.status == TaskStatus.COMPLETED
    assert b.status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_evaluate_active_skips_terminal(tracker: InMemoryTaskTracker) -> None:
    state = tracker.submit(_spec("a", predicate=AlwaysTrue()))
    tracker.evaluate(state, _ctx())
    snapshot = await tracker.evaluate_active(_ctx())
    assert snapshot == ()
