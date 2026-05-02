"""Property tests for ``InMemoryTaskTracker``.

Hypothesis explores arbitrary submit/cancel/evaluate sequences and asserts
the tracker's invariants:

* History is bounded by ``history_size``.
* Active set never exceeds ``max_active``.
* No terminal task ever appears in ``active()``.
* Cancelling a terminal task does not change its status.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import HarnessTrackerConfig
from mousedroid.harness.predicates import AlwaysFalse, AlwaysTrue
from mousedroid.harness.protocol import TaskSpec, TaskStatus, TickContext
from mousedroid.harness.task_tracker import InMemoryTaskTracker, TaskTrackerError


def _ctx(tick: int = 0) -> TickContext:
    return TickContext(tick_index=tick, timestamp_s=float(tick))


@given(
    actions=st.lists(
        st.tuples(
            st.sampled_from(["submit_true", "submit_false", "evaluate", "cancel"]),
            st.text(alphabet="abcdef", min_size=1, max_size=2),
        ),
        max_size=40,
    ),
    history_size=st.integers(min_value=1, max_value=8),
    max_active=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=80, deadline=None)
def test_tracker_invariants_under_random_sequences(
    actions: list[tuple[str, str]],
    history_size: int,
    max_active: int,
) -> None:
    cfg = HarnessTrackerConfig(
        enabled=True,
        history_size=history_size,
        default_timeout_s=999.0,
        max_active=max_active,
    )
    tracker = InMemoryTaskTracker(cfg)

    for op, key in actions:
        try:
            if op == "submit_true":
                tracker.submit(TaskSpec(id=key, goal="g", acceptance_predicate=AlwaysTrue()))
            elif op == "submit_false":
                tracker.submit(TaskSpec(id=key, goal="g", acceptance_predicate=AlwaysFalse()))
            elif op == "evaluate":
                state = tracker.get(key)
                if state is not None:
                    tracker.evaluate(state, _ctx())
            elif op == "cancel" and tracker.get(key) is not None:
                tracker.cancel(key)
        except TaskTrackerError:
            # Allowed: duplicate ids, capacity limits.
            pass

        # ---------------- invariants ----------------
        active = list(tracker.active())
        assert len(active) <= max_active
        for state in active:
            assert state.status in {TaskStatus.PENDING, TaskStatus.RUNNING}
        history = list(tracker.history())
        assert len(history) <= history_size
        for state in history:
            assert state.is_terminal


@given(st.text(alphabet="abc", min_size=1, max_size=2))
@settings(max_examples=40, deadline=None)
def test_cancel_is_idempotent_for_terminal_tasks(task_id: str) -> None:
    cfg = HarnessTrackerConfig(enabled=True, history_size=8, max_active=4)
    tracker = InMemoryTaskTracker(cfg)
    spec = TaskSpec(id=task_id, goal="g", acceptance_predicate=AlwaysTrue())
    state = tracker.submit(spec)
    tracker.evaluate(state, _ctx())
    assert state.status == TaskStatus.COMPLETED
    again = tracker.cancel(task_id)
    assert again.status == TaskStatus.COMPLETED  # unchanged
