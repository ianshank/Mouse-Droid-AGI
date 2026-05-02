"""Tests for ``mousedroid.harness.protocol`` — TaskSpec / TaskState invariants."""

from __future__ import annotations

import pytest

from mousedroid.harness.predicates import (
    AllOf,
    AlwaysFalse,
    AlwaysTrue,
    AnyOf,
    CallablePredicate,
    Negate,
    ObservationFieldEquals,
    TickCountReached,
)
from mousedroid.harness.protocol import (
    HookPhase,
    TaskSpec,
    TaskState,
    TaskStatus,
    TickContext,
    is_terminal,
)


def _ctx(tick: int = 0, observation=None) -> TickContext:
    return TickContext(tick_index=tick, timestamp_s=float(tick), observation=observation)


_DEFAULT_PREDICATE = AlwaysTrue()


def _state(predicate=_DEFAULT_PREDICATE, **kwargs) -> TaskState:
    spec = TaskSpec(id=kwargs.pop("id", "t1"), goal="g", acceptance_predicate=predicate)
    return TaskState(spec=spec, **kwargs)


# ---------------------------------------------------------------------------
# TaskStatus / is_terminal
# ---------------------------------------------------------------------------


def test_task_status_values_match_strings() -> None:
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.COMPLETED.value == "completed"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (TaskStatus.PENDING, False),
        (TaskStatus.RUNNING, False),
        (TaskStatus.COMPLETED, True),
        (TaskStatus.FAILED, True),
        (TaskStatus.TIMED_OUT, True),
        (TaskStatus.CANCELLED, True),
    ],
)
def test_is_terminal(status: TaskStatus, expected: bool) -> None:
    assert is_terminal(status) is expected


def test_hook_phase_values() -> None:
    assert HookPhase.PRE_TICK.value == "pre_tick"
    assert HookPhase.ON_ERROR.value == "on_error"


# ---------------------------------------------------------------------------
# TaskSpec immutability
# ---------------------------------------------------------------------------


def test_task_spec_is_frozen() -> None:
    import dataclasses

    spec = TaskSpec(id="t1", goal="navigate", acceptance_predicate=AlwaysTrue())
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.id = "other"  # type: ignore[misc]


def test_task_spec_default_constraints_and_metadata() -> None:
    spec = TaskSpec(id="t1", goal="g", acceptance_predicate=AlwaysTrue())
    assert spec.constraints == ()
    assert spec.metadata == {}
    assert spec.timeout_s is None
    assert spec.parent_id is None


def test_task_state_is_terminal_property() -> None:
    state = _state(status=TaskStatus.COMPLETED)
    assert state.is_terminal
    state2 = _state(status=TaskStatus.RUNNING)
    assert not state2.is_terminal


def test_task_state_id_proxy_returns_spec_id() -> None:
    state = _state(id="alpha")
    assert state.id == "alpha"


# ---------------------------------------------------------------------------
# Predicate combinators
# ---------------------------------------------------------------------------


def test_always_true_and_always_false() -> None:
    assert AlwaysTrue()(_state(), _ctx())
    assert not AlwaysFalse()(_state(), _ctx())


def test_negate_inverts() -> None:
    p = Negate(AlwaysTrue())
    assert not p(_state(), _ctx())


def test_all_of_requires_every_predicate() -> None:
    p = AllOf((AlwaysTrue(), AlwaysTrue()))
    assert p(_state(), _ctx())
    p2 = AllOf((AlwaysTrue(), AlwaysFalse()))
    assert not p2(_state(), _ctx())


def test_any_of_short_circuits_on_match() -> None:
    p = AnyOf((AlwaysFalse(), AlwaysTrue()))
    assert p(_state(), _ctx())
    p2 = AnyOf((AlwaysFalse(), AlwaysFalse()))
    assert not p2(_state(), _ctx())


def test_callable_predicate_wraps_lambda() -> None:
    p = CallablePredicate(lambda s, c: c.tick_index > 5)
    assert not p(_state(), _ctx(tick=1))
    assert p(_state(), _ctx(tick=10))


def test_observation_field_equals_dict() -> None:
    p = ObservationFieldEquals("battery_v", 12)
    assert p(_state(), _ctx(observation={"battery_v": 12}))
    assert not p(_state(), _ctx(observation={"battery_v": 9}))


def test_observation_field_equals_object_attr() -> None:
    class _Obs:
        def __init__(self, v: int) -> None:
            self.battery_v = v

    p = ObservationFieldEquals("battery_v", 12)
    assert p(_state(), _ctx(observation=_Obs(12)))


def test_observation_field_equals_none_obs() -> None:
    p = ObservationFieldEquals("battery_v", 12)
    assert not p(_state(), _ctx(observation=None))


def test_tick_count_reached_requires_metadata() -> None:
    spec = TaskSpec(
        id="t",
        goal="",
        acceptance_predicate=AlwaysTrue(),
        metadata={"submitted_at_tick": 5},
    )
    state = TaskState(spec=spec, status=TaskStatus.RUNNING)
    p = TickCountReached(n=3)
    assert not p(state, _ctx(tick=7))
    assert p(state, _ctx(tick=8))


def test_tick_count_reached_without_metadata_is_false() -> None:
    spec = TaskSpec(id="t", goal="", acceptance_predicate=AlwaysTrue())
    state = TaskState(spec=spec)
    p = TickCountReached(n=1)
    assert not p(state, _ctx(tick=100))


def test_tick_count_reached_uses_started_at_tick_field() -> None:
    """When ``TaskSpec.metadata`` lacks ``submitted_at_tick`` but the
    tracker has populated ``state.started_at_tick``, the predicate should
    still fire correctly (the tracker auto-populates this on first
    evaluation, so callers no longer need to thread metadata manually)."""
    spec = TaskSpec(id="t", goal="", acceptance_predicate=AlwaysTrue())
    state = TaskState(spec=spec, started_at_tick=10)
    p = TickCountReached(n=3)
    assert not p(state, _ctx(tick=12))  # delta = 2
    assert p(state, _ctx(tick=13))  # delta = 3
