"""Class-surface characterization for the mixin-composed orchestrator (ADR-017).

`src/mousedroid/orchestrator/orchestrator.py` (2,191 lines, 44 methods on
`MouseDroidOrchestrator`) was split into `__init__` + `tick()` (the only two
methods left in the concrete class) plus 7 sibling `_*_mixin.py` files the
class now composes via Python's MRO.

Two mixins accidentally defining the same method name is a silent failure
mode: Python's MRO resolves the collision by keeping whichever mixin is
listed first in the class's base-class tuple and dropping the other with
*no error* -- the dropped method simply stops being callable in the way its
own file's tests expect. Neither `mypy --strict` nor `ruff` catches this
(each mixin file type-checks and lints cleanly on its own). This file is
the only thing that would catch it.
"""

from __future__ import annotations

import inspect

from mousedroid.orchestrator._action_mixin import _ActionMixin
from mousedroid.orchestrator._background_cadence_mixin import _BackgroundCadenceMixin
from mousedroid.orchestrator._lifecycle_mixin import _LifecycleMixin
from mousedroid.orchestrator._mission_mixin import _MissionMixin
from mousedroid.orchestrator._state import _OrchestratorState
from mousedroid.orchestrator._telemetry_experience_mixin import _TelemetryExperienceMixin
from mousedroid.orchestrator._voice_face_mixin import _VoiceFaceMixin
from mousedroid.orchestrator._world_model_state_mixin import _WorldModelStateMixin
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

# Captured from `sorted(n for n in dir(MouseDroidOrchestrator) if not
# n.startswith("__"))` immediately after the mixin split landed (ADR-017).
# A deliberate future change to the orchestrator's method surface (adding,
# removing, or renaming a method) is expected to touch this literal list in
# the same PR -- that's the point of a characterization test: it forces a
# conscious update instead of a silent drift.
_EXPECTED_PUBLIC_SURFACE = [
    "_apply_one_pending_update",
    "_apply_pending_weight_update",
    "_compute_curiosity_scores",
    "_consolidation_loop",
    "_drain_background_tasks",
    "_execute_action",
    "_growth_distill_loop",
    "_growth_enabled",
    "_log_experience",
    "_maybe_export_memory",
    "_maybe_fire_startup_greeting",
    "_maybe_project_action",
    "_maybe_rearm_latent_sink",
    "_maybe_reset_curiosity",
    "_maybe_tick_mission_lifecycle",
    "_normalize_cognitive_action",
    "_on_device_learning_enabled",
    "_on_device_update_loop",
    "_project_action_to_executable_axes",
    "_publish_raw_lidar",
    "_publish_telemetry",
    "_run_slow_cadence_loop",
    "_select_action",
    "_spawn_slow_background_tasks",
    "_start_cloud_subsystems",
    "_start_mission_lifecycle_if_wired",
    "_stop_cloud_subsystems",
    "_try_cognitive_action",
    "_try_sensor_recovery",
    "_try_vla_action",
    "_update_face",
    "_update_world_model",
    "_validate_latent",
    "_voice_event",
    "_voice_lifecycle",
    "_voice_observe",
    "dispatch_tool",
    "health_check",
    "process_mission",
    "run",
    "start",
    "stop",
    "tick",
]

_ALL_MIXINS = (
    _LifecycleMixin,
    _MissionMixin,
    _WorldModelStateMixin,
    _ActionMixin,
    _TelemetryExperienceMixin,
    _VoiceFaceMixin,
    _BackgroundCadenceMixin,
)


# The only two dunders Python auto-injects into every class's own __dict__
# regardless of what the class body declares (confirmed empirically against
# every real mixin here: each one's own vars() is exactly its real methods
# plus these two, nothing more). Excluding only these two -- not every name
# starting with "__" -- means a *real* dunder a mixin defines on purpose
# (__repr__, __eq__, __enter__, ...) stays visible to the collision check
# below instead of silently bypassing it the way a blanket "__" prefix
# filter would.
_ALWAYS_PRESENT_DUNDERS = frozenset({"__module__", "__doc__"})


def _own_methods(cls: type) -> set[str]:
    """Names defined directly on `cls.__dict__`, not inherited."""
    return {name for name in vars(cls) if name not in _ALWAYS_PRESENT_DUNDERS}


def test_orchestrator_class_surface_matches_pinned_snapshot() -> None:
    actual = sorted(n for n in dir(MouseDroidOrchestrator) if not n.startswith("__"))
    assert actual == _EXPECTED_PUBLIC_SURFACE, (
        "MouseDroidOrchestrator's method surface changed. If this is an "
        "intentional addition/removal/rename, update _EXPECTED_PUBLIC_SURFACE "
        "in this file in the same PR. If it's not intentional, a mixin may "
        "have silently shadowed another (see test below for the direct check)."
    )


def test_no_two_mixins_define_the_same_method_name() -> None:
    """Direct collision check, independent of the pinned list above.

    Compares the 7 mixins against each other only. `MouseDroidOrchestrator`
    itself is checked separately below -- Python's MRO always checks the
    concrete class's own `__dict__` before any base class, so a name
    collision between the concrete class and a mixin is a DIFFERENT failure
    mode (silent shadowing, not silent last-wins-among-bases) and needs its
    own assertion, not a shared one.
    """
    seen: dict[str, type] = {}
    collisions: list[str] = []
    for mixin in _ALL_MIXINS:
        for name in _own_methods(mixin):
            if name in seen and seen[name] is not mixin:
                collisions.append(
                    f"{name!r} defined in both {seen[name].__name__} and {mixin.__name__}"
                )
            else:
                seen[name] = mixin
    assert not collisions, (
        "two mixins define the same method name -- Python's MRO silently "
        f"keeps one and drops the other with no error: {collisions}"
    )


def test_concrete_class_defines_only_init_and_tick() -> None:
    """`MouseDroidOrchestrator.__dict__` itself must hold only `__init__`/`tick`.

    This is the one check in this file that actually enforces ADR-017's
    "orchestrator.py holds only __init__ and tick()" claim. It was missing
    from the first version of this file, which caught a real, live bug:
    orchestrator.py still directly defined 7 methods
    (start/_maybe_fire_startup_greeting/_start_cloud_subsystems/
    _spawn_slow_background_tasks/stop/_drain_background_tasks/
    _stop_cloud_subsystems) that duplicated `_lifecycle_mixin.py`'s
    versions. Because Python's MRO always checks the concrete class's own
    `__dict__` before any base class, the duplicates silently won and
    `_LifecycleMixin`'s copies became unreachable dead code -- invisible to
    `test_orchestrator_class_surface_matches_pinned_snapshot` (a name in
    `dir()` doesn't say which class in the MRO supplied it) and to
    `test_no_two_mixins_define_the_same_method_name` (which only compares
    mixins against each other, not against the concrete class).
    """
    own_names = _own_methods(MouseDroidOrchestrator)
    unexpected = sorted(own_names - {"__init__", "tick"})
    assert not unexpected, (
        f"MouseDroidOrchestrator defines method(s) directly that ADR-017 says "
        f"belong in a mixin instead: {unexpected}. Either remove the "
        f"duplicate from orchestrator.py (its mixin twin already provides it "
        f"via MRO) or, if this is a deliberate new concrete-class method, "
        f"revise this test's allowed set in the same PR."
    )


def test_every_mixin_method_is_reachable_on_the_concrete_class() -> None:
    """Every method a mixin defines must actually be reachable via MRO.

    A mixin left out of `MouseDroidOrchestrator`'s base-class tuple (a
    copy-paste miss when adding an 8th mixin later, say) would leave its
    methods defined but unreachable -- `getattr` would raise, not silently
    return a stale/wrong implementation, but this makes the failure explicit
    and points at the exact missing method rather than an opaque
    AttributeError deep in a caller.
    """
    unreachable: list[str] = []
    for mixin in _ALL_MIXINS:
        for name in _own_methods(mixin):
            if not hasattr(MouseDroidOrchestrator, name):
                unreachable.append(f"{mixin.__name__}.{name}")
    assert not unreachable, (
        f"mixin method(s) not reachable on MouseDroidOrchestrator: {unreachable} "
        "-- is the mixin missing from the class's base-class tuple?"
    )


def _state_stub_method_names() -> set[str]:
    """The cross-mixin method stubs `_OrchestratorState` declares for typing only.

    Filters to real function objects so the 61 bare attribute annotations
    (which never appear as `vars()` entries -- only in `__annotations__`) and
    the handful of dunders Python auto-injects onto `_OrchestratorState`
    itself (`__dict__`, `__weakref__`; it is the diamond's base, unlike the
    mixins, which inherit those from it) are both excluded without needing a
    hand-maintained list -- forward-looking the same way the checks above
    derive their expectations from real source instead of a frozen roster.
    """
    return {name for name, value in vars(_OrchestratorState).items() if inspect.isfunction(value)}


def test_every_state_stub_is_overridden_somewhere_reachable() -> None:
    """Every `_OrchestratorState` stub must have a real implementor, not just a type.

    `_OrchestratorState` sits last in the MRO (diamond base, right before
    `object`), so today every real mixin implementation wins over its stub.
    But every stub now raises `NotImplementedError` (never a silent `...`
    no-op -- see `_state.py`'s own comment), specifically so that IF a
    future rename or deletion ever left one un-overridden, the failure would
    be loud at runtime. This test makes the same class of drift loud at
    review time instead of runtime: it fails the moment a stub's real
    implementor disappears, rather than waiting for that code path to
    execute in production and raise. Complements
    `test_no_two_mixins_define_the_same_method_name` (at most one mixin
    implementor) with the missing other half (at least one implementor,
    counting the concrete class too -- `tick` is intentionally only ever
    defined there, per ADR-014).
    """
    implementors: dict[str, type] = dict.fromkeys(
        _own_methods(MouseDroidOrchestrator), MouseDroidOrchestrator
    )
    for mixin in _ALL_MIXINS:
        for name in _own_methods(mixin):
            implementors.setdefault(name, mixin)
    unimplemented = sorted(_state_stub_method_names() - implementors.keys())
    assert not unimplemented, (
        f"_state.py declares stub(s) {unimplemented} that neither the concrete "
        "class nor any mixin defines directly -- any caller would resolve to "
        "_OrchestratorState's own `raise NotImplementedError` stub."
    )
