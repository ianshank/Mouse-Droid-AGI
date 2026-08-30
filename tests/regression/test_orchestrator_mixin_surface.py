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

from mousedroid.orchestrator._action_mixin import _ActionMixin
from mousedroid.orchestrator._background_cadence_mixin import _BackgroundCadenceMixin
from mousedroid.orchestrator._lifecycle_mixin import _LifecycleMixin
from mousedroid.orchestrator._mission_mixin import _MissionMixin
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


def _own_methods(cls: type) -> set[str]:
    """Names defined directly on `cls.__dict__`, not inherited."""
    return {name for name in vars(cls) if not name.startswith("__")}


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
