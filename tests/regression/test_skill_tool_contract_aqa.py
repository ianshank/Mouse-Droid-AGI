"""Automated Quality Assurance (AQA) — builtin skill / tool-registry contract.

``SkillSpec.tool_names`` is a whitelist that ``FilteredToolRegistry``
intersects with the parent registry, so a declared tool that nothing registers
is dropped rather than rejected: the skill advertises a capability it can never
dispatch, and no gate notices.

This is a **ratchet**, not a clean-slate assertion. The baseline below records
the unresolved names that already existed when the gate landed, each with the
reason it is still there. The gate fails on any *new* unresolved name, and also
fails when a baseline entry starts resolving — so fixing one forces deleting
its excuse rather than letting the list rot.
"""

from __future__ import annotations

import structlog

from mousedroid.common.tools.motor_tools import MotorToolDeps, register_motor_tools
from mousedroid.common.tools.registry import ToolRegistry, create_default_registry
from mousedroid.config.schema import Settings
from mousedroid.skills import builtin
from mousedroid.skills.contract import unresolved_tool_names
from mousedroid.skills.protocol import SkillSpec
from mousedroid.skills.registry import SkillRegistry

# Unresolved at the time this gate landed. Each entry is debt with a reason,
# not an approved permanent state.
_BASELINE_UNRESOLVED: dict[str, frozenset[str]] = {
    # No pose subsystem exists: no pose estimator, odometry source, scan
    # matcher or map anywhere in src/. The WAVE ROVER chassis is encoder-less
    # (vendor audit R3) so EncoderReading.odometry_* is structurally zero.
    # This tool cannot be implemented without a localization subsystem that
    # has never been scoped, so the claim is aspirational, not pending.
    "mousedroid-navigate": frozenset({"query_world_model_pose"}),
    # Same pose gap, plus two read-only introspection tools that were
    # specified with the skill but never registered.
    "mousedroid-world-model": frozenset(
        {"query_world_model_belief", "query_world_model_pose", "episodic_recent_summary"}
    ),
    # Sensor reads exist on the drivers but were never exposed as tools.
    "mousedroid-sensor-report": frozenset({"query_health", "read_battery", "read_distance"}),
    # The voice subsystem exists (src/mousedroid/voice/) but registers no tools.
    "mousedroid-voice": frozenset({"play_phrase", "speak_event"}),
}


class _NullEsp32:
    """Registration-only ESP32 stand-in — never dispatched by this gate."""

    async def send_velocity(self, vx: float, vy: float, omega: float) -> None:
        """No-op."""

    async def emergency_stop(self) -> None:
        """No-op."""

    async def read_encoders(self) -> None:
        """No-op."""


def _builtin_specs() -> list[SkillSpec]:
    """Every builtin skill spec that declares a tool whitelist."""
    specs: list[SkillSpec] = []
    for attr in sorted(dir(builtin)):
        candidate = getattr(builtin, attr)
        if isinstance(candidate, SkillSpec) and candidate.tool_names:
            specs.append(candidate)
    return specs


def _full_registry() -> ToolRegistry:
    """The MOUSE_DROID tool surface: default registry plus motor tools.

    ``factory/orchestrator.py`` gates ``register_motor_tools`` on
    ``cfg.platform == PlatformType.MOUSE_DROID``, so on the arm platform
    ``set_velocity`` is absent too. This gate measures the rover surface,
    which is the one the builtin skills are written against.
    """
    cfg = Settings.model_validate({"mock_hardware": True})
    registry = create_default_registry()
    register_motor_tools(registry, MotorToolDeps(esp32=_NullEsp32(), cfg=cfg))  # type: ignore[arg-type]
    return registry


def test_builtin_skills_are_discoverable() -> None:
    """Guards the gate itself: an empty sweep would pass everything."""
    specs = _builtin_specs()
    assert specs, "no builtin SkillSpec with tool_names found"
    assert {spec.name for spec in specs} >= set(_BASELINE_UNRESOLVED)


def test_no_new_unresolved_skill_tool_names() -> None:
    """A newly declared tool must be registered, or the skill must not claim it."""
    unresolved = unresolved_tool_names(_builtin_specs(), _full_registry())
    new = {
        skill: sorted(missing - _BASELINE_UNRESOLVED.get(skill, frozenset()))
        for skill, missing in unresolved.items()
        if missing - _BASELINE_UNRESOLVED.get(skill, frozenset())
    }
    assert not new, (
        f"skills declare tool names no registry provides: {new}. "
        "Register the tool, or drop it from the skill's tool_names."
    )


def test_baseline_has_no_stale_entries() -> None:
    """A fixed entry must be removed from the baseline, not left behind."""
    unresolved = unresolved_tool_names(_builtin_specs(), _full_registry())
    stale = {
        skill: sorted(expected - unresolved.get(skill, frozenset()))
        for skill, expected in _BASELINE_UNRESOLVED.items()
        if expected - unresolved.get(skill, frozenset())
    }
    assert not stale, (
        f"baseline lists tool names that now resolve: {stale}. "
        "Delete them from _BASELINE_UNRESOLVED."
    )


def test_unresolved_names_are_logged_not_swallowed() -> None:
    """Handing a skill its filtered registry must surface the missing tools.

    ``capture_logs`` swaps the processor chain for the block and restores it
    after, so this cannot leak configuration into sibling tests.
    """
    spec = next(s for s in _builtin_specs() if s.name in _BASELINE_UNRESOLVED)
    registry = SkillRegistry()
    registry.register(spec)
    parent = _full_registry()
    with structlog.testing.capture_logs() as logs:
        registry.tools_for(spec.name, parent)

    events = [entry for entry in logs if entry["event"] == "skill_tools_unresolved"]
    assert events, "unresolved tool names were dropped without a warning"
    assert events[0]["skill"] == spec.name
    assert set(events[0]["missing"]) == set(_BASELINE_UNRESOLVED[spec.name])
    assert events[0]["log_level"] == "warning"


def test_fully_resolved_skill_logs_nothing() -> None:
    """The audit stays quiet when every declared tool exists."""
    registry = SkillRegistry()
    spec = SkillSpec(name="fully-resolved", tool_names=frozenset({"set_velocity"}))
    registry.register(spec)
    parent = _full_registry()
    with structlog.testing.capture_logs() as logs:
        filtered = registry.tools_for(spec.name, parent)

    assert not [entry for entry in logs if entry["event"] == "skill_tools_unresolved"]
    assert filtered.names == ["set_velocity"]
