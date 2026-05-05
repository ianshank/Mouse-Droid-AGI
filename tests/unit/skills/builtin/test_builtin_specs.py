"""Unit tests for the four OpenClaw builtin :class:`SkillSpec` packages."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
from mousedroid.skills.builtin import (
    NAVIGATE_SPEC,
    SENSOR_REPORT_SPEC,
    VOICE_SPEC,
    WORLD_MODEL_SPEC,
    all_builtin_specs,
)
from mousedroid.skills.builtin.navigate import NavigateInput
from mousedroid.skills.builtin.sensor_report import SensorReportInput
from mousedroid.skills.builtin.voice import VoiceInput
from mousedroid.skills.builtin.world_model import WorldModelInput
from mousedroid.skills.registry import FilteredToolRegistry, SkillRegistry


def test_all_builtin_specs_returns_four_in_stable_order() -> None:
    specs = all_builtin_specs()
    assert len(specs) == 4
    assert [s.name for s in specs] == [
        "mousedroid-navigate",
        "mousedroid-sensor-report",
        "mousedroid-voice",
        "mousedroid-world-model",
    ]


@pytest.mark.parametrize(
    "spec",
    [NAVIGATE_SPEC, SENSOR_REPORT_SPEC, VOICE_SPEC, WORLD_MODEL_SPEC],
)
def test_tool_names_is_frozenset(spec: Any) -> None:
    assert isinstance(spec.tool_names, frozenset)
    assert len(spec.tool_names) >= 1


@pytest.mark.parametrize(
    "spec",
    [NAVIGATE_SPEC, SENSOR_REPORT_SPEC, VOICE_SPEC, WORLD_MODEL_SPEC],
)
def test_metadata_has_required_keys(spec: Any) -> None:
    md = spec.metadata
    assert "actuation" in md
    assert isinstance(md["actuation"], bool)
    assert "version" in md
    assert "channel" in md


def test_only_navigate_is_actuation() -> None:
    """Only the navigation skill is allowed to actuate motors."""
    assert NAVIGATE_SPEC.metadata["actuation"] is True
    assert SENSOR_REPORT_SPEC.metadata["actuation"] is False
    assert VOICE_SPEC.metadata["actuation"] is False
    assert WORLD_MODEL_SPEC.metadata["actuation"] is False


def test_navigate_input_validates_speed_bounds() -> None:
    NavigateInput(nl_command="go", max_speed=0.5)
    with pytest.raises(ValueError, match="max_speed"):
        NavigateInput(nl_command="go", max_speed=1.5)
    with pytest.raises(ValueError, match="nl_command"):
        NavigateInput(nl_command="", max_speed=0.5)


def test_sensor_report_input_defaults_include_everything() -> None:
    cfg = SensorReportInput()
    assert cfg.include_lidar is True
    assert cfg.include_imu is True
    assert cfg.include_battery is True


def test_voice_input_requires_exactly_one_of_event_or_text() -> None:
    VoiceInput(event="greeting")
    VoiceInput(text="hello")
    with pytest.raises(ValueError, match="exactly one"):
        VoiceInput()
    with pytest.raises(ValueError, match="exactly one"):
        VoiceInput(event="greeting", text="hello")


def test_world_model_input_window_bounds() -> None:
    WorldModelInput(episodic_window=0)
    WorldModelInput(episodic_window=512)
    with pytest.raises(ValueError, match="episodic_window"):
        WorldModelInput(episodic_window=-1)
    with pytest.raises(ValueError, match="episodic_window"):
        WorldModelInput(episodic_window=513)


def _registry_with(*names: str) -> ToolRegistry:
    reg = ToolRegistry()

    async def _ok() -> dict[str, str]:
        return {"status": "ok"}

    for n in names:
        reg.register(ToolSpec(n, n, _ok))
    return reg


@pytest.mark.parametrize(
    "spec",
    [NAVIGATE_SPEC, SENSOR_REPORT_SPEC, VOICE_SPEC, WORLD_MODEL_SPEC],
)
def test_filtered_registry_dispatches_only_whitelisted_tools(spec: Any) -> None:
    parent = _registry_with(*spec.tool_names, "off_whitelist")
    skills = SkillRegistry()
    skills.register(spec)
    filtered = skills.tools_for(spec.name, parent)
    assert isinstance(filtered, FilteredToolRegistry)
    for name in spec.tool_names:
        assert filtered.get(name) is not None
    assert filtered.get("off_whitelist") is None


@settings(max_examples=20, deadline=None)
@given(
    extra=st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=20,
    ),
)
def test_property_filtered_registry_rejects_random_unknown_tools(extra: str) -> None:
    """Property: dispatching ANY name not in the whitelist raises KeyError."""
    spec = NAVIGATE_SPEC
    if extra in spec.tool_names:
        # Hypothesis may generate a name that happens to be on the
        # whitelist; skip those cases since the contract only governs
        # *out-of-whitelist* names.
        return

    import asyncio

    async def _drive() -> None:
        parent = _registry_with(*spec.tool_names, extra)
        skills = SkillRegistry()
        skills.register(spec)
        filtered = skills.tools_for(spec.name, parent)
        with pytest.raises(KeyError):
            await filtered.dispatch(extra)

    asyncio.run(_drive())
