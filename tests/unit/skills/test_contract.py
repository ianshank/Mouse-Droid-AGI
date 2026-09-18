"""Unit tests for :mod:`mousedroid.skills.contract`."""

from __future__ import annotations

import structlog

from mousedroid.skills.contract import (
    ToolNameSource,
    log_unresolved_tool_names,
    unresolved_tool_names,
)
from mousedroid.skills.protocol import SkillSpec


class _NameSource:
    """Minimal :class:`ToolNameSource` over a fixed name list."""

    def __init__(self, *names: str) -> None:
        self._names = list(names)

    @property
    def names(self) -> list[str]:
        return list(self._names)


def test_name_source_satisfies_the_protocol() -> None:
    """``runtime_checkable`` only proves attribute presence — check the value too."""
    source = _NameSource("alpha")
    assert isinstance(source, ToolNameSource)
    assert source.names == ["alpha"]


def test_method_style_name_source_is_tolerated() -> None:
    """``SkillRegistry.names`` is a method, ``ToolRegistry.names`` a property.

    ``@runtime_checkable`` accepts both, so the audit must not raise on the
    method form — skill wiring degrades, never breaks.
    """

    class _MethodStyle:
        def names(self) -> tuple[str, ...]:
            return ("alpha",)

    spec = SkillSpec(name="partial", tool_names=frozenset({"alpha", "missing"}))
    assert unresolved_tool_names([spec], _MethodStyle()) == {  # type: ignore[arg-type]
        "partial": frozenset({"missing"})
    }


def test_fully_resolved_skill_is_omitted() -> None:
    """An empty mapping is the 'everything resolves' signal."""
    spec = SkillSpec(name="ok", tool_names=frozenset({"alpha", "beta"}))
    assert unresolved_tool_names([spec], _NameSource("alpha", "beta", "gamma")) == {}


def test_missing_names_are_reported_per_skill() -> None:
    """Only the names absent from the source come back."""
    spec = SkillSpec(name="partial", tool_names=frozenset({"alpha", "missing"}))
    assert unresolved_tool_names([spec], _NameSource("alpha")) == {
        "partial": frozenset({"missing"})
    }


def test_skill_without_tool_names_is_omitted() -> None:
    """A skill that declares nothing cannot be in breach of the contract."""
    assert unresolved_tool_names([SkillSpec(name="bare")], _NameSource()) == {}


def test_multiple_skills_are_reported_independently() -> None:
    """One skill's breach does not mask or merge into another's."""
    specs = [
        SkillSpec(name="first", tool_names=frozenset({"missing_one"})),
        SkillSpec(name="second", tool_names=frozenset({"alpha"})),
        SkillSpec(name="third", tool_names=frozenset({"missing_two", "alpha"})),
    ]
    assert unresolved_tool_names(specs, _NameSource("alpha")) == {
        "first": frozenset({"missing_one"}),
        "third": frozenset({"missing_two"}),
    }


def test_empty_spec_iterable_returns_empty_mapping() -> None:
    """Degenerate input must not raise."""
    assert unresolved_tool_names([], _NameSource("alpha")) == {}


def test_log_helper_emits_one_warning_per_breaching_skill() -> None:
    """Each breach is individually actionable, so each gets its own event."""
    specs = [
        SkillSpec(name="first", tool_names=frozenset({"missing_one"})),
        SkillSpec(name="clean", tool_names=frozenset({"alpha"})),
        SkillSpec(name="second", tool_names=frozenset({"missing_two"})),
    ]
    with structlog.testing.capture_logs() as logs:
        result = log_unresolved_tool_names(specs, _NameSource("alpha"))

    events = [entry for entry in logs if entry["event"] == "skill_tools_unresolved"]
    assert [entry["skill"] for entry in events] == ["first", "second"]
    assert all(entry["log_level"] == "warning" for entry in events)
    assert result == {
        "first": frozenset({"missing_one"}),
        "second": frozenset({"missing_two"}),
    }


def test_log_helper_is_silent_when_everything_resolves() -> None:
    """No warning noise on a healthy wiring."""
    spec = SkillSpec(name="ok", tool_names=frozenset({"alpha"}))
    with structlog.testing.capture_logs() as logs:
        assert log_unresolved_tool_names([spec], _NameSource("alpha")) == {}
    assert not [entry for entry in logs if entry["event"] == "skill_tools_unresolved"]
