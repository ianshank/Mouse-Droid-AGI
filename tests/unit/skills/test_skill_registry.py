"""Tests for ``mousedroid.skills.registry``."""

from __future__ import annotations

import pytest

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
from mousedroid.skills.protocol import SkillRegistryProtocol, SkillSpec
from mousedroid.skills.registry import (
    FilteredToolRegistry,
    SkillRegistry,
    SkillRegistryError,
)


async def _h() -> str:
    return "ok"


@pytest.fixture
def parent_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for name in ("alpha", "beta", "gamma"):
        reg.register(ToolSpec(name=name, description="", handler=_h))
    return reg


@pytest.fixture
def registry() -> SkillRegistry:
    return SkillRegistry()


def test_implements_protocol(registry: SkillRegistry) -> None:
    assert isinstance(registry, SkillRegistryProtocol)


def test_register_and_get(registry: SkillRegistry) -> None:
    spec = SkillSpec(name="diag", description="diagnostics", tool_names=frozenset({"alpha"}))
    registry.register(spec)
    assert registry.get("diag") is spec
    assert registry.names() == ("diag",)
    assert len(registry) == 1


def test_register_empty_name_raises(registry: SkillRegistry) -> None:
    with pytest.raises(SkillRegistryError):
        registry.register(SkillSpec(name=""))


def test_register_duplicate_replaces(registry: SkillRegistry) -> None:
    registry.register(SkillSpec(name="x", description="first"))
    registry.register(SkillSpec(name="x", description="second"))
    assert registry.get("x").description == "second"
    assert len(registry) == 1


def test_get_missing_returns_none(registry: SkillRegistry) -> None:
    assert registry.get("ghost") is None


# ---------------------------------------------------------------------------
# FilteredToolRegistry
# ---------------------------------------------------------------------------


def test_tools_for_unknown_skill_raises(
    registry: SkillRegistry, parent_registry: ToolRegistry
) -> None:
    with pytest.raises(SkillRegistryError):
        registry.tools_for("nope", parent_registry)


def test_filtered_registry_exposes_only_whitelisted_tools(
    registry: SkillRegistry, parent_registry: ToolRegistry
) -> None:
    registry.register(SkillSpec(name="diag", tool_names=frozenset({"alpha", "beta"})))
    filtered = registry.tools_for("diag", parent_registry)
    assert isinstance(filtered, FilteredToolRegistry)
    assert set(filtered.names) == {"alpha", "beta"}
    assert filtered.get("alpha") is not None
    assert filtered.get("gamma") is None
    assert filtered.allowed == frozenset({"alpha", "beta"})


@pytest.mark.asyncio
async def test_filtered_registry_dispatch_allowed(
    registry: SkillRegistry, parent_registry: ToolRegistry
) -> None:
    registry.register(SkillSpec(name="x", tool_names=frozenset({"alpha"})))
    filtered = registry.tools_for("x", parent_registry)
    assert await filtered.dispatch("alpha") == "ok"


@pytest.mark.asyncio
async def test_filtered_registry_dispatch_outside_whitelist_raises(
    registry: SkillRegistry, parent_registry: ToolRegistry
) -> None:
    registry.register(SkillSpec(name="x", tool_names=frozenset({"alpha"})))
    filtered = registry.tools_for("x", parent_registry)
    with pytest.raises(KeyError):
        await filtered.dispatch("beta")


def test_filtered_registry_len(registry: SkillRegistry, parent_registry: ToolRegistry) -> None:
    registry.register(SkillSpec(name="x", tool_names=frozenset({"alpha", "beta"})))
    filtered = registry.tools_for("x", parent_registry)
    assert len(filtered) == 2


def test_filtered_registry_unknown_in_parent_returns_none(
    parent_registry: ToolRegistry,
) -> None:
    filtered = FilteredToolRegistry(parent_registry, ["alpha", "ghost"])
    assert filtered.get("ghost") is None  # not in parent
    assert filtered.get("alpha") is not None


# ---------------------------------------------------------------------------
# load_all
# ---------------------------------------------------------------------------


class _FakeLoader:
    def __init__(self, specs):  # type: ignore[no-untyped-def]
        self._specs = specs

    def load(self):  # type: ignore[no-untyped-def]
        yield from self._specs


def test_load_all_returns_count(registry: SkillRegistry) -> None:
    loaders = [
        _FakeLoader([SkillSpec(name="a")]),
        _FakeLoader([SkillSpec(name="b"), SkillSpec(name="c")]),
    ]
    count = registry.load_all(loaders)
    assert count == 3
    assert set(registry.names()) == {"a", "b", "c"}
