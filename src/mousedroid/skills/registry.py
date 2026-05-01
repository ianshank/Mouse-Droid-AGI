"""Concrete :class:`SkillRegistry` and a filtered tool-registry view.

The registry is hot-loaded from :class:`SkillLoaderProtocol` instances by
the factory; sub-agents only see the slice of the parent ``ToolRegistry``
that their skill explicitly whitelists.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
from mousedroid.logging.setup import get_logger
from mousedroid.skills.protocol import (
    SkillLoaderProtocol,
    SkillRegistryProtocol,
    SkillSpec,
)

_log = get_logger(__name__)


class SkillRegistryError(RuntimeError):
    """Raised when the skill registry is asked to do something invalid."""


class FilteredToolRegistry:
    """Read-only view of a :class:`ToolRegistry` exposing only whitelisted tools.

    Sub-agents are handed an instance of this class so they cannot reach
    tools outside their skill's whitelist. Dispatching an out-of-whitelist
    tool raises ``KeyError`` exactly like dispatching an unregistered tool
    on the parent registry.
    """

    def __init__(self, parent: ToolRegistry, allowed: Iterable[str]) -> None:
        self._parent = parent
        self._allowed = frozenset(allowed)

    def get(self, name: str) -> ToolSpec | None:
        if name not in self._allowed:
            return None
        return self._parent.get(name)

    @property
    def names(self) -> list[str]:
        return [n for n in self._parent.names if n in self._allowed]

    @property
    def allowed(self) -> frozenset[str]:
        return self._allowed

    def __len__(self) -> int:
        return len(self.names)

    async def dispatch(self, name: str, **kwargs: Any) -> Any:
        if name not in self._allowed:
            msg = (
                f"Tool {name!r} is not whitelisted by this skill (allowed={sorted(self._allowed)})"
            )
            _log.warning(
                "filtered_tool_dispatch_rejected",
                name=name,
                allowed=sorted(self._allowed),
            )
            raise KeyError(msg)
        _log.debug("filtered_tool_dispatch", name=name)
        return await self._parent.dispatch(name, **kwargs)


class SkillRegistry:
    """In-memory map of skill name → :class:`SkillSpec`."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> None:
        if not spec.name:
            msg = "Skill name must be a non-empty string"
            raise SkillRegistryError(msg)
        if spec.name in self._skills:
            _log.warning("skill_replaced", name=spec.name)
        self._skills[spec.name] = spec
        _log.debug(
            "skill_registered",
            name=spec.name,
            tools=sorted(spec.tool_names),
            source=spec.source,
        )

    def load_all(self, loaders: Iterable[SkillLoaderProtocol]) -> int:
        """Eagerly drain every loader; returns the count of new skills."""
        count = 0
        for loader in loaders:
            for spec in loader.load():
                self.register(spec)
                count += 1
        _log.info("skill_registry_loaded", count=count)
        return count

    def get(self, name: str) -> SkillSpec | None:
        return self._skills.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._skills.keys())

    def __len__(self) -> int:
        return len(self._skills)

    def tools_for(self, name: str, parent: ToolRegistry) -> FilteredToolRegistry:
        """Return a :class:`FilteredToolRegistry` over ``parent``."""
        spec = self._skills.get(name)
        if spec is None:
            msg = f"Unknown skill: {name!r}"
            raise SkillRegistryError(msg)
        return FilteredToolRegistry(parent, spec.tool_names)


_PROTOCOL_CHECK: SkillRegistryProtocol = SkillRegistry()
del _PROTOCOL_CHECK


__all__ = ["FilteredToolRegistry", "SkillRegistry", "SkillRegistryError"]
