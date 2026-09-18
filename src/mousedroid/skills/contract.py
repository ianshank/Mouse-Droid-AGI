"""Skill-to-tool contract auditing.

A :class:`~mousedroid.skills.protocol.SkillSpec` declares ``tool_names`` as a
*whitelist*: :class:`~mousedroid.skills.registry.FilteredToolRegistry`
intersects it with the parent registry, so a declared name that no registry
registers is not an error — it simply disappears. The skill keeps advertising a
capability it can never dispatch, and nothing says so.

This module makes that intersection inspectable, so the loss can be logged at
wiring time and pinned by a regression gate instead of being discovered from a
sub-agent that silently does nothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from mousedroid.logging.setup import get_logger
from mousedroid.skills.protocol import SkillSpec

_log = get_logger(__name__)


@runtime_checkable
class ToolNameSource(Protocol):
    """Anything that can report the tool names it can dispatch."""

    @property
    def names(self) -> list[str]:
        """Registered tool names."""
        ...


def unresolved_tool_names(
    specs: Iterable[SkillSpec], source: ToolNameSource
) -> dict[str, frozenset[str]]:
    """Map each skill to the tool names it declares but ``source`` lacks.

    Skills whose whitelist resolves completely are omitted, so an empty
    mapping means every declared tool exists.

    Args:
        specs: Skill specifications to audit.
        source: Registry (or any name source) the skills dispatch through.

    Returns:
        ``{skill_name: frozenset_of_missing_tool_names}`` for skills with at
        least one unresolved name.
    """
    names = source.names
    if callable(names):
        # ``SkillRegistry.names`` is a *method*; ``ToolRegistry.names`` is a
        # property. ``@runtime_checkable`` only checks attribute presence, so
        # both satisfy ``ToolNameSource`` and the wrong one would raise
        # ``TypeError`` here. Skill wiring must degrade, never break.
        names = names()
    available = frozenset(names)
    unresolved: dict[str, frozenset[str]] = {}
    for spec in specs:
        missing = frozenset(spec.tool_names) - available
        if missing:
            unresolved[spec.name] = missing
    return unresolved


def log_unresolved_tool_names(
    specs: Iterable[SkillSpec], source: ToolNameSource
) -> dict[str, frozenset[str]]:
    """Audit and emit one WARNING per skill with unresolved tool names.

    Returns the same mapping as :func:`unresolved_tool_names` so callers can
    act on it as well as have it logged.
    """
    unresolved = unresolved_tool_names(specs, source)
    for skill_name, missing in sorted(unresolved.items()):
        _log.warning(
            "skill_tools_unresolved",
            skill=skill_name,
            missing=sorted(missing),
            detail="declared in tool_names but registered by no tool registry",
        )
    return unresolved


__all__ = ["ToolNameSource", "log_unresolved_tool_names", "unresolved_tool_names"]
