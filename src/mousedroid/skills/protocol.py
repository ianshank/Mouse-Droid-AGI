"""Protocols and dataclasses for the skill / sub-agent registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


@dataclass(frozen=True)
class SkillSpec:
    """Static specification of a sub-agent skill.

    Attributes:
        name: Unique skill identifier.
        description: Human-readable description (used in tool listings).
        tool_names: Whitelist of tools the skill may dispatch. ``frozenset``
            to make :class:`SkillSpec` hashable / immutable.
        system_prompt: Prompt scaffolding for the skill's sub-agent.
        schema_in: Optional Pydantic model validating the task payload.
        schema_out: Optional Pydantic model validating sub-agent output.
        source: Where the skill was loaded from (``manifest|markdown|code``).
        metadata: Arbitrary tags (author, version, capabilities, ...).
    """

    name: str
    description: str = ""
    tool_names: frozenset[str] = frozenset()
    system_prompt: str = ""
    schema_in: type[BaseModel] | None = None
    schema_out: type[BaseModel] | None = None
    source: str = "code"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubAgentResult:
    """The outcome of invoking a sub-agent for a task."""

    task_id: str
    status: str
    output: Any = None
    journal_refs: tuple[int, ...] = ()
    latency_ms: float = 0.0
    error: str | None = None


@runtime_checkable
class SubAgentProtocol(Protocol):
    """A sub-agent that executes one task end-to-end.

    Distinct from the tensor-level :class:`mousedroid.agents.base.AgentProtocol`
    used by the navigation policy — this protocol describes the higher-level
    task-oriented surface used by the harness's :class:`SkillDelegator`.
    """

    @property
    def name(self) -> str:
        """Stable sub-agent identifier (e.g. the skill name)."""
        ...

    @property
    def is_busy(self) -> bool:
        """True while a task is in flight."""
        ...

    async def invoke(self, spec: Any, parent_ctx: Any | None = None) -> SubAgentResult:
        """Execute ``spec`` end-to-end and return the outcome."""
        ...

    def cancel(self) -> None:
        """Best-effort cancellation; safe to call when idle."""
        ...


@runtime_checkable
class SkillRegistryProtocol(Protocol):
    """Registry of :class:`SkillSpec`s available for delegation."""

    def register(self, spec: SkillSpec) -> None:
        """Add or replace a :class:`SkillSpec` in the registry."""
        ...

    def get(self, name: str) -> SkillSpec | None:
        """Return the spec registered as ``name`` or ``None``."""
        ...

    def names(self) -> tuple[str, ...]:
        """Return registered skill names in registration order."""
        ...

    def __len__(self) -> int:
        """Number of registered skills."""
        ...


@runtime_checkable
class SkillLoaderProtocol(Protocol):
    """Load skill specs from a backing source (YAML, Markdown, code)."""

    def load(self) -> Iterable[SkillSpec]:
        """Yield every skill spec produced by this loader."""
        ...


__all__ = [
    "SkillLoaderProtocol",
    "SkillRegistryProtocol",
    "SkillSpec",
    "SubAgentProtocol",
    "SubAgentResult",
]
