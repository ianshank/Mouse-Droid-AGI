"""Agent protocol — base interface for all MouseDroid agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor

from mousedroid.safety.context import SafetyContext


@runtime_checkable
class AgentProtocol(Protocol):
    """Base protocol for all MouseDroid agents."""

    @property
    def name(self) -> str:
        """Agent identifier."""
        ...

    def act(
        self,
        h: Tensor,
        z: Tensor,
        safety_ctx: SafetyContext,
    ) -> Tensor:
        """Select action. Returns shape ``(action_dim,)``, values in ``[-1, 1]``."""
        ...

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        ...
