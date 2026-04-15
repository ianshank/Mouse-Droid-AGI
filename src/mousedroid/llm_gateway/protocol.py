"""LLM Gateway protocol — NL mission to velocity command translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class GoalVector:
    """3D velocity target from NL mission translation.

    All values normalised to ``[-1, 1]``.
    """

    vx_target: float = 0.0
    vy_target: float = 0.0
    omega_target: float = 0.0


@runtime_checkable
class LLMGatewayProtocol(Protocol):
    """Interface for NL -> velocity command translation."""

    @property
    def is_ready(self) -> bool:
        """Whether the gateway has a loaded model ready to serve translations."""
        ...

    async def start(self) -> None:
        """Load model and warm up. Raises RuntimeError if deps missing."""
        ...

    async def translate_mission(self, nl_command: str) -> GoalVector:
        """Translate NL mission description to a GoalVector.

        Args:
            nl_command: Natural language mission (must be non-empty).

        Returns:
            ``GoalVector`` with ``(vx_target, vy_target, omega_target)`` in ``[-1, 1]``.

        Raises:
            ValueError: If nl_command is empty or missing.
        """
        ...

    async def stop(self) -> None:
        """Unload model and release GPU memory."""
        ...
