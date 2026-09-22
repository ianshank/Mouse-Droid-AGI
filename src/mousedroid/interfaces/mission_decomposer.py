"""Mission decomposer protocol for off-loop NL command decomposition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from mousedroid.llm_gateway.mission_parser import MissionIntent

__all__ = ["MissionDecomposerProtocol"]


@runtime_checkable
class MissionDecomposerProtocol(Protocol):
    """Mission decomposer protocol for off-loop NL command decomposition."""

    @property
    def is_ready(self) -> bool:
        """Returns True if the decomposer is ready to accept commands.

        Returns:
            bool: Readiness state.
        """
        ...

    @property
    def is_degraded(self) -> bool:
        """Returns True if the decomposer is in a degraded state.

        Returns:
            bool: Degraded state.
        """
        ...

    async def decompose(self, command: str) -> list[MissionIntent]:
        """Decomposes a natural language command into mission intents.

        Args:
            command: The natural language command to decompose.

        Returns:
            list[MissionIntent]: The decomposed mission intents.
        """
        ...
