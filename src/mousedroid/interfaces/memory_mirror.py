"""Memory mirror protocol for remote long-horizon memory sync."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["MemoryMirrorProtocol"]


@runtime_checkable
class MemoryMirrorProtocol(Protocol):
    """Memory mirror protocol for remote long-horizon memory sync."""

    @property
    def is_degraded(self) -> bool:
        """Returns True if the memory mirror is in a degraded state.

        Returns:
            bool: Degraded state.
        """
        ...

    async def sync_to_remote(self) -> int:
        """Synchronizes local memory to the remote backend.

        Returns:
            int: The number of entries successfully mirrored.
        """
        ...

    async def recall(self, query: str, *, k: int | None = None) -> list[str]:
        """Recalls memories based on a query.

        Args:
            query: The search query.
            k: The maximum number of results to return. If None, defaults
                to the adapter's configured maximum.

        Returns:
            list[str]: The recalled pre-sanitized text memories.
        """
        ...
