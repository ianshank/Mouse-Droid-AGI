"""Memory system protocols."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class MemoryProtocol(Protocol):
    """Interface for memory storage systems."""

    def store(self, key: str, value: NDArray[np.float32]) -> None:
        """Store a value in memory."""
        ...

    def retrieve(self, query: NDArray[np.float32], k: int = 1) -> list[Any]:
        """Retrieve k nearest matches."""
        ...


@runtime_checkable
class ReplayBufferProtocol(Protocol):
    """Interface for experience replay buffers."""

    def push(self, experience: Any, priority: float = 1.0) -> None:
        """Add experience with priority."""
        ...

    def sample(self, batch_size: int) -> list[Any]:
        """Sample a batch of experiences."""
        ...

    def __len__(self) -> int:
        """Current buffer size."""
        ...
