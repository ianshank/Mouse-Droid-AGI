"""Continual learning protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class ContinualLearnerProtocol(Protocol):
    """Interface for continual learning methods."""

    def compute_penalty(self) -> Tensor:
        """Compute regularization penalty for current parameters."""
        ...

    def consolidate(self) -> None:
        """Consolidate current task knowledge."""
        ...
