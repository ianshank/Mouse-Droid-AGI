"""Reward model protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class RewardModelProtocol(Protocol):
    """Interface for reward models."""

    def compute_reward(self, state: Tensor) -> dict[str, Tensor]:
        """Compute per-objective reward scores."""
        ...

    def aggregate(self, scores: dict[str, Tensor]) -> Tensor:
        """Aggregate per-objective scores into scalar reward."""
        ...
