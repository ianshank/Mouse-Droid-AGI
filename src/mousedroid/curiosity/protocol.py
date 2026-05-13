"""Curiosity module protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class CuriosityProtocol(Protocol):
    """Interface for curiosity-driven exploration modules."""

    def intrinsic_reward(self, s: Tensor, a: Tensor, s_next: Tensor) -> Tensor:
        """Compute intrinsic curiosity reward."""
        ...

    def reset_episode(self) -> None:
        """Reset per-episode accumulators (e.g. novelty counters, running stats).

        Called by the orchestrator on mission completion so that curiosity
        scores are not polluted by prior-episode experience.
        """
        ...
