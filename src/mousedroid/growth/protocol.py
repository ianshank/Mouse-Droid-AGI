"""Self-improvement / knowledge growth protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class GrowthProtocol(Protocol):
    """Interface for knowledge distillation / growth methods."""

    def distill_step(self, x: Tensor, hard_labels: Tensor) -> Tensor:
        """One distillation training step. Returns loss."""
        ...
