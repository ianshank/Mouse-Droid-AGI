"""Self-improvement / knowledge growth protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class GrowthProtocol(Protocol):
    """Interface for knowledge distillation / growth methods."""

    def distill_step(self, x: Tensor, hard_labels: Tensor | None = None) -> Tensor:
        """One distillation training step. Returns loss.

        ``hard_labels`` is required for classification-objective distillers and
        optional for regression-objective ones (``None`` = soft term only).
        """
        ...
