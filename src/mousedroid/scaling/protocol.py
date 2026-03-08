"""Scaling protocol for sparse efficiency modules."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class ScalingProtocol(Protocol):
    """Interface for scaling / efficiency layers."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with sparse routing."""
        ...
