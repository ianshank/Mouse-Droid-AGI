"""Efficiency protocol for TensorRT and power profiling."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch.nn as nn
from torch import Tensor


@runtime_checkable
class EfficiencyProtocol(Protocol):
    """Interface for model optimization and profiling."""

    def optimize(self, model: nn.Module, sample_input: Tensor) -> Any:
        """Optimize model for deployment."""
        ...
