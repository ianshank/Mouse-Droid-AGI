"""TensorRT model optimization for Jetson deployment (Pillar 10)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch.nn as nn
from torch import Tensor

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import JetsonConfig

_log = get_logger(__name__)


class TensorRTOptimizer:
    """Convert PyTorch models to TensorRT for efficient Jetson inference.

    Precision and workspace size read from ``JetsonConfig``.
    """

    def __init__(self, cfg: JetsonConfig) -> None:
        """Initialise TensorRT optimizer.

        Args:
            cfg: Jetson hardware configuration.
        """
        self.enabled = cfg.tensorrt_enabled
        self.precision = cfg.precision
        self.workspace_gb = cfg.workspace_gb
        self.dla_enabled = cfg.dla_enabled

    def optimize(self, model: nn.Module, sample_input: Tensor) -> Any:
        """Convert model to TensorRT engine.

        Args:
            model: PyTorch model to optimize.
            sample_input: Sample input tensor for tracing.

        Returns:
            Optimized model or engine. Returns original model if disabled.
        """
        if not self.enabled:
            _log.info("tensorrt_disabled")
            return model

        _log.info(
            "tensorrt_optimize",
            precision=self.precision,
            workspace_gb=self.workspace_gb,
            dla=self.dla_enabled,
        )
        return self._compile(model, sample_input)

    def _compile(self, model: nn.Module, sample_input: Tensor) -> Any:  # pragma: no cover
        """Compile model to TensorRT (requires tensorrt package).

        Args:
            model: PyTorch model.
            sample_input: Sample input for tracing.

        Returns:
            TensorRT engine or traced model.
        """
        try:
            import torch_tensorrt

            return torch_tensorrt.compile(
                model,
                inputs=[sample_input],
                enabled_precisions={self.precision},
                workspace_size=int(self.workspace_gb * (1 << 30)),
            )
        except ImportError:
            _log.warning("torch_tensorrt_not_available_falling_back_to_traced")
            import torch

            return torch.jit.trace(model, sample_input)
