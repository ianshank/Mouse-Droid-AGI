"""Optimized inference wrapper -- compiles models via TensorRT on first call.

Wraps an RSSM world model and transparently applies TensorRT compilation
when available, falling back to standard PyTorch inference otherwise.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.efficiency.tensorrt import TensorRTCompilerProtocol
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


@runtime_checkable
class OptimizedInferenceProtocol(Protocol):
    """Interface for optimized model inference."""

    async def infer(self, obs: Tensor) -> Tensor:
        """Run inference on an observation tensor.

        Args:
            obs: Observation tensor.

        Returns:
            Model output tensor.
        """
        ...


class OptimizedInference:
    """TensorRT-accelerated inference wrapper for RSSM-based world models.

    On the first call to :meth:`infer`, the underlying model is compiled via
    the provided :class:`TensorRTCompilerProtocol` (if available).  Subsequent
    calls use the compiled version.  Falls back to standard PyTorch when
    TensorRT compilation is unavailable or fails.

    All inference runs under ``torch.no_grad()`` for safety.

    Args:
        model: PyTorch model (typically RSSM or its sub-module).
        compiler: TensorRT compiler implementing the protocol.
        input_shapes: Expected input shapes for compilation.
        precision: Target precision (``"fp16"`` or ``"fp32"``).
    """

    def __init__(
        self,
        model: nn.Module,
        compiler: TensorRTCompilerProtocol,
        input_shapes: dict[str, list[int]],
        precision: str = "fp16",
    ) -> None:
        self._original_model = model
        self._compiler = compiler
        self._input_shapes = input_shapes
        self._precision = precision
        self._compiled_model: Any | None = None
        self._compilation_attempted: bool = False

    @property
    def is_compiled(self) -> bool:
        """Whether the model has been compiled via TensorRT.

        Returns:
            True if a compiled model is available.
        """
        return self._compiled_model is not None

    async def _ensure_compiled(self) -> None:
        """Compile the model on first invocation if TensorRT is available."""
        if self._compilation_attempted:
            return

        self._compilation_attempted = True

        if not self._compiler.is_available():
            _log.info("tensorrt_not_available_using_pytorch_fallback")
            return

        try:
            _log.info(
                "optimized_inference_compiling",
                precision=self._precision,
                input_shapes=self._input_shapes,
            )
            self._compiled_model = await self._compiler.compile_model(
                self._original_model,
                self._input_shapes,
                self._precision,
            )
            _log.info("optimized_inference_compiled")
        except Exception:
            _log.warning(
                "tensorrt_compilation_failed_using_pytorch_fallback",
                exc_info=True,
            )

    def _get_active_model(self) -> Any:
        """Return the compiled model if available, else the original.

        Returns:
            The model to use for inference.
        """
        if self._compiled_model is not None:
            return self._compiled_model
        return self._original_model

    async def infer(self, obs: Tensor) -> Tensor:
        """Run inference with automatic TensorRT compilation on first call.

        Compilation is triggered lazily on the first invocation.  Latency
        is logged for every call.  All forward passes run under
        ``torch.no_grad()`` to avoid unnecessary gradient tracking.

        Args:
            obs: Observation tensor, shape matching ``input_shapes``.

        Returns:
            Model output tensor.
        """
        await self._ensure_compiled()

        model = self._get_active_model()
        start = time.monotonic()

        with torch.no_grad():
            result: Tensor = model(obs)

        elapsed_ms = (time.monotonic() - start) * 1000.0
        _log.debug(
            "optimized_inference_step",
            latency_ms=round(elapsed_ms, 2),
            compiled=self.is_compiled,
        )
        return result
