"""Unit tests for the TensorRT-optimized inference wrapper.

Uses fake compilers implementing ``TensorRTCompilerProtocol`` so no real
TensorRT is required. Verifies the lazy-compile-on-first-call contract, the
PyTorch fallback when TensorRT is unavailable or compilation fails, and that
compilation is attempted at most once.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mousedroid.efficiency.optimized_inference import (
    OptimizedInference,
    OptimizedInferenceProtocol,
)


class _UnavailableCompiler:
    """Compiler that reports TensorRT is not present on this host."""

    def is_available(self) -> bool:
        return False

    async def compile_model(self, model: nn.Module, shapes: Any, precision: str) -> Any:
        raise AssertionError("compile_model must not be called when unavailable")

    async def load_compiled(self, path: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError


class _RecordingCompiler:
    """Compiler that returns a sentinel compiled model and counts attempts."""

    def __init__(self, *, fail: bool = False) -> None:
        self.attempts = 0
        self._fail = fail
        self.compiled = nn.Linear(3, 2)

    def is_available(self) -> bool:
        return True

    async def compile_model(self, model: nn.Module, shapes: Any, precision: str) -> Any:
        self.attempts += 1
        if self._fail:
            raise RuntimeError("boom")
        return self.compiled

    async def load_compiled(self, path: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError


def _wrapper(compiler: Any) -> OptimizedInference:
    torch.manual_seed(0)
    return OptimizedInference(
        model=nn.Linear(3, 2),
        compiler=compiler,
        input_shapes={"x": [1, 3]},
        precision="fp16",
    )


def test_satisfies_protocol() -> None:
    """The wrapper structurally conforms to ``OptimizedInferenceProtocol``."""
    assert isinstance(_wrapper(_UnavailableCompiler()), OptimizedInferenceProtocol)


async def test_falls_back_to_pytorch_when_unavailable() -> None:
    """With TensorRT absent, inference uses the original model and stays uncompiled."""
    wrapper = _wrapper(_UnavailableCompiler())
    out = await wrapper.infer(torch.randn(1, 3))
    assert out.shape == (1, 2)
    assert wrapper.is_compiled is False


async def test_compiles_on_first_call_then_reuses() -> None:
    """Compilation runs once on the first infer; later calls reuse the engine."""
    compiler = _RecordingCompiler()
    wrapper = _wrapper(compiler)
    assert wrapper.is_compiled is False

    await wrapper.infer(torch.randn(1, 3))
    assert wrapper.is_compiled is True
    assert compiler.attempts == 1

    await wrapper.infer(torch.randn(1, 3))
    assert compiler.attempts == 1  # not re-compiled


async def test_compilation_failure_falls_back() -> None:
    """A compiler that raises leaves the wrapper on the PyTorch path, no crash."""
    compiler = _RecordingCompiler(fail=True)
    wrapper = _wrapper(compiler)
    out = await wrapper.infer(torch.randn(2, 3))
    assert out.shape == (2, 2)
    assert wrapper.is_compiled is False
    assert compiler.attempts == 1
