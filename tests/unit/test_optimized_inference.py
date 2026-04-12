"""Tests for optimized inference wrapper -- compilation, fallback, latency."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from mousedroid.efficiency.optimized_inference import (
    OptimizedInference,
    OptimizedInferenceProtocol,
)
from mousedroid.efficiency.tensorrt import MockTensorRTCompiler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _SimpleModel(nn.Module):
    """Minimal model for inference testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.linear(x)  # type: ignore[no-any-return]


@pytest.fixture
def simple_model() -> nn.Module:
    return _SimpleModel()


@pytest.fixture
def mock_compiler() -> MockTensorRTCompiler:
    return MockTensorRTCompiler()


@pytest.fixture
def input_shapes() -> dict[str, list[int]]:
    return {"obs": [1, 4]}


@pytest.fixture
def obs_tensor() -> Tensor:
    return torch.randn(1, 4)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Verify OptimizedInference satisfies the protocol."""

    def test_satisfies_protocol(
        self,
        simple_model: nn.Module,
        mock_compiler: MockTensorRTCompiler,
        input_shapes: dict[str, list[int]],
    ) -> None:
        wrapper = OptimizedInference(simple_model, mock_compiler, input_shapes)
        assert isinstance(wrapper, OptimizedInferenceProtocol)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class TestInference:
    """Tests for inference execution."""

    @pytest.mark.asyncio
    async def test_infer_returns_tensor(
        self,
        simple_model: nn.Module,
        mock_compiler: MockTensorRTCompiler,
        input_shapes: dict[str, list[int]],
        obs_tensor: Tensor,
    ) -> None:
        wrapper = OptimizedInference(simple_model, mock_compiler, input_shapes)
        result = await wrapper.infer(obs_tensor)
        assert isinstance(result, Tensor)
        assert result.shape == (1, 2)

    @pytest.mark.asyncio
    async def test_infer_multiple_calls(
        self,
        simple_model: nn.Module,
        mock_compiler: MockTensorRTCompiler,
        input_shapes: dict[str, list[int]],
        obs_tensor: Tensor,
    ) -> None:
        """Multiple infer calls should not re-trigger compilation."""
        wrapper = OptimizedInference(simple_model, mock_compiler, input_shapes)
        r1 = await wrapper.infer(obs_tensor)
        r2 = await wrapper.infer(obs_tensor)
        assert isinstance(r1, Tensor)
        assert isinstance(r2, Tensor)
        # Compilation should have been called only once.
        assert len(mock_compiler.compiled_models) == 1


# ---------------------------------------------------------------------------
# Compilation trigger
# ---------------------------------------------------------------------------


class TestCompilationTrigger:
    """Tests for lazy compilation on first call."""

    @pytest.mark.asyncio
    async def test_compilation_on_first_call(
        self,
        simple_model: nn.Module,
        mock_compiler: MockTensorRTCompiler,
        input_shapes: dict[str, list[int]],
        obs_tensor: Tensor,
    ) -> None:
        wrapper = OptimizedInference(simple_model, mock_compiler, input_shapes)
        assert not wrapper.is_compiled
        await wrapper.infer(obs_tensor)
        assert wrapper.is_compiled

    @pytest.mark.asyncio
    async def test_not_compiled_before_first_call(
        self,
        simple_model: nn.Module,
        mock_compiler: MockTensorRTCompiler,
        input_shapes: dict[str, list[int]],
    ) -> None:
        wrapper = OptimizedInference(simple_model, mock_compiler, input_shapes)
        assert not wrapper.is_compiled


# ---------------------------------------------------------------------------
# Fallback to PyTorch
# ---------------------------------------------------------------------------


class TestFallback:
    """Tests for graceful fallback when TensorRT is unavailable."""

    @pytest.mark.asyncio
    async def test_fallback_when_not_available(
        self,
        simple_model: nn.Module,
        input_shapes: dict[str, list[int]],
        obs_tensor: Tensor,
    ) -> None:
        """When compiler reports not available, use original model."""
        compiler = MockTensorRTCompiler()
        compiler.is_available = lambda: False  # type: ignore[assignment]

        wrapper = OptimizedInference(simple_model, compiler, input_shapes)
        result = await wrapper.infer(obs_tensor)
        assert isinstance(result, Tensor)
        assert not wrapper.is_compiled

    @pytest.mark.asyncio
    async def test_fallback_on_compilation_error(
        self,
        simple_model: nn.Module,
        input_shapes: dict[str, list[int]],
        obs_tensor: Tensor,
    ) -> None:
        """If compilation raises, fall back to original model."""
        compiler = MockTensorRTCompiler()

        async def _fail(*args: object, **kwargs: object) -> None:
            msg = "compilation failed"
            raise RuntimeError(msg)

        compiler.compile_model = _fail  # type: ignore[assignment]

        wrapper = OptimizedInference(simple_model, compiler, input_shapes)
        result = await wrapper.infer(obs_tensor)
        assert isinstance(result, Tensor)
        assert not wrapper.is_compiled


# ---------------------------------------------------------------------------
# torch.no_grad verification
# ---------------------------------------------------------------------------


class TestNoGrad:
    """Verify inference runs under torch.no_grad."""

    @pytest.mark.asyncio
    async def test_no_grad_during_inference(
        self,
        simple_model: nn.Module,
        mock_compiler: MockTensorRTCompiler,
        input_shapes: dict[str, list[int]],
    ) -> None:
        """Ensure no gradients are computed during infer()."""
        wrapper = OptimizedInference(simple_model, mock_compiler, input_shapes)
        obs = torch.randn(1, 4, requires_grad=True)
        result = await wrapper.infer(obs)
        # Under no_grad, result should not require grad even though input does.
        assert not result.requires_grad


# ---------------------------------------------------------------------------
# Precision configuration
# ---------------------------------------------------------------------------


class TestPrecisionConfig:
    """Test precision parameter passes through correctly."""

    @pytest.mark.asyncio
    async def test_fp16_precision(
        self,
        simple_model: nn.Module,
        input_shapes: dict[str, list[int]],
        obs_tensor: Tensor,
    ) -> None:
        compiler = MockTensorRTCompiler()
        wrapper = OptimizedInference(simple_model, compiler, input_shapes, precision="fp16")
        await wrapper.infer(obs_tensor)
        assert wrapper._precision == "fp16"

    @pytest.mark.asyncio
    async def test_fp32_precision(
        self,
        simple_model: nn.Module,
        input_shapes: dict[str, list[int]],
        obs_tensor: Tensor,
    ) -> None:
        compiler = MockTensorRTCompiler()
        wrapper = OptimizedInference(simple_model, compiler, input_shapes, precision="fp32")
        await wrapper.infer(obs_tensor)
        assert wrapper._precision == "fp32"
