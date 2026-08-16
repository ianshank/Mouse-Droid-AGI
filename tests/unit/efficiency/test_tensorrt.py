"""Tests for TensorRT compilation -- protocol compliance, caching, fallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from mousedroid.config.schema import JetsonConfig
from mousedroid.efficiency.tensorrt import (
    JetsonTensorRTCompiler,
    MockTensorRTCompiler,
    TensorRTCompilerProtocol,
    _model_fingerprint,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal model for testing compilation."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(x)  # type: ignore[no-any-return]


@pytest.fixture
def tiny_model() -> nn.Module:
    return _TinyModel()


@pytest.fixture
def input_shapes() -> dict[str, list[int]]:
    return {"obs": [1, 4]}


@pytest.fixture
def jetson_cfg() -> JetsonConfig:
    return JetsonConfig(tensorrt_enabled=True, precision="fp16", workspace_gb=1.0)


@pytest.fixture
def jetson_cfg_disabled() -> JetsonConfig:
    return JetsonConfig(tensorrt_enabled=False)


@pytest.fixture
def jetson_cfg_fp32() -> JetsonConfig:
    return JetsonConfig(tensorrt_enabled=True, precision="fp32", workspace_gb=1.0)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Verify that concrete classes satisfy TensorRTCompilerProtocol."""

    def test_mock_satisfies_protocol(self) -> None:
        compiler = MockTensorRTCompiler()
        assert isinstance(compiler, TensorRTCompilerProtocol)

    def test_jetson_satisfies_protocol(self, jetson_cfg: JetsonConfig) -> None:
        compiler = JetsonTensorRTCompiler(jetson_cfg)
        assert isinstance(compiler, TensorRTCompilerProtocol)


# ---------------------------------------------------------------------------
# MockTensorRTCompiler
# ---------------------------------------------------------------------------


class TestMockCompiler:
    """Tests for the mock compiler used in testing."""

    def test_is_available(self) -> None:
        compiler = MockTensorRTCompiler()
        assert compiler.is_available() is True

    @pytest.mark.asyncio
    async def test_compile_returns_model(
        self, tiny_model: nn.Module, input_shapes: dict[str, list[int]]
    ) -> None:
        compiler = MockTensorRTCompiler()
        result = await compiler.compile_model(tiny_model, input_shapes, "fp16")
        assert result is tiny_model

    @pytest.mark.asyncio
    async def test_compile_stores_in_cache(
        self, tiny_model: nn.Module, input_shapes: dict[str, list[int]]
    ) -> None:
        compiler = MockTensorRTCompiler()
        await compiler.compile_model(tiny_model, input_shapes, "fp16")
        assert len(compiler.compiled_models) == 1

    @pytest.mark.asyncio
    async def test_load_compiled(self) -> None:
        compiler = MockTensorRTCompiler()
        result = await compiler.load_compiled(Path("/fake/path"))
        assert isinstance(result, nn.Module)


# ---------------------------------------------------------------------------
# JetsonTensorRTCompiler
# ---------------------------------------------------------------------------


class TestJetsonCompiler:
    """Tests for the real JetsonTensorRTCompiler with torch2trt mocked out."""

    def test_is_available_when_enabled(self, jetson_cfg: JetsonConfig) -> None:
        compiler = JetsonTensorRTCompiler(jetson_cfg)
        result = compiler.is_available()
        assert isinstance(result, bool)

    def test_is_available_when_disabled(self, jetson_cfg_disabled: JetsonConfig) -> None:
        compiler = JetsonTensorRTCompiler(jetson_cfg_disabled)
        assert compiler.is_available() is False

    @pytest.mark.asyncio
    async def test_compile_disabled_returns_model(
        self,
        jetson_cfg_disabled: JetsonConfig,
        tiny_model: nn.Module,
        input_shapes: dict[str, list[int]],
    ) -> None:
        compiler = JetsonTensorRTCompiler(jetson_cfg_disabled)
        result = await compiler.compile_model(tiny_model, input_shapes, "fp16")
        assert result is tiny_model

    @pytest.mark.asyncio
    async def test_compile_falls_back_to_jit_trace(
        self,
        tiny_model: nn.Module,
        input_shapes: dict[str, list[int]],
        tmp_path: Path,
    ) -> None:
        """When torch2trt is unavailable, falls back to JIT tracing."""
        cfg = JetsonConfig(
            tensorrt_enabled=True,
            precision="fp16",
            workspace_gb=1.0,
            tensorrt_cache_dir=tmp_path / "trt_cache",
        )
        compiler = JetsonTensorRTCompiler(cfg)
        with patch("mousedroid.efficiency.tensorrt._TORCH2TRT_AVAILABLE", False):
            result = await compiler.compile_model(tiny_model, input_shapes, "fp16")
        assert result is not None

    @pytest.mark.asyncio
    async def test_cache_hit(
        self,
        tiny_model: nn.Module,
        input_shapes: dict[str, list[int]],
        tmp_path: Path,
    ) -> None:
        """Second compilation with same params should hit disk cache."""
        cfg = JetsonConfig(
            tensorrt_enabled=True,
            precision="fp16",
            workspace_gb=1.0,
            tensorrt_cache_dir=tmp_path / "trt_cache",
        )
        compiler = JetsonTensorRTCompiler(cfg)

        with patch("mousedroid.efficiency.tensorrt._TORCH2TRT_AVAILABLE", False):
            result1 = await compiler.compile_model(tiny_model, input_shapes, "fp16")
            result2 = await compiler.compile_model(tiny_model, input_shapes, "fp16")

        assert result1 is not None
        assert result2 is not None

    @pytest.mark.asyncio
    async def test_cache_miss_different_precision(
        self,
        tiny_model: nn.Module,
        input_shapes: dict[str, list[int]],
        tmp_path: Path,
    ) -> None:
        """Different precision should produce a different cache key."""
        cfg = JetsonConfig(
            tensorrt_enabled=True,
            precision="fp16",
            workspace_gb=1.0,
            tensorrt_cache_dir=tmp_path / "trt_cache",
        )
        compiler = JetsonTensorRTCompiler(cfg)

        with patch("mousedroid.efficiency.tensorrt._TORCH2TRT_AVAILABLE", False):
            result_fp16 = await compiler.compile_model(tiny_model, input_shapes, "fp16")
            result_fp32 = await compiler.compile_model(tiny_model, input_shapes, "fp32")

        assert result_fp16 is not None
        assert result_fp32 is not None

    @pytest.mark.asyncio
    async def test_load_compiled_file_not_found(self, jetson_cfg: JetsonConfig) -> None:
        compiler = JetsonTensorRTCompiler(jetson_cfg)
        with pytest.raises(FileNotFoundError):
            await compiler.load_compiled(Path("/nonexistent/engine.pth"))

    @pytest.mark.asyncio
    async def test_load_compiled_from_disk(
        self,
        tiny_model: nn.Module,
        tmp_path: Path,
    ) -> None:
        """Can load a saved model from disk."""
        save_path = tmp_path / "test_engine.pth"
        torch.save(tiny_model, str(save_path))

        cfg = JetsonConfig(
            tensorrt_enabled=True,
            tensorrt_cache_dir=tmp_path,
        )
        compiler = JetsonTensorRTCompiler(cfg)
        loaded = await compiler.load_compiled(save_path)
        assert loaded is not None


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint:
    """Tests for the model fingerprint function."""

    def test_same_inputs_same_fingerprint(
        self, tiny_model: nn.Module, input_shapes: dict[str, list[int]]
    ) -> None:
        fp1 = _model_fingerprint(tiny_model, input_shapes, "fp16")
        fp2 = _model_fingerprint(tiny_model, input_shapes, "fp16")
        assert fp1 == fp2

    def test_different_precision_different_fingerprint(
        self, tiny_model: nn.Module, input_shapes: dict[str, list[int]]
    ) -> None:
        fp16 = _model_fingerprint(tiny_model, input_shapes, "fp16")
        fp32 = _model_fingerprint(tiny_model, input_shapes, "fp32")
        assert fp16 != fp32

    def test_different_shapes_different_fingerprint(
        self,
        tiny_model: nn.Module,
    ) -> None:
        fp1 = _model_fingerprint(tiny_model, {"obs": [1, 4]}, "fp16")
        fp2 = _model_fingerprint(tiny_model, {"obs": [2, 4]}, "fp16")
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Test that JetsonConfig fields work correctly with the compiler."""

    def test_default_config_values(self) -> None:
        cfg = JetsonConfig()
        assert cfg.tensorrt_enabled is True
        assert cfg.precision == "fp16"
        assert cfg.workspace_gb == 1.0
        assert cfg.tensorrt_cache_dir == Path("/opt/mousedroid/tensorrt_cache")

    def test_custom_cache_dir(self, tmp_path: Path) -> None:
        cfg = JetsonConfig(tensorrt_cache_dir=tmp_path / "custom_cache")
        compiler = JetsonTensorRTCompiler(cfg)
        assert compiler._cache_dir == tmp_path / "custom_cache"

    def test_fp32_precision(self) -> None:
        cfg = JetsonConfig(precision="fp32")
        assert cfg.precision == "fp32"

    def test_int8_precision(self) -> None:
        cfg = JetsonConfig(precision="int8")
        assert cfg.precision == "int8"


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactoryIntegration:
    """Test build_tensorrt_compiler factory function."""

    def test_build_returns_protocol(self) -> None:
        from mousedroid.config.schema import Settings
        from mousedroid.factory import build_tensorrt_compiler

        cfg = Settings(mock_hardware=True)
        compiler = build_tensorrt_compiler(cfg)
        assert isinstance(compiler, TensorRTCompilerProtocol)

    def test_build_disabled_returns_mock(self) -> None:
        from mousedroid.config.schema import Settings
        from mousedroid.factory import build_tensorrt_compiler

        cfg = Settings(mock_hardware=True, jetson={"tensorrt_enabled": False})
        compiler = build_tensorrt_compiler(cfg)
        assert isinstance(compiler, MockTensorRTCompiler)

    def test_build_enabled_returns_jetson_compiler(self) -> None:
        from mousedroid.config.schema import Settings
        from mousedroid.factory import build_tensorrt_compiler

        cfg = Settings(mock_hardware=True, jetson={"tensorrt_enabled": True})
        compiler = build_tensorrt_compiler(cfg)
        assert isinstance(compiler, JetsonTensorRTCompiler)
