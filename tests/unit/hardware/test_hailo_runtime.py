from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mousedroid.config.schema import HailoConfig
from mousedroid.hardware.accelerator.hailo_runtime import (
    HailoRuntime,
    HailoRuntimeProtocol,
    MockHailoRuntime,
)


def _make_hailo_cfg(**overrides: Any) -> HailoConfig:
    defaults: dict[str, Any] = {
        "enabled": True,
        "device_path": "/dev/hailo0",
        "yolo_hef_path": "models/hailo/yolo.hef",
        "feature_extractor_hef_path": "models/hailo/feat.hef",
    }
    defaults.update(overrides)
    return HailoConfig(**defaults)


# ---------------------------------------------------------------------------
# MockHailoRuntime
# ---------------------------------------------------------------------------


class TestMockHailoRuntime:
    @pytest.mark.asyncio
    async def test_start_sets_available(self) -> None:
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        assert not rt.is_available()
        await rt.start()
        assert rt.is_available()

    @pytest.mark.asyncio
    async def test_stop_clears_available(self) -> None:
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        await rt.start()
        await rt.stop()
        assert not rt.is_available()

    @pytest.mark.asyncio
    async def test_yolo_output_shape(self) -> None:
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        await rt.start()
        dummy_input = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rt.infer_sync("yolo", dummy_input)
        assert result.shape == (25200, 85)
        assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_feature_extractor_output_shape(self) -> None:
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        await rt.start()
        dummy_input = np.zeros((480, 640, 3), dtype=np.uint8)
        result = rt.infer_sync("feature_extractor", dummy_input)
        assert result.shape == (256,)
        assert result.dtype == np.float32

    def test_unknown_model_raises(self) -> None:
        import asyncio

        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        asyncio.run(rt.start())
        with pytest.raises(RuntimeError, match="Unknown model"):
            rt.infer_sync("nonexistent", np.zeros(1, dtype=np.uint8))

    def test_not_started_raises(self) -> None:
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        with pytest.raises(RuntimeError, match="not available"):
            rt.infer_sync("yolo", np.zeros(1, dtype=np.uint8))

    def test_implements_protocol(self) -> None:
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        assert isinstance(rt, HailoRuntimeProtocol)

    @pytest.mark.asyncio
    async def test_custom_output_shapes(self) -> None:
        cfg = _make_hailo_cfg()
        custom = {"yolo": (100, 7), "feature_extractor": (512,)}
        rt = MockHailoRuntime(cfg, output_shapes=custom)
        await rt.start()
        result = rt.infer_sync("yolo", np.zeros(1, dtype=np.uint8))
        assert result.shape == (100, 7)
        result = rt.infer_sync("feature_extractor", np.zeros(1, dtype=np.uint8))
        assert result.shape == (512,)

    def test_default_output_shapes_class_attr(self) -> None:
        assert "yolo" in MockHailoRuntime.DEFAULT_OUTPUT_SHAPES
        assert "feature_extractor" in MockHailoRuntime.DEFAULT_OUTPUT_SHAPES


# ---------------------------------------------------------------------------
# HailoRuntime — init & start with mocked hailo_platform
# ---------------------------------------------------------------------------


def _build_mock_hailort() -> MagicMock:
    """Build a mock hailo_platform module with all needed stubs."""
    mock_hp = MagicMock()
    # VDevice
    mock_device = MagicMock()
    mock_hp.VDevice.create_params.return_value = MagicMock()
    mock_hp.VDevice.return_value = mock_device
    # HEF
    mock_hef = MagicMock()
    mock_hp.HEF.return_value = mock_hef
    # Network group from configure
    mock_ng = MagicMock()
    input_info = MagicMock()
    input_info.name = "input_0"
    output_info = MagicMock()
    output_info.name = "output_0"
    mock_ng.get_input_vstream_infos.return_value = [input_info]
    mock_ng.get_output_vstream_infos.return_value = [output_info]
    mock_device.configure.return_value = [mock_ng]
    # VStream params
    mock_hp.InputVStreamParams.make.return_value = MagicMock()
    mock_hp.OutputVStreamParams.make.return_value = MagicMock()
    mock_hp.FormatType.UINT8 = "uint8"
    mock_hp.FormatType.FLOAT32 = "float32"
    # InferVStreams context manager
    mock_pipeline = MagicMock()
    mock_pipeline.infer.return_value = {"output_0": np.zeros((1, 256), dtype=np.float32)}
    mock_hp.InferVStreams.return_value.__enter__ = MagicMock(return_value=mock_pipeline)
    mock_hp.InferVStreams.return_value.__exit__ = MagicMock(return_value=False)
    return mock_hp


class TestHailoRuntimeInit:
    def test_init_stores_config(self) -> None:
        cfg = _make_hailo_cfg()
        rt = HailoRuntime(cfg)
        assert rt._cfg is cfg
        assert rt._available is False
        assert rt._models == {}

    def test_uses_threading_lock(self) -> None:
        cfg = _make_hailo_cfg()
        rt = HailoRuntime(cfg)
        assert type(rt._lock).__name__ == "_RLock" or hasattr(rt._lock, "acquire")
        # Verify it's NOT an asyncio.Lock
        import asyncio

        assert not isinstance(rt._lock, asyncio.Lock)


class TestHailoRuntimeStart:
    @pytest.mark.asyncio
    async def test_start_without_hailort_stays_unavailable(self) -> None:
        cfg = _make_hailo_cfg()
        rt = HailoRuntime(cfg)
        with patch("mousedroid.hardware.accelerator.hailo_runtime._hailort", None):
            await rt.start()
        assert not rt.is_available()

    @pytest.mark.asyncio
    async def test_start_device_discovery_failure(self) -> None:
        cfg = _make_hailo_cfg()
        rt = HailoRuntime(cfg)
        mock_hp = _build_mock_hailort()
        mock_hp.VDevice.side_effect = OSError("device not found")
        with patch("mousedroid.hardware.accelerator.hailo_runtime._hailort", mock_hp):
            await rt.start()
        assert not rt.is_available()

    @pytest.mark.asyncio
    async def test_start_success_with_existing_hefs(self, tmp_path: Path) -> None:
        yolo_hef = tmp_path / "yolo.hef"
        feat_hef = tmp_path / "feat.hef"
        yolo_hef.touch()
        feat_hef.touch()
        cfg = _make_hailo_cfg(
            yolo_hef_path=str(yolo_hef),
            feature_extractor_hef_path=str(feat_hef),
        )
        rt = HailoRuntime(cfg)
        mock_hp = _build_mock_hailort()
        with patch("mousedroid.hardware.accelerator.hailo_runtime._hailort", mock_hp):
            await rt.start()
        assert rt.is_available()
        assert "yolo" in rt._models
        assert "feature_extractor" in rt._models

    @pytest.mark.asyncio
    async def test_start_missing_hef_skips_model(self) -> None:
        cfg = _make_hailo_cfg(
            yolo_hef_path="nonexistent/yolo.hef",
            feature_extractor_hef_path="nonexistent/feat.hef",
        )
        rt = HailoRuntime(cfg)
        mock_hp = _build_mock_hailort()
        with patch("mousedroid.hardware.accelerator.hailo_runtime._hailort", mock_hp):
            await rt.start()
        assert not rt.is_available()

    @pytest.mark.asyncio
    async def test_start_hef_load_exception_skips_model(self, tmp_path: Path) -> None:
        yolo_hef = tmp_path / "yolo.hef"
        yolo_hef.touch()
        cfg = _make_hailo_cfg(
            yolo_hef_path=str(yolo_hef),
            feature_extractor_hef_path="nonexistent/feat.hef",
        )
        rt = HailoRuntime(cfg)
        mock_hp = _build_mock_hailort()
        mock_hp.HEF.side_effect = RuntimeError("corrupt HEF")
        with patch("mousedroid.hardware.accelerator.hailo_runtime._hailort", mock_hp):
            await rt.start()
        assert not rt.is_available()


class TestHailoRuntimeStop:
    @pytest.mark.asyncio
    async def test_stop_clears_state(self, tmp_path: Path) -> None:
        yolo_hef = tmp_path / "yolo.hef"
        feat_hef = tmp_path / "feat.hef"
        yolo_hef.touch()
        feat_hef.touch()
        cfg = _make_hailo_cfg(
            yolo_hef_path=str(yolo_hef),
            feature_extractor_hef_path=str(feat_hef),
        )
        rt = HailoRuntime(cfg)
        mock_hp = _build_mock_hailort()
        with patch("mousedroid.hardware.accelerator.hailo_runtime._hailort", mock_hp):
            await rt.start()
            assert rt.is_available()
            await rt.stop()
        assert not rt.is_available()
        assert rt._models == {}
        assert rt._device is None

    @pytest.mark.asyncio
    async def test_stop_handles_release_exception(self, tmp_path: Path) -> None:
        yolo_hef = tmp_path / "yolo.hef"
        yolo_hef.touch()
        cfg = _make_hailo_cfg(
            yolo_hef_path=str(yolo_hef),
            feature_extractor_hef_path="nonexistent.hef",
        )
        rt = HailoRuntime(cfg)
        mock_hp = _build_mock_hailort()
        mock_device = mock_hp.VDevice.return_value
        mock_device.release.side_effect = OSError("release failed")
        with patch("mousedroid.hardware.accelerator.hailo_runtime._hailort", mock_hp):
            await rt.start()
            await rt.stop()
        assert not rt.is_available()


class TestHailoRuntimeInference:
    def test_inference_not_available_raises(self) -> None:
        cfg = _make_hailo_cfg()
        rt = HailoRuntime(cfg)
        with pytest.raises(RuntimeError, match="not available"):
            rt.infer_sync("yolo", np.zeros(1, dtype=np.uint8))

    @pytest.mark.asyncio
    async def test_inference_unknown_model_raises(self, tmp_path: Path) -> None:
        yolo_hef = tmp_path / "yolo.hef"
        yolo_hef.touch()
        cfg = _make_hailo_cfg(
            yolo_hef_path=str(yolo_hef),
            feature_extractor_hef_path="nonexistent.hef",
        )
        rt = HailoRuntime(cfg)
        mock_hp = _build_mock_hailort()
        with patch("mousedroid.hardware.accelerator.hailo_runtime._hailort", mock_hp):
            await rt.start()
            with pytest.raises(RuntimeError, match="not loaded"):
                rt.infer_sync("nonexistent", np.zeros(1, dtype=np.uint8))

    @pytest.mark.asyncio
    async def test_inference_success(self, tmp_path: Path) -> None:
        yolo_hef = tmp_path / "yolo.hef"
        feat_hef = tmp_path / "feat.hef"
        yolo_hef.touch()
        feat_hef.touch()
        cfg = _make_hailo_cfg(
            yolo_hef_path=str(yolo_hef),
            feature_extractor_hef_path=str(feat_hef),
        )
        rt = HailoRuntime(cfg)
        mock_hp = _build_mock_hailort()
        with patch("mousedroid.hardware.accelerator.hailo_runtime._hailort", mock_hp):
            await rt.start()
            result = rt.infer_sync("yolo", np.zeros((480, 640, 3), dtype=np.uint8))
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestHailoRuntimeImportGuard:
    def test_runtime_available_without_hailort(self) -> None:
        """HailoRuntime module loads even if hailo_platform is absent."""
        with patch.dict("sys.modules", {"hailo_platform": None}):
            import importlib

            import mousedroid.hardware.accelerator.hailo_runtime as mod

            importlib.reload(mod)
            assert mod._hailort is None
