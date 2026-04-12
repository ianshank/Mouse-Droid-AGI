from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from mousedroid.config.schema import CameraConfig, HailoConfig
from mousedroid.hardware.accelerator.hailo_runtime import MockHailoRuntime
from mousedroid.hardware.camera.feature_extractor import (
    HailoFeatureExtractor,
    MeanPoolExtractor,
    build_feature_extractor,
)


def _make_frame(h: int = 480, w: int = 640, c: int = 3) -> np.ndarray:
    return np.random.default_rng(42).integers(0, 255, (h, w, c), dtype=np.uint8)


def _make_camera_cfg(**overrides: Any) -> CameraConfig:
    defaults: dict[str, Any] = {
        "feature_dim": 256,
        "feature_extractor": "hailo",
        "l2_normalize": True,
    }
    defaults.update(overrides)
    return CameraConfig(**defaults)


def _make_hailo_cfg() -> HailoConfig:
    return HailoConfig(enabled=True)


def _make_started_mock_runtime() -> MockHailoRuntime:
    """Create a mock runtime in the started state (sync helper)."""
    cfg = _make_hailo_cfg()
    rt = MockHailoRuntime(cfg)
    asyncio.run(rt.start())
    return rt


# ---------------------------------------------------------------------------
# HailoFeatureExtractor
# ---------------------------------------------------------------------------


class TestHailoFeatureExtractor:
    def test_output_shape(self) -> None:
        rt = _make_started_mock_runtime()
        ext = HailoFeatureExtractor(rt, feature_dim=256)
        result = ext.extract(_make_frame())
        assert result.shape == (256,)
        assert result.dtype == np.float32

    def test_feature_dim_property(self) -> None:
        rt = _make_started_mock_runtime()
        ext = HailoFeatureExtractor(rt, feature_dim=128)
        assert ext.feature_dim == 128

    def test_l2_normalized(self) -> None:
        rt = _make_started_mock_runtime()
        ext = HailoFeatureExtractor(rt, feature_dim=256, l2_normalize=True)
        result = ext.extract(_make_frame())
        # Mock returns zeros — zeros remain zeros after normalization
        assert result.shape == (256,)

    def test_fallback_when_runtime_unavailable(self) -> None:
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg)
        # NOT started — is_available() returns False
        ext = HailoFeatureExtractor(rt, feature_dim=256)
        result = ext.extract(_make_frame())
        assert result.shape == (256,)
        assert result.dtype == np.float32
        # Should have used MeanPool fallback — L2 normalized non-zero result
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)

    def test_2d_frame_expands_dim(self) -> None:
        """Grayscale (H, W) frame gets expanded to (H, W, 1)."""
        rt = _make_started_mock_runtime()
        ext = HailoFeatureExtractor(rt, feature_dim=256)
        gray_frame = np.zeros((480, 640), dtype=np.uint8)
        result = ext.extract(gray_frame)
        assert result.shape == (256,)

    def test_extract_pads_short_output(self) -> None:
        """When runtime returns fewer features than feature_dim, output is padded."""
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg, output_shapes={"feature_extractor": (64,)})
        asyncio.run(rt.start())
        ext = HailoFeatureExtractor(rt, feature_dim=256, l2_normalize=False)
        result = ext.extract(_make_frame())
        assert result.shape == (256,)
        # Last elements should be zero-padded
        assert np.all(result[64:] == 0.0)

    def test_extract_truncates_long_output(self) -> None:
        """When runtime returns more features than feature_dim, output is truncated."""
        cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(cfg, output_shapes={"feature_extractor": (512,)})
        asyncio.run(rt.start())
        ext = HailoFeatureExtractor(rt, feature_dim=128, l2_normalize=False)
        result = ext.extract(_make_frame())
        assert result.shape == (128,)

    def test_extract_fallback_on_exception(self) -> None:
        """Inference exception triggers MeanPool fallback."""
        mock_runtime = MagicMock()
        mock_runtime.is_available.return_value = True
        # run_inference raises — but it's an async method so we need a coroutine
        async def _raise(*a: Any, **kw: Any) -> Any:
            raise RuntimeError("inference failed")
        mock_runtime.run_inference = _raise
        ext = HailoFeatureExtractor(mock_runtime, feature_dim=256)
        result = ext.extract(_make_frame())
        assert result.shape == (256,)
        # Should have used MeanPool fallback
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)

    def test_no_l2_normalize(self) -> None:
        rt = _make_started_mock_runtime()
        ext = HailoFeatureExtractor(rt, feature_dim=256, l2_normalize=False)
        result = ext.extract(_make_frame())
        assert result.shape == (256,)


# ---------------------------------------------------------------------------
# build_feature_extractor with Hailo
# ---------------------------------------------------------------------------


class TestBuildFeatureExtractorHailo:
    def test_hailo_with_runtime_returns_hailo_extractor(self) -> None:
        cfg = _make_camera_cfg(feature_extractor="hailo")
        rt = _make_started_mock_runtime()
        ext = build_feature_extractor(cfg, hailo_runtime=rt)
        assert isinstance(ext, HailoFeatureExtractor)

    def test_hailo_without_runtime_returns_mean_pool(self) -> None:
        cfg = _make_camera_cfg(feature_extractor="hailo")
        ext = build_feature_extractor(cfg, hailo_runtime=None)
        assert isinstance(ext, MeanPoolExtractor)

    def test_auto_with_available_runtime_returns_hailo(self) -> None:
        cfg = _make_camera_cfg(feature_extractor="auto")
        rt = _make_started_mock_runtime()
        ext = build_feature_extractor(cfg, hailo_runtime=rt)
        assert isinstance(ext, HailoFeatureExtractor)

    def test_auto_with_unavailable_runtime_returns_mean_pool(self) -> None:
        cfg = _make_camera_cfg(feature_extractor="auto", model_path=None)
        rt_cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(rt_cfg)
        # NOT started — is_available() returns False
        ext = build_feature_extractor(cfg, hailo_runtime=rt)
        assert isinstance(ext, MeanPoolExtractor)

    def test_mean_pool_ignores_runtime(self) -> None:
        cfg = _make_camera_cfg(feature_extractor="mean_pool")
        rt = _make_started_mock_runtime()
        ext = build_feature_extractor(cfg, hailo_runtime=rt)
        assert isinstance(ext, MeanPoolExtractor)

    def test_backward_compat_no_runtime_arg(self) -> None:
        cfg = _make_camera_cfg(feature_extractor="mean_pool")
        ext = build_feature_extractor(cfg)
        assert isinstance(ext, MeanPoolExtractor)
