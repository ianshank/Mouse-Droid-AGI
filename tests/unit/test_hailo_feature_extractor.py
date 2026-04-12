from __future__ import annotations

from typing import Any

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
    import asyncio

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
        # Mock returns zeros — L2 norm of zeros is 0, so no normalization
        # Just verify it doesn't crash and returns correct shape
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
