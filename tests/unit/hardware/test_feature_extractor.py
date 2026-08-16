from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mousedroid.config.schema import CameraConfig
from mousedroid.hardware.camera.feature_extractor import (
    MeanPoolExtractor,
    TensorRTExtractor,
    build_feature_extractor,
)


def _make_frame(h: int = 480, w: int = 640, c: int = 3) -> np.ndarray:
    return np.random.default_rng(42).integers(0, 255, (h, w, c), dtype=np.uint8)


def _make_cfg(**overrides: Any) -> CameraConfig:
    defaults: dict[str, Any] = {
        "feature_dim": 256,
        "feature_extractor": "mean_pool",
        "l2_normalize": True,
    }
    defaults.update(overrides)
    return CameraConfig(**defaults)


# ---------------------------------------------------------------------------
# MeanPoolExtractor
# ---------------------------------------------------------------------------


class TestMeanPoolExtractor:
    def test_output_shape(self) -> None:
        ext = MeanPoolExtractor(256)
        result = ext.extract(_make_frame())
        assert result.shape == (256,)
        assert result.dtype == np.float32

    def test_l2_normalized(self) -> None:
        ext = MeanPoolExtractor(256, l2_normalize=True)
        result = ext.extract(_make_frame())
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)

    def test_no_l2_norm(self) -> None:
        ext = MeanPoolExtractor(256, l2_normalize=False)
        result = ext.extract(_make_frame())
        assert np.linalg.norm(result) != pytest.approx(1.0, abs=1e-3)

    def test_small_frame_zero_padded(self) -> None:
        small = np.array([[[128, 64, 32]]], dtype=np.uint8)  # 1x1x3 = 3 pixels
        ext = MeanPoolExtractor(256, l2_normalize=False)
        result = ext.extract(small)
        assert result.shape == (256,)
        assert result[3:].sum() == 0.0  # rest is zero-padded

    def test_feature_dim_property(self) -> None:
        ext = MeanPoolExtractor(128)
        assert ext.feature_dim == 128

    def test_custom_dim(self) -> None:
        ext = MeanPoolExtractor(64)
        result = ext.extract(_make_frame())
        assert result.shape == (64,)


# ---------------------------------------------------------------------------
# TensorRTExtractor
# ---------------------------------------------------------------------------


class TestTensorRTExtractor:
    def test_fallback_when_ort_unavailable(self) -> None:
        with patch("mousedroid.hardware.camera.feature_extractor._ort", None):
            ext = TensorRTExtractor(Path("dummy.onnx"), 256)
            result = ext.extract(_make_frame())
            assert result.shape == (256,)
            assert result.dtype == np.float32

    def test_fallback_when_model_load_fails(self) -> None:
        mock_ort = MagicMock()
        mock_ort.InferenceSession.side_effect = RuntimeError("bad model")
        with patch("mousedroid.hardware.camera.feature_extractor._ort", mock_ort):
            ext = TensorRTExtractor(Path("bad.onnx"), 256)
            result = ext.extract(_make_frame())
            assert result.shape == (256,)

    def test_feature_dim_property(self) -> None:
        with patch("mousedroid.hardware.camera.feature_extractor._ort", None):
            ext = TensorRTExtractor(Path("dummy.onnx"), 128)
            assert ext.feature_dim == 128


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestBuildFeatureExtractor:
    def test_default_returns_mean_pool(self) -> None:
        cfg = _make_cfg()
        ext = build_feature_extractor(cfg)
        assert isinstance(ext, MeanPoolExtractor)

    def test_auto_no_model_path_returns_mean_pool(self) -> None:
        cfg = _make_cfg(feature_extractor="auto", model_path=None)
        ext = build_feature_extractor(cfg)
        assert isinstance(ext, MeanPoolExtractor)

    def test_tensorrt_no_model_path_returns_mean_pool(self) -> None:
        cfg = _make_cfg(feature_extractor="tensorrt", model_path=None)
        ext = build_feature_extractor(cfg)
        assert isinstance(ext, MeanPoolExtractor)

    def test_tensorrt_with_model_path_returns_tensorrt(self) -> None:
        cfg = _make_cfg(feature_extractor="tensorrt", model_path=Path("/tmp/m.onnx"))
        with patch("mousedroid.hardware.camera.feature_extractor._ort", None):
            ext = build_feature_extractor(cfg)
            assert isinstance(ext, TensorRTExtractor)

    def test_auto_with_model_path_returns_tensorrt(self) -> None:
        cfg = _make_cfg(feature_extractor="auto", model_path=Path("/tmp/m.onnx"))
        with patch("mousedroid.hardware.camera.feature_extractor._ort", None):
            ext = build_feature_extractor(cfg)
            assert isinstance(ext, TensorRTExtractor)

    def test_l2_normalize_passed_through(self) -> None:
        cfg = _make_cfg(l2_normalize=False)
        ext = build_feature_extractor(cfg)
        assert isinstance(ext, MeanPoolExtractor)
        result = ext.extract(_make_frame())
        assert np.linalg.norm(result) != pytest.approx(1.0, abs=1e-3)
