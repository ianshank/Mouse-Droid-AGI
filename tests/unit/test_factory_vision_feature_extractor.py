"""build_vision_feature_extractor returns a MeanPool extractor matching camera cfg."""

from __future__ import annotations

import numpy as np

from mousedroid.config.schema import Settings
from mousedroid.factory import build_vision_feature_extractor


def test_returns_extractor_with_camera_feature_dim() -> None:
    cfg = Settings(mock_hardware=True)
    ext = build_vision_feature_extractor(cfg)
    assert ext.feature_dim == cfg.camera.feature_dim


def test_extract_produces_feature_vector() -> None:
    cfg = Settings(mock_hardware=True)
    ext = build_vision_feature_extractor(cfg)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    feats = ext.extract(rgb)
    assert feats.shape == (cfg.camera.feature_dim,)
    assert feats.dtype == np.float32
