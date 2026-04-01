"""Tests for YOLO object detector."""

from __future__ import annotations

import numpy as np

from mousedroid.arm.perception.object_detector import ObjectDetector
from mousedroid.config.schema import ArmPerceptionConfig


def _make_detector() -> ObjectDetector:
    return ObjectDetector(ArmPerceptionConfig())


class TestObjectDetector:
    """Tests for ObjectDetector."""

    def test_init_sets_threshold(self) -> None:
        cfg = ArmPerceptionConfig(yolo_confidence_threshold=0.7)
        det = ObjectDetector(cfg)
        assert det._confidence_threshold == 0.7

    def test_load_model_missing_path(self) -> None:
        det = _make_detector()
        det.load_model()
        assert det._model is None

    def test_detect_without_model_returns_empty(self) -> None:
        det = _make_detector()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(img)
        assert result == []

    def test_detect_after_load_missing_returns_empty(self) -> None:
        det = _make_detector()
        det.load_model()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(img)
        assert result == []
