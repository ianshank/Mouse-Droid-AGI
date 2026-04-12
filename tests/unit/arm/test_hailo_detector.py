from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from mousedroid.arm.perception.hailo_detector import (
    HailoYOLODetector,
    _hailo_yolo_postprocess,
    _nms_numpy,
)
from mousedroid.config.schema import ArmPerceptionConfig, HailoConfig
from mousedroid.hardware.accelerator.hailo_runtime import MockHailoRuntime


def _make_perception_cfg(**overrides: Any) -> ArmPerceptionConfig:
    defaults: dict[str, Any] = {
        "depth_camera_type": "mock",
        "yolo_model_path": "models/yolo11_disk_detector.pt",
        "yolo_confidence_threshold": 0.5,
        "yolo_backend": "hailo",
    }
    defaults.update(overrides)
    return ArmPerceptionConfig(**defaults)


def _make_hailo_cfg() -> HailoConfig:
    return HailoConfig(enabled=True)


def _make_started_mock_runtime() -> MockHailoRuntime:
    cfg = _make_hailo_cfg()
    rt = MockHailoRuntime(cfg)
    asyncio.run(rt.start())
    return rt


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------


class TestNmsNumpy:
    def test_empty_input(self) -> None:
        boxes = np.array([], dtype=np.float64).reshape(0, 4)
        scores = np.array([], dtype=np.float64)
        keep = _nms_numpy(boxes, scores)
        assert len(keep) == 0

    def test_single_box(self) -> None:
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float64)
        scores = np.array([0.9], dtype=np.float64)
        keep = _nms_numpy(boxes, scores)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_overlapping_suppressed(self) -> None:
        boxes = np.array(
            [
                [10, 10, 50, 50],
                [12, 12, 52, 52],  # High overlap with box 0
                [100, 100, 200, 200],  # No overlap
            ],
            dtype=np.float64,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float64)
        keep = _nms_numpy(boxes, scores, iou_threshold=0.5)
        assert 0 in keep
        assert 2 in keep
        assert 1 not in keep  # suppressed by box 0

    def test_non_overlapping_all_kept(self) -> None:
        boxes = np.array(
            [
                [0, 0, 10, 10],
                [100, 100, 200, 200],
                [300, 300, 400, 400],
            ],
            dtype=np.float64,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float64)
        keep = _nms_numpy(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 3


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


class TestHailoYoloPostprocess:
    def test_empty_after_confidence_filter(self) -> None:
        # All confidences below threshold
        raw = np.array([[10, 10, 20, 20, 0.1, 0.05, 0.04]], dtype=np.float32)
        result = _hailo_yolo_postprocess(raw, confidence_threshold=0.5)
        assert result == []

    def test_valid_detection(self) -> None:
        # cx=50, cy=50, w=40, h=40, objectness=0.9, class_0=0.95, class_1=0.05
        raw = np.array([[50, 50, 40, 40, 0.9, 0.95, 0.05]], dtype=np.float32)
        result = _hailo_yolo_postprocess(
            raw,
            confidence_threshold=0.5,
            class_names={0: "disk_1", 1: "disk_2"},
        )
        assert len(result) == 1
        assert result[0].class_name == "disk_1"
        assert result[0].confidence == pytest.approx(0.9 * 0.95, abs=1e-5)

    def test_batch_dimension_squeezed(self) -> None:
        raw = np.array([[[50, 50, 40, 40, 0.9, 0.95, 0.05]]], dtype=np.float32)
        result = _hailo_yolo_postprocess(raw, confidence_threshold=0.5)
        assert len(result) == 1

    def test_unexpected_shape_returns_empty(self) -> None:
        raw = np.array([[1, 2, 3]], dtype=np.float32)  # Only 3 cols, need >= 6
        result = _hailo_yolo_postprocess(raw, confidence_threshold=0.5)
        assert result == []

    def test_class_names_default(self) -> None:
        raw = np.array([[50, 50, 40, 40, 0.9, 0.95, 0.05]], dtype=np.float32)
        result = _hailo_yolo_postprocess(raw, confidence_threshold=0.5)
        assert len(result) == 1
        assert result[0].class_name == "class_0"


# ---------------------------------------------------------------------------
# HailoYOLODetector
# ---------------------------------------------------------------------------


class TestHailoYOLODetector:
    def test_detect_with_unavailable_runtime_uses_fallback(self) -> None:
        cfg = _make_perception_cfg()
        rt_cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(rt_cfg)
        # NOT started
        det = HailoYOLODetector(cfg, rt)
        det.load_model()
        # Fallback ObjectDetector has no model loaded — returns empty
        result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result == []

    def test_load_model_with_available_runtime(self) -> None:
        cfg = _make_perception_cfg()
        rt = _make_started_mock_runtime()
        det = HailoYOLODetector(cfg, rt)
        # Should not raise
        det.load_model()
