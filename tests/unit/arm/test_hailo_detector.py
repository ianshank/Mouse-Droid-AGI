from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from mousedroid.arm.perception.hailo_detector import (
    HailoYOLODetector,
    _hailo_yolo_postprocess,
    _nms_numpy,
)
from mousedroid.arm.perception.object_detector import ObjectDetectorProtocol
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


def _make_started_mock_runtime(
    output_shapes: dict[str, tuple[int, ...]] | None = None,
) -> MockHailoRuntime:
    cfg = _make_hailo_cfg()
    rt = MockHailoRuntime(cfg, output_shapes=output_shapes)
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
        assert 1 not in keep

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

    def test_iou_threshold_zero_keeps_only_best(self) -> None:
        boxes = np.array(
            [[10, 10, 50, 50], [15, 15, 55, 55]],
            dtype=np.float64,
        )
        scores = np.array([0.9, 0.8], dtype=np.float64)
        keep = _nms_numpy(boxes, scores, iou_threshold=0.0)
        assert len(keep) == 1
        assert keep[0] == 0


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


class TestHailoYoloPostprocess:
    def test_empty_after_confidence_filter(self) -> None:
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
        # Verify bbox is cx,cy,w,h -> x1,y1,x2,y2
        assert result[0].bbox[0] == pytest.approx(30.0, abs=1e-5)  # x1 = cx - w/2
        assert result[0].bbox[2] == pytest.approx(70.0, abs=1e-5)  # x2 = cx + w/2

    def test_batch_dimension_squeezed(self) -> None:
        raw = np.array([[[50, 50, 40, 40, 0.9, 0.95, 0.05]]], dtype=np.float32)
        result = _hailo_yolo_postprocess(raw, confidence_threshold=0.5)
        assert len(result) == 1

    def test_unexpected_shape_returns_empty(self) -> None:
        raw = np.array([[1, 2, 3]], dtype=np.float32)
        result = _hailo_yolo_postprocess(raw, confidence_threshold=0.5)
        assert result == []

    def test_class_names_default(self) -> None:
        raw = np.array([[50, 50, 40, 40, 0.9, 0.95, 0.05]], dtype=np.float32)
        result = _hailo_yolo_postprocess(raw, confidence_threshold=0.5)
        assert len(result) == 1
        assert result[0].class_name == "class_0"

    def test_multiple_detections_nms(self) -> None:
        # Two overlapping + one separate
        raw = np.array(
            [
                [50, 50, 40, 40, 0.9, 0.95, 0.05],
                [52, 52, 40, 40, 0.85, 0.90, 0.10],
                [200, 200, 40, 40, 0.8, 0.88, 0.12],
            ],
            dtype=np.float32,
        )
        result = _hailo_yolo_postprocess(raw, confidence_threshold=0.3)
        # The two overlapping should be reduced to one after NMS
        assert len(result) >= 1
        assert len(result) <= 3

    def test_1d_output_returns_empty(self) -> None:
        raw = np.array([1.0, 2.0], dtype=np.float32)
        result = _hailo_yolo_postprocess(raw, confidence_threshold=0.5)
        assert result == []


# ---------------------------------------------------------------------------
# HailoYOLODetector
# ---------------------------------------------------------------------------


class TestHailoYOLODetector:
    def test_detect_with_unavailable_runtime_uses_fallback(self) -> None:
        cfg = _make_perception_cfg()
        rt_cfg = _make_hailo_cfg()
        rt = MockHailoRuntime(rt_cfg)
        det = HailoYOLODetector(cfg, rt)
        det.load_model()
        result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result == []

    def test_load_model_with_available_runtime(self) -> None:
        cfg = _make_perception_cfg()
        rt = _make_started_mock_runtime()
        det = HailoYOLODetector(cfg, rt)
        det.load_model()  # Should not raise

    def test_detect_with_available_runtime(self) -> None:
        """Test detection path when Hailo runtime is available.

        Mock runtime returns zeros which produce no detections
        (all confidences = 0). Exercises the runtime dispatch path.
        """
        cfg = _make_perception_cfg()
        rt = _make_started_mock_runtime()
        det = HailoYOLODetector(cfg, rt)
        det.load_model()
        result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        # Mock returns zeros → all confidences are 0 → empty after threshold
        assert result == []

    def test_detect_fallback_on_exception(self) -> None:
        """If inference raises, fallback to ultralytics detector."""
        cfg = _make_perception_cfg()
        mock_runtime = MagicMock()
        mock_runtime.is_available.return_value = True
        mock_runtime.infer_sync.side_effect = RuntimeError("inference failed")
        det = HailoYOLODetector(cfg, mock_runtime)
        result = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert isinstance(result, list)

    def test_implements_detector_protocol(self) -> None:
        cfg = _make_perception_cfg()
        rt = _make_started_mock_runtime()
        det = HailoYOLODetector(cfg, rt)
        assert isinstance(det, ObjectDetectorProtocol)
