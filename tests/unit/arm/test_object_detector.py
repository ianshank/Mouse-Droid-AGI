"""Tests for YOLO object detector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

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


class TestObjectDetectorWithMockedYOLO:
    """Tests for ObjectDetector with mocked ultralytics."""

    def test_load_model_with_existing_path(self, tmp_path) -> None:
        model_file = tmp_path / "yolov8n.pt"
        model_file.write_text("fake")

        cfg = ArmPerceptionConfig(yolo_model_path=str(model_file))
        det = ObjectDetector(cfg)

        mock_yolo_cls = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_yolo_cls.return_value = mock_yolo_instance
        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO = mock_yolo_cls

        with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
            det.load_model()

        mock_yolo_cls.assert_called_once_with(str(model_file))
        assert det._model is mock_yolo_instance

    def test_load_model_ultralytics_not_installed(self, tmp_path) -> None:
        model_file = tmp_path / "yolov8n.pt"
        model_file.write_text("fake")

        cfg = ArmPerceptionConfig(yolo_model_path=str(model_file))
        det = ObjectDetector(cfg)

        with patch.dict("sys.modules", {"ultralytics": None}):
            det.load_model()

        assert det._model is None

    def test_detect_with_mocked_model(self) -> None:
        det = _make_detector()

        # Build mock YOLO result structure
        mock_boxes = MagicMock()
        mock_boxes.conf = torch.tensor([0.95, 0.3])
        mock_boxes.cls = torch.tensor([0, 1])
        mock_boxes.xyxy = torch.tensor([[10.0, 20.0, 100.0, 200.0], [5.0, 5.0, 50.0, 50.0]])
        mock_boxes.__len__ = lambda self: 2

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_result.names = {0: "disk", 1: "garment"}

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        det._model = mock_model

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = det.detect(img)

        # Only the first detection (0.95) is above default threshold (0.5)
        assert len(detections) == 1
        assert detections[0].class_name == "disk"
        assert abs(detections[0].confidence - 0.95) < 1e-5
        assert detections[0].object_id == "disk_0"

    def test_detect_filters_below_threshold(self) -> None:
        cfg = ArmPerceptionConfig(yolo_confidence_threshold=0.99)
        det = ObjectDetector(cfg)

        mock_boxes = MagicMock()
        mock_boxes.conf = torch.tensor([0.8])
        mock_boxes.cls = torch.tensor([0])
        mock_boxes.xyxy = torch.tensor([[10.0, 20.0, 100.0, 200.0]])
        mock_boxes.__len__ = lambda self: 1

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_result.names = {0: "disk"}

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        det._model = mock_model

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = det.detect(img)

        assert len(detections) == 0
