"""YOLO-based object detection for disks and garments.

Wraps YOLO inference to detect Tower of Hanoi disks or laundry
garments from RGB images, returning bounding boxes with class labels.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from mousedroid.arm.protocols import DetectedObject
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmPerceptionConfig

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol — implemented by ObjectDetector and HailoYOLODetector
# ---------------------------------------------------------------------------


@runtime_checkable
class ObjectDetectorProtocol(Protocol):
    """Interface for object detectors used by the perception facade."""

    def load_model(self) -> None:
        """Load detection model weights."""
        ...  # pragma: no cover

    def detect(self, rgb_image: NDArray[np.uint8]) -> list[DetectedObject]:
        """Run detection on an RGB image.

        Args:
            rgb_image: RGB image, shape ``(H, W, 3)``.

        Returns:
            List of detected objects.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Concrete implementation — ultralytics YOLO on GPU
# ---------------------------------------------------------------------------


class ObjectDetector:
    """YOLO-based object detector for arm manipulation tasks.

    Loads a YOLO model and runs inference on RGB images to produce
    detected objects with bounding boxes and class labels.

    Args:
        cfg: Arm perception configuration.
    """

    def __init__(self, cfg: ArmPerceptionConfig) -> None:
        """Initialise object detector.

        Args:
            cfg: Arm perception configuration with model path and thresholds.
        """
        self._cfg = cfg
        self._model_path = Path(cfg.yolo_model_path)
        self._confidence_threshold = cfg.yolo_confidence_threshold
        self._model: Any = None
        _log.info(
            "object_detector_init",
            model_path=str(self._model_path),
            threshold=self._confidence_threshold,
        )

    def load_model(self) -> None:
        """Load YOLO model weights.

        Attempts to load from the configured path. Falls back to a
        stub detector if the model file is not found (useful for testing).
        """
        if self._model_path.exists():
            try:
                ultralytics_module = importlib.import_module("ultralytics")
                ultralytics_api = cast(Any, ultralytics_module)
                yolo_ctor = cast(
                    Callable[[str], Any],
                    ultralytics_api.YOLO,
                )
                self._model = yolo_ctor(str(self._model_path))
                _log.info("yolo_model_loaded", path=str(self._model_path))
            except (ImportError, AttributeError):
                _log.warning("ultralytics_not_installed", fallback="stub_detector")
                self._model = None
        else:
            _log.warning("yolo_model_not_found", path=str(self._model_path))
            self._model = None

    def detect(self, rgb_image: NDArray[np.uint8]) -> list[DetectedObject]:
        """Run object detection on an RGB image.

        Args:
            rgb_image: RGB image, shape ``(H, W, 3)``.

        Returns:
            List of detected objects above confidence threshold.
        """
        if self._model is None:
            _log.debug("detector_no_model", action="returning_empty")
            return []

        results = self._model(rgb_image, verbose=False)
        detections: list[DetectedObject] = []

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < self._confidence_threshold:
                    continue

                cls_id = int(boxes.cls[i])
                class_name = result.names[cls_id]
                bbox = boxes.xyxy[i].cpu().numpy()

                detected = DetectedObject(
                    object_id=f"{class_name}_{i}",
                    class_name=class_name,
                    confidence=conf,
                    position_m=np.zeros(3, dtype=np.float64),  # filled by pose estimator
                    orientation_rad=np.zeros(3, dtype=np.float64),
                    bbox=bbox.astype(np.float64),
                )
                detections.append(detected)

        _log.debug("detection_complete", count=len(detections))
        return detections
