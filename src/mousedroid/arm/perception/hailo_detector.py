"""YOLO object detection via Hailo-8 neural accelerator.

Offloads YOLO inference to the Hailo-8 M.2 accelerator (INT8, 26 TOPS),
freeing the Jetson GPU for reasoning workloads.  Falls back to the
standard ``ObjectDetector`` (ultralytics on GPU) when the Hailo runtime
is unavailable or inference fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.arm.perception.object_detector import ObjectDetector
from mousedroid.arm.protocols import DetectedObject
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmPerceptionConfig
    from mousedroid.hardware.accelerator.hailo_runtime import HailoRuntimeProtocol

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# NMS post-processing (numpy-only — no torch dependency on accelerator path)
# ---------------------------------------------------------------------------


def _nms_numpy(
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    iou_threshold: float = 0.45,
) -> NDArray[np.intp]:
    """Non-maximum suppression using pure numpy.

    Args:
        boxes: Bounding boxes, shape ``(N, 4)`` as ``[x1, y1, x2, y2]``.
        scores: Confidence scores, shape ``(N,)``.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Indices of boxes to keep.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.intp)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while len(order) > 0:
        i = order[0]
        keep.append(int(i))

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = np.where(union > 0, intersection / union, 0.0)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return np.array(keep, dtype=np.intp)


def _hailo_yolo_postprocess(
    raw_output: NDArray[np.float32],
    confidence_threshold: float,
    iou_threshold: float = 0.45,
    class_names: dict[int, str] | None = None,
) -> list[DetectedObject]:
    """Post-process raw Hailo YOLO output into detected objects.

    Handles the standard YOLO output format:
    ``(N, 5 + num_classes)`` where each row is
    ``[cx, cy, w, h, objectness, class_scores...]``.

    Args:
        raw_output: Raw model output, shape ``(N, 5+C)`` or ``(1, N, 5+C)``.
        confidence_threshold: Minimum confidence to keep a detection.
        iou_threshold: IoU threshold for NMS.
        class_names: Optional mapping of class index to name.

    Returns:
        List of detected objects above confidence threshold after NMS.
    """
    if class_names is None:
        class_names = {}

    # Squeeze batch dimension if present
    if raw_output.ndim == 3 and raw_output.shape[0] == 1:
        raw_output = raw_output[0]

    if raw_output.ndim != 2 or raw_output.shape[1] < 6:
        _log.warning(
            "hailo_yolo_unexpected_output_shape",
            shape=raw_output.shape,
        )
        return []

    # Extract components
    cx = raw_output[:, 0]
    cy = raw_output[:, 1]
    w = raw_output[:, 2]
    h = raw_output[:, 3]
    objectness = raw_output[:, 4]
    class_scores = raw_output[:, 5:]

    # Compute per-class confidence
    class_ids = class_scores.argmax(axis=1)
    max_class_scores = class_scores.max(axis=1)
    confidences = objectness * max_class_scores

    # Filter by confidence
    mask = confidences >= confidence_threshold
    if not np.any(mask):
        return []

    cx = cx[mask]
    cy = cy[mask]
    w = w[mask]
    h = h[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    # Convert cx/cy/w/h to x1/y1/x2/y2
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float64)

    # Apply NMS
    keep = _nms_numpy(boxes, confidences.astype(np.float64), iou_threshold)
    if len(keep) == 0:
        return []

    detections: list[DetectedObject] = []
    for idx in keep:
        cls_id = int(class_ids[idx])
        class_name = class_names.get(cls_id, f"class_{cls_id}")
        detected = DetectedObject(
            object_id=f"{class_name}_{idx}",
            class_name=class_name,
            confidence=float(confidences[idx]),
            position_m=np.zeros(3, dtype=np.float64),  # filled by pose estimator
            orientation_rad=np.zeros(3, dtype=np.float64),
            bbox=boxes[idx],
        )
        detections.append(detected)

    return detections


# ---------------------------------------------------------------------------
# HailoYOLODetector
# ---------------------------------------------------------------------------


class HailoYOLODetector:
    """YOLO object detector using Hailo-8 accelerator.

    Wraps Hailo-8 inference to detect Tower of Hanoi disks or laundry
    garments from RGB images.  Falls back to :class:`ObjectDetector`
    (ultralytics on GPU) when the Hailo runtime is unavailable.

    Args:
        cfg: Arm perception configuration.
        runtime: Hailo-8 runtime instance for inference dispatch.
    """

    def __init__(
        self,
        cfg: ArmPerceptionConfig,
        runtime: HailoRuntimeProtocol,
    ) -> None:
        """Initialise Hailo YOLO detector.

        Args:
            cfg: Arm perception configuration with thresholds.
            runtime: Shared Hailo-8 runtime.
        """
        self._cfg = cfg
        self._runtime = runtime
        self._confidence_threshold = cfg.yolo_confidence_threshold
        self._fallback = ObjectDetector(cfg)
        self._class_names: dict[int, str] = {}
        _log.info(
            "hailo_yolo_detector_init",
            threshold=self._confidence_threshold,
        )

    def load_model(self) -> None:
        """Verify Hailo HEF model is loaded, or load fallback.

        If the Hailo runtime has the YOLO model available, this is a
        no-op (HEF was loaded during runtime start).  Otherwise, loads
        the ultralytics fallback.
        """
        if self._runtime.is_available():
            _log.info("hailo_yolo_model_ready")
        else:
            _log.warning("hailo_unavailable_loading_fallback_yolo")
            self._fallback.load_model()

    def detect(self, rgb_image: NDArray[np.uint8]) -> list[DetectedObject]:
        """Run object detection on an RGB image.

        Args:
            rgb_image: RGB image, shape ``(H, W, 3)``.

        Returns:
            List of detected objects above confidence threshold.
        """
        if not self._runtime.is_available():
            return self._fallback.detect(rgb_image)

        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                # Inside event loop — delegate to fallback to avoid nesting
                return self._fallback.detect(rgb_image)

            raw_output = asyncio.run(self._runtime.run_inference("yolo", rgb_image))

            detections = _hailo_yolo_postprocess(
                raw_output,
                confidence_threshold=self._confidence_threshold,
                class_names=self._class_names,
            )

            _log.debug("hailo_detection_complete", count=len(detections))
            return detections
        except Exception:
            _log.warning("hailo_yolo_detection_failed_using_fallback", exc_info=True)
            return self._fallback.detect(rgb_image)
