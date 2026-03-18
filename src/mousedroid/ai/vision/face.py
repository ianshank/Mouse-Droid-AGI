"""MediaPipe face detector for Jetson.

Implements ``FaceDetectorProtocol`` using MediaPipe's short-range face
detection model. Face detections feed human proximity data into the
Three Laws safety layer (Law 1).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.ai.vision.protocols import FaceDetection
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import VisionAIConfig

_log = get_logger(__name__)

_mp: Any
try:
    import mediapipe as _mp  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _mp = None


class MediaPipeFaceDetector:
    """MediaPipe face detector implementing ``FaceDetectorProtocol``.

    Uses the short-range face detection model (~5 MB) for detecting
    human faces to inform the Three Laws safety layer.

    All blocking inference is delegated to ``asyncio.to_thread()``.
    """

    def __init__(self, cfg: VisionAIConfig) -> None:
        self._cfg = cfg
        self._detector: Any = None
        self._last_inference_t: float = 0.0
        self._min_interval: float = 1.0 / cfg.face_max_hz

    async def start(self) -> None:
        """Initialise MediaPipe face detection."""
        if _mp is None:
            msg = "mediapipe is not installed — install mousedroid[ai-vision]"
            raise RuntimeError(msg)
        await asyncio.to_thread(self._init_detector)
        _log.info("face_detector_started")

    def _init_detector(self) -> None:
        """Create MediaPipe face detection instance (blocking)."""
        mp_face = _mp.solutions.face_detection
        self._detector = mp_face.FaceDetection(
            model_selection=0,  # 0 = short-range (< 2m)
            min_detection_confidence=0.5,
        )

    async def stop(self) -> None:
        """Release MediaPipe resources."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        _log.info("face_detector_stopped")

    async def detect_faces(self, frame: NDArray[np.uint8]) -> list[FaceDetection]:
        """Detect faces in a BGR frame.

        Rate-limited to ``face_max_hz``.

        Args:
            frame: BGR image, shape ``(H, W, 3)``.

        Returns:
            List of face detections.
        """
        now = time.monotonic()
        if now - self._last_inference_t < self._min_interval:
            return []
        self._last_inference_t = now

        if self._detector is None:
            return []

        return await asyncio.to_thread(self._infer, frame)

    def _infer(self, frame: NDArray[np.uint8]) -> list[FaceDetection]:
        """Run MediaPipe face detection (blocking)."""
        # MediaPipe expects RGB
        rgb = frame[:, :, ::-1].copy()
        results = self._detector.process(rgb)

        faces: list[FaceDetection] = []
        if results.detections is None:
            return faces

        h, w = frame.shape[:2]
        for detection in results.detections:
            bbox_rel = detection.location_data.relative_bounding_box
            x1 = bbox_rel.xmin * w
            y1 = bbox_rel.ymin * h
            x2 = (bbox_rel.xmin + bbox_rel.width) * w
            y2 = (bbox_rel.ymin + bbox_rel.height) * h
            conf = detection.score[0] if detection.score else 0.0
            faces.append(
                FaceDetection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                )
            )

        return faces
