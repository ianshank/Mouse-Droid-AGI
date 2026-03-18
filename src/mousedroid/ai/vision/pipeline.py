"""Vision AI pipeline — orchestrates all vision AI models.

Runs YOLO detection, CLIP embedding, face detection, and gesture
recognition on each camera frame and produces a unified
``VisionAIResult``.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.ai.vision.protocols import (
    Detection,
    FaceDetection,
    Gesture,
    VisionAIResult,
)
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.ai.vision.detector import JetsonYOLODetector
    from mousedroid.ai.vision.embedder import CLIPEmbedder
    from mousedroid.ai.vision.face import MediaPipeFaceDetector
    from mousedroid.ai.vision.gesture import MediaPipeGestureRecognizer

_log = get_logger(__name__)


class VisionAIPipeline:
    """Unified vision pipeline running all AI models on each frame.

    Models run concurrently via ``asyncio.gather()`` since they are
    independent and each uses ``to_thread()`` internally.

    Parameters
    ----------
    detector:
        YOLO object detector or None to skip.
    embedder:
        CLIP semantic embedder or None to skip.
    face_detector:
        MediaPipe face detector or None to skip.
    gesture_recognizer:
        MediaPipe gesture recognizer or None to skip.
    """

    def __init__(
        self,
        detector: JetsonYOLODetector | None = None,
        embedder: CLIPEmbedder | None = None,
        face_detector: MediaPipeFaceDetector | None = None,
        gesture_recognizer: MediaPipeGestureRecognizer | None = None,
    ) -> None:
        self._detector = detector
        self._embedder = embedder
        self._face_detector = face_detector
        self._gesture_recognizer = gesture_recognizer

    async def start(self) -> None:
        """Start all configured AI models concurrently."""
        coros = []
        if self._detector is not None:
            coros.append(self._detector.start())
        if self._embedder is not None:
            coros.append(self._embedder.start())
        if self._face_detector is not None:
            coros.append(self._face_detector.start())
        if self._gesture_recognizer is not None:
            coros.append(self._gesture_recognizer.start())
        if coros:
            await asyncio.gather(*coros)
        _log.info("vision_ai_pipeline_started", models=len(coros))

    async def stop(self) -> None:
        """Stop all configured AI models."""
        coros = []
        if self._detector is not None:
            coros.append(self._detector.stop())
        if self._embedder is not None:
            coros.append(self._embedder.stop())
        if self._face_detector is not None:
            coros.append(self._face_detector.stop())
        if self._gesture_recognizer is not None:
            coros.append(self._gesture_recognizer.stop())
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)
        _log.info("vision_ai_pipeline_stopped")

    async def process(self, frame: NDArray[np.uint8]) -> VisionAIResult:
        """Run all vision AI models on a single frame.

        Models are run concurrently for maximum throughput. Failures in
        individual models are logged but do not crash the pipeline.

        Args:
            frame: BGR image, shape ``(H, W, 3)``.

        Returns:
            Unified vision AI result.
        """
        # Build concurrent tasks with error handling wrappers
        det_task = (
            self._safe_detect(frame) if self._detector else _empty_detections()
        )
        emb_task = (
            self._safe_embed(frame) if self._embedder else _zero_embedding()
        )
        face_task = (
            self._safe_detect_faces(frame)
            if self._face_detector
            else _empty_faces()
        )
        gesture_task = (
            self._safe_recognize(frame)
            if self._gesture_recognizer
            else _empty_gestures()
        )

        detections, embedding, faces, gestures = await asyncio.gather(
            det_task, emb_task, face_task, gesture_task,
        )

        return VisionAIResult(
            detections=detections,
            embedding=embedding,
            faces=faces,
            gestures=gestures,
            frame_shape=frame.shape,
            timestamp=time.time(),
        )

    async def _safe_detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run detection with graceful error handling."""
        try:
            return await self._detector.detect(frame)
        except Exception:
            _log.warning("vision_detection_failed", exc_info=True)
            return []

    async def _safe_embed(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Run embedding with graceful error handling."""
        try:
            return await self._embedder.embed(frame)
        except Exception:
            _log.warning("vision_embedding_failed", exc_info=True)
            return np.zeros(0, dtype=np.float32)

    async def _safe_detect_faces(self, frame: NDArray[np.uint8]) -> list[FaceDetection]:
        """Run face detection with graceful error handling."""
        try:
            return await self._face_detector.detect_faces(frame)
        except Exception:
            _log.warning("vision_face_detection_failed", exc_info=True)
            return []

    async def _safe_recognize(self, frame: NDArray[np.uint8]) -> list[Gesture]:
        """Run gesture recognition with graceful error handling."""
        try:
            return await self._gesture_recognizer.recognize(frame)
        except Exception:
            _log.warning("vision_gesture_recognition_failed", exc_info=True)
            return []


# Fallback coroutines for disabled models
async def _empty_detections() -> list[Detection]:
    return []

async def _zero_embedding() -> NDArray[np.float32]:
    return np.zeros(0, dtype=np.float32)

async def _empty_faces() -> list[FaceDetection]:
    return []

async def _empty_gestures() -> list[Gesture]:
    return []
