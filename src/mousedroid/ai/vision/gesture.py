"""MediaPipe gesture recognizer for Jetson.

Implements ``GestureRecognizerProtocol`` using MediaPipe Hands for
hand landmark detection and simple gesture classification.
Gesture results feed into Law 2 (human command obedience).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.ai.vision.protocols import Gesture
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import VisionAIConfig

_log = get_logger(__name__)

_mp: Any
try:
    import mediapipe as _mp  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _mp = None

# Finger tip and PIP landmark indices (MediaPipe hand model)
_THUMB_TIP = 4
_INDEX_TIP = 8
_MIDDLE_TIP = 12
_RING_TIP = 16
_PINKY_TIP = 20
_INDEX_PIP = 6
_MIDDLE_PIP = 10
_RING_PIP = 14
_PINKY_PIP = 18
_WRIST = 0


class MediaPipeGestureRecognizer:
    """MediaPipe gesture recognizer implementing ``GestureRecognizerProtocol``.

    Detects hand landmarks and classifies gestures:
    - ``stop``: Open palm, all fingers extended (Law 2: stop command)
    - ``point``: Index finger extended, others closed (directional)
    - ``come``: All fingers curled inward (beckoning)
    - ``thumbs_up``: Thumb extended upward, others closed (affirmative)

    All blocking inference is delegated to ``asyncio.to_thread()``.
    """

    def __init__(self, cfg: VisionAIConfig) -> None:
        self._cfg = cfg
        self._hands: Any = None
        self._last_inference_t: float = 0.0
        self._min_interval: float = 1.0 / cfg.gesture_max_hz

    async def start(self) -> None:
        """Initialise MediaPipe Hands."""
        if _mp is None:
            msg = "mediapipe is not installed — install mousedroid[ai-vision]"
            raise RuntimeError(msg)
        await asyncio.to_thread(self._init_hands)
        _log.info("gesture_recognizer_started")

    def _init_hands(self) -> None:
        """Create MediaPipe Hands instance (blocking)."""
        mp_hands = _mp.solutions.hands
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self._cfg.gesture_min_detection_confidence,
            min_tracking_confidence=self._cfg.gesture_min_tracking_confidence,
        )

    async def stop(self) -> None:
        """Release MediaPipe resources."""
        if self._hands is not None:
            self._hands.close()
            self._hands = None
        _log.info("gesture_recognizer_stopped")

    async def recognize(self, frame: NDArray[np.uint8]) -> list[Gesture]:
        """Recognise hand gestures in a BGR frame.

        Rate-limited to ``gesture_max_hz``.

        Args:
            frame: BGR image, shape ``(H, W, 3)``.

        Returns:
            List of recognised gestures.
        """
        now = time.monotonic()
        if now - self._last_inference_t < self._min_interval:
            return []
        self._last_inference_t = now

        if self._hands is None:
            return []

        return await asyncio.to_thread(self._infer, frame)

    def _infer(self, frame: NDArray[np.uint8]) -> list[Gesture]:
        """Run gesture recognition (blocking)."""
        rgb = frame[:, :, ::-1].copy()
        results = self._hands.process(rgb)

        gestures: list[Gesture] = []
        if results.multi_hand_landmarks is None:
            return gestures

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = "right"
            if results.multi_handedness and i < len(results.multi_handedness):
                handedness = results.multi_handedness[i].classification[0].label.lower()

            label, confidence = self._classify_gesture(hand_landmarks)
            if label is not None:
                gestures.append(
                    Gesture(label=label, confidence=confidence, hand=handedness)
                )

        return gestures

    def _classify_gesture(
        self, landmarks: Any,
    ) -> tuple[str | None, float]:
        """Classify a hand gesture from landmarks.

        Returns:
            Tuple of (gesture_label, confidence). Label is None if
            no gesture matches.
        """
        lm = landmarks.landmark
        fingers_up = self._count_fingers_up(lm)

        if fingers_up == 5:
            return "stop", 0.9
        if fingers_up == 0:
            return "come", 0.7
        if fingers_up == 1 and lm[_INDEX_TIP].y < lm[_INDEX_PIP].y:
            return "point", 0.8
        if (
            fingers_up == 1
            and lm[_THUMB_TIP].y < lm[_WRIST].y
            and lm[_INDEX_TIP].y > lm[_INDEX_PIP].y
        ):
            return "thumbs_up", 0.75

        return None, 0.0

    @staticmethod
    def _count_fingers_up(lm: Any) -> int:
        """Count how many fingers are extended."""
        count = 0
        # Thumb: tip is to the left of IP joint (for right hand)
        if abs(lm[_THUMB_TIP].x - lm[_WRIST].x) > abs(lm[3].x - lm[_WRIST].x):
            count += 1
        # Other fingers: tip above PIP
        for tip, pip_ in [
            (_INDEX_TIP, _INDEX_PIP),
            (_MIDDLE_TIP, _MIDDLE_PIP),
            (_RING_TIP, _RING_PIP),
            (_PINKY_TIP, _PINKY_PIP),
        ]:
            if lm[tip].y < lm[pip_].y:
                count += 1
        return count
