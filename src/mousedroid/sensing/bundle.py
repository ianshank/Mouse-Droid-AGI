"""MouseDroid observation bundle — fused sensor data.

Implements :class:`~mousedroid.sensing.protocol.ObservationProtocol` as a
concrete dataclass that carries one timestep of fused sensor readings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.constants import (
    DEFAULT_AUDIO_CHUNK_SIZE,
    DEFAULT_MAX_DISTANCE_M,
    DEFAULT_MOTOR_STATE_DIM,
    DEFAULT_VISION_DIM,
    N_SENSOR_MODALITIES,
)

if TYPE_CHECKING:
    from mousedroid.ai.audio.protocols import AudioAIResult
    from mousedroid.ai.fusion.sensor_fusion import FusedDepthResult
    from mousedroid.ai.vision.protocols import VisionAIResult


@dataclass
class MouseDroidObservationBundle:
    """Fused observation from all MouseDroid sensors.

    Each control-loop iteration produces one bundle.  Failed sensor reads
    are represented by zeroed-out arrays and a ``0.0`` entry in the
    corresponding :pyattr:`valid_mask` slot.

    Slot layout for :pyattr:`valid_mask`:
        * ``[0]`` — vision
        * ``[1]`` — ultrasonic
        * ``[2]`` — motor / ESP32
        * ``[3]`` — audio / microphone

    Implements :class:`~mousedroid.sensing.protocol.ObservationProtocol`.
    """

    _timestamp: float = field(default_factory=time.monotonic)
    """Monotonic timestamp captured when the bundle is created."""

    _vision_features: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(DEFAULT_VISION_DIM, dtype=np.float32),
    )
    """Vision feature vector, shape ``(feature_dim,)``."""

    _distance_m: float = DEFAULT_MAX_DISTANCE_M
    """Forward ultrasonic distance in metres (defaults to max range)."""

    _motor_state: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(DEFAULT_MOTOR_STATE_DIM, dtype=np.float32),
    )
    """Motor state ``[vx, vy, omega, battery_v]``, shape ``(4,)``."""

    _audio_chunk: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(DEFAULT_AUDIO_CHUNK_SIZE, dtype=np.float32),
    )
    """Audio samples, shape ``(chunk_size * channels,)``."""

    _valid_mask: NDArray[np.float32] = field(
        default_factory=lambda: np.ones(N_SENSOR_MODALITIES, dtype=np.float32),
    )
    """Per-modality validity flags, shape ``(n_modalities,)``."""

    # --- AI pipeline results (optional) ---

    _vision_ai_result: VisionAIResult | None = None
    """Vision AI pipeline result (detections, embeddings, faces, gestures)."""

    _audio_ai_result: AudioAIResult | None = None
    """Audio AI pipeline result (transcription, wake word, sound events)."""

    _fused_depth: FusedDepthResult | None = None
    """Fused depth map from MiDaS + ultrasonic Kalman filter."""

    # --- Configurable label/keyword sets ---

    _person_class_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"person"}),
    )
    """Class names treated as 'human' for Law 1 safety."""

    _stop_keywords: frozenset[str] = field(
        default_factory=lambda: frozenset({"stop", "halt", "freeze", "no", "danger"}),
    )
    """Voice-command keywords that trigger a stop (Law 2)."""

    _law2_gesture_labels: frozenset[str] = field(
        default_factory=lambda: frozenset({"stop"}),
    )
    """Gesture labels that trigger a stop (Law 2)."""

    # -- ObservationProtocol properties ------------------------------------

    @property
    def timestamp(self) -> float:
        """Monotonic timestamp in seconds."""
        return self._timestamp

    @property
    def vision_features(self) -> NDArray[np.float32]:
        """Vision feature vector, shape ``(feature_dim,)``."""
        return self._vision_features

    @property
    def distance_m(self) -> float:
        """Forward distance measurement in metres."""
        return self._distance_m

    @property
    def motor_state(self) -> NDArray[np.float32]:
        """Motor state ``[vx, vy, omega, battery_v]``, shape ``(4,)``."""
        return self._motor_state

    @property
    def audio_chunk(self) -> NDArray[np.float32]:
        """Audio samples, shape ``(chunk_size * channels,)``."""
        return self._audio_chunk

    @property
    def valid_mask(self) -> NDArray[np.float32]:
        """Per-sensor validity scores, shape ``(n_modalities,)``."""
        return self._valid_mask

    @property
    def n_modalities(self) -> int:
        """Number of sensor modalities tracked by valid_mask."""
        return N_SENSOR_MODALITIES

    @property
    def vision_ai_result(self) -> VisionAIResult | None:
        """Vision AI pipeline result (detections, embeddings, faces, gestures)."""
        return self._vision_ai_result

    @property
    def audio_ai_result(self) -> AudioAIResult | None:
        """Audio AI pipeline result (transcription, wake word, sound events)."""
        return self._audio_ai_result

    @property
    def fused_depth(self) -> FusedDepthResult | None:
        """Fused depth map from MiDaS + ultrasonic Kalman filter."""
        return self._fused_depth

    # --- Three Laws safety properties (derived from AI results) -----------

    @property
    def human_detected(self) -> bool:
        """True if YOLO detected at least one person in the current frame."""
        if self._vision_ai_result is None:
            return False
        return any(
            d.class_name in self._person_class_names
            for d in self._vision_ai_result.detections
        )

    @property
    def human_dist_m(self) -> float:
        """Closest detected person distance in metres.

        Uses Kalman-fused depth if available, otherwise falls back to the
        raw ultrasonic reading.  Returns ``inf`` when no person is detected.
        """
        if not self.human_detected:
            return float("inf")
        # Prefer fused depth centre estimate
        if self._fused_depth is not None:
            return float(self._fused_depth.center_distance_m)
        # Fall back to raw ultrasonic
        return self._distance_m

    @property
    def gesture_stop_commanded(self) -> bool:
        """True if a stop gesture was recognised this frame (Law 2)."""
        if self._vision_ai_result is None or self._vision_ai_result.gestures is None:
            return False
        return any(g.label in self._law2_gesture_labels for g in self._vision_ai_result.gestures)

    @property
    def voice_stop_commanded(self) -> bool:
        """True if the ASR transcription contains a stop/halt command (Law 2)."""
        if self._audio_ai_result is None or self._audio_ai_result.transcription is None:
            return False
        text = self._audio_ai_result.transcription.text.lower()
        return any(kw in text for kw in self._stop_keywords)
