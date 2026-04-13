"""MouseDroid observation bundle — fused sensor data.

Implements :class:`~mousedroid.sensing.protocol.ObservationProtocol` as a
concrete dataclass that carries one timestep of fused sensor readings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mousedroid.constants import (
    DEFAULT_AUDIO_CHUNK_SIZE,
    DEFAULT_MAX_DISTANCE_M,
    DEFAULT_MOTOR_STATE_DIM,
    DEFAULT_VISION_DIM,
    N_SENSOR_MODALITIES,
)


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
        * ``[4]`` — LiDAR (when present)

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

    _lidar_features: NDArray[np.float32] | None = None
    """LiDAR sector-binned features, shape ``(lidar_dim,)``, or ``None``."""

    _valid_mask: NDArray[np.float32] = field(
        default_factory=lambda: np.ones(N_SENSOR_MODALITIES, dtype=np.float32),
    )
    """Per-modality validity flags, shape ``(n_modalities,)``."""

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
    def lidar_features(self) -> NDArray[np.float32] | None:
        """LiDAR sector-binned features, or ``None`` if LiDAR not configured."""
        return self._lidar_features

    @property
    def valid_mask(self) -> NDArray[np.float32]:
        """Per-sensor validity scores, shape ``(n_modalities,)``."""
        return self._valid_mask

    @property
    def n_modalities(self) -> int:
        """Number of sensor modalities tracked by valid_mask."""
        return len(self._valid_mask)
