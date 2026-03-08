"""MouseDroid observation bundle — fused sensor data.

Implements :class:`~mousedroid.sensing.protocol.ObservationProtocol` as a
concrete dataclass that carries one timestep of fused sensor readings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

_N_MODALITIES: int = 3
"""Number of sensor modalities tracked: vision, ultrasonic, motor."""


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

    Implements :class:`~mousedroid.sensing.protocol.ObservationProtocol`.
    """

    _timestamp: float = field(default_factory=time.monotonic)
    """Monotonic timestamp captured when the bundle is created."""

    _vision_features: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(256, dtype=np.float32),
    )
    """Vision feature vector, shape ``(feature_dim,)``."""

    _distance_m: float = 4.0
    """Forward ultrasonic distance in metres (defaults to max range)."""

    _motor_state: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32),
    )
    """Motor state ``[vx, vy, omega, battery_v]``, shape ``(4,)``."""

    _valid_mask: NDArray[np.float32] = field(
        default_factory=lambda: np.ones(_N_MODALITIES, dtype=np.float32),
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
    def valid_mask(self) -> NDArray[np.float32]:
        """Per-sensor validity scores, shape ``(n_modalities,)``."""
        return self._valid_mask

    @property
    def n_modalities(self) -> int:
        """Number of sensor modalities tracked by valid_mask."""
        return _N_MODALITIES
