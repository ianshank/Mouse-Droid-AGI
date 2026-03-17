"""Drone observation bundle — fused sensor data for aerial platform.

Implements :class:`~mousedroid.sensing.protocol.ObservationProtocol` with
additional drone-specific fields (altitude, GPS, IMU).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mousedroid.constants import (
    DEFAULT_AUDIO_CHUNK_SIZE,
    DEFAULT_DRONE_MOTOR_STATE_DIM,
    DEFAULT_MAX_DISTANCE_M,
    DEFAULT_VISION_DIM,
    N_DRONE_SENSOR_MODALITIES,
)


@dataclass
class DroneObservationBundle:
    """Fused observation from all drone sensors.

    Each control-loop iteration produces one bundle.  Failed sensor reads
    are represented by zeroed-out arrays and a ``0.0`` entry in the
    corresponding :pyattr:`valid_mask` slot.

    Slot layout for :pyattr:`valid_mask`:
        * ``[0]`` — vision
        * ``[1]`` — distance (ultrasonic/lidar)
        * ``[2]`` — motor / flight controller
        * ``[3]`` — audio / microphone
        * ``[4]`` — IMU
        * ``[5]`` — GPS
        * ``[6]`` — altitude (barometer)

    Implements :class:`~mousedroid.sensing.protocol.ObservationProtocol`.
    """

    _timestamp: float = field(default_factory=time.monotonic)
    """Monotonic timestamp captured when the bundle is created."""

    _vision_features: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(DEFAULT_VISION_DIM, dtype=np.float32),
    )
    """Vision feature vector, shape ``(feature_dim,)``."""

    _distance_m: float = DEFAULT_MAX_DISTANCE_M
    """Forward distance in metres (defaults to max range)."""

    _motor_state: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(DEFAULT_DRONE_MOTOR_STATE_DIM, dtype=np.float32),
    )
    """Motor state ``[vx, vy, vz, yaw_rate, altitude, battery_v, armed]``."""

    _audio_chunk: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(DEFAULT_AUDIO_CHUNK_SIZE, dtype=np.float32),
    )
    """Audio samples, shape ``(chunk_size * channels,)``."""

    _valid_mask: NDArray[np.float32] = field(
        default_factory=lambda: np.ones(N_DRONE_SENSOR_MODALITIES, dtype=np.float32),
    )
    """Per-modality validity flags, shape ``(n_modalities,)``."""

    _altitude_m: float = 0.0
    """Current altitude AGL in metres."""

    _gps_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """GPS position ``(lat_deg, lon_deg, alt_msl_m)``."""

    _imu_data: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(6, dtype=np.float32),
    )
    """IMU data ``[ax, ay, az, gx, gy, gz]``."""

    _gps_fix: bool = True
    """Whether GPS has a valid fix."""

    _imu_healthy: bool = True
    """Whether IMU data is valid."""

    _armed: bool = False
    """Whether the motors are armed."""

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
        """Motor state array."""
        return self._motor_state

    @property
    def audio_chunk(self) -> NDArray[np.float32]:
        """Audio samples."""
        return self._audio_chunk

    @property
    def valid_mask(self) -> NDArray[np.float32]:
        """Per-sensor validity scores."""
        return self._valid_mask

    @property
    def n_modalities(self) -> int:
        """Number of sensor modalities tracked by valid_mask."""
        return N_DRONE_SENSOR_MODALITIES

    # -- Drone-specific properties -----------------------------------------

    @property
    def altitude_m(self) -> float:
        """Current altitude AGL in metres."""
        return self._altitude_m

    @property
    def gps_position(self) -> tuple[float, float, float]:
        """GPS position ``(lat_deg, lon_deg, alt_msl_m)``."""
        return self._gps_position

    @property
    def imu_data(self) -> NDArray[np.float32]:
        """IMU data ``[ax, ay, az, gx, gy, gz]``."""
        return self._imu_data

    @property
    def gps_fix(self) -> bool:
        """Whether GPS has a valid fix."""
        return self._gps_fix

    @property
    def imu_healthy(self) -> bool:
        """Whether IMU data is valid."""
        return self._imu_healthy

    @property
    def armed(self) -> bool:
        """Whether the motors are armed."""
        return self._armed
