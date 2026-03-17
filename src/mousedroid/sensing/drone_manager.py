"""Drone sensor manager — orchestrates all sensor reads for aerial platform.

Extends the ground sensor read pattern with IMU, GPS, and altitude from
the flight controller. Returns :class:`DroneObservationBundle`.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from mousedroid.constants import (
    DEFAULT_AUDIO_CHUNK_SIZE,
    MILLISECONDS_PER_SECOND,
)
from mousedroid.logging.setup import get_logger
from mousedroid.sensing.drone_bundle import DroneObservationBundle

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mousedroid.comms.flight_protocol import FlightControllerProtocol
    from mousedroid.comms.motor_protocol import MotorControlProtocol
    from mousedroid.config.schema import Settings
    from mousedroid.hardware.protocols import AudioProtocol, DistanceSensorProtocol, VisionProtocol

_log = get_logger(__name__)

_IMU_DIM: int = 6
"""IMU data dimension ``[ax, ay, az, gx, gy, gz]``."""


class DroneSensorManager:
    """Orchestrates concurrent sensor reads for drone platform.

    Reads vision, distance, motor/FC state, audio, IMU, GPS, and
    altitude concurrently. Failed reads are zeroed out with the
    corresponding valid_mask entry set to 0.0.

    Args:
        vision: Vision driver implementing :class:`VisionProtocol`.
        distance: Distance sensor implementing :class:`DistanceSensorProtocol`.
        motor_controller: Motor controller implementing :class:`MotorControlProtocol`.
        flight_controller: Flight controller for IMU/GPS/altitude reads.
        cfg: Root application settings.
        microphone: Optional audio driver implementing :class:`AudioProtocol`.
    """

    def __init__(
        self,
        vision: VisionProtocol,
        distance: DistanceSensorProtocol,
        motor_controller: MotorControlProtocol,
        flight_controller: FlightControllerProtocol,
        cfg: Settings,
        microphone: AudioProtocol | None = None,
    ) -> None:
        self._vision = vision
        self._distance = distance
        self._motor_controller = motor_controller
        self._fc = flight_controller
        self._cfg = cfg
        self._microphone = microphone

        self._motor_state_dim = cfg.model.motor_state_dim

        # Ring buffer sizes derived from config loop rates.
        vision_buf_size = max(1, int(cfg.loop.perception_hz))
        distance_buf_size = max(1, int(cfg.loop.ultrasonic_hz))
        motor_buf_size = max(1, int(cfg.loop.control_hz))
        audio_buf_size = max(1, int(cfg.loop.audio_hz))

        self._vision_buf: deque[NDArray[np.float32]] = deque(maxlen=vision_buf_size)
        self._distance_buf: deque[float] = deque(maxlen=distance_buf_size)
        self._motor_buf: deque[NDArray[np.float32]] = deque(maxlen=motor_buf_size)
        self._audio_buf: deque[NDArray[np.float32]] = deque(maxlen=audio_buf_size)

        self._audio_chunk_size = (
            microphone.chunk_size * microphone.channels
            if microphone is not None
            else DEFAULT_AUDIO_CHUNK_SIZE
        )

        _log.info(
            "drone_sensor_manager_init",
            vision_buf=vision_buf_size,
            distance_buf=distance_buf_size,
            motor_buf=motor_buf_size,
            audio_buf=audio_buf_size,
            motor_state_dim=self._motor_state_dim,
        )

    # -- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Start all sensor hardware."""
        await self._vision.start()
        if self._microphone is not None:
            await self._microphone.start()

    async def stop(self) -> None:
        """Stop all sensor hardware."""
        await self._vision.stop()
        if self._microphone is not None:
            await self._microphone.stop()

    # -- Public API --------------------------------------------------------

    async def read_all(self) -> DroneObservationBundle:
        """Read all sensors concurrently and return a drone observation bundle.

        Returns:
            A fully-populated :class:`DroneObservationBundle`.
        """
        t0 = time.monotonic()
        timestamp = t0

        # Kick off all reads concurrently.
        vision_task = asyncio.create_task(self._safe_vision_read())
        distance_task = asyncio.create_task(self._safe_distance_read())
        motor_task = asyncio.create_task(self._safe_motor_read())
        audio_task = asyncio.create_task(self._safe_audio_read())
        imu_task = asyncio.create_task(self._safe_imu_read())
        gps_task = asyncio.create_task(self._safe_gps_read())
        altitude_task = asyncio.create_task(self._safe_altitude_read())

        vision_result, vision_ok = await vision_task
        distance_result, distance_ok = await distance_task
        motor_result, motor_ok = await motor_task
        audio_result, audio_ok = await audio_task
        imu_result, imu_ok = await imu_task
        gps_result, gps_ok = await gps_task
        altitude_result, altitude_ok = await altitude_task

        # Build validity mask for 7 modalities.
        valid_mask = np.array(
            [
                float(vision_ok),
                float(distance_ok),
                float(motor_ok),
                float(audio_ok),
                float(imu_ok),
                float(gps_ok),
                float(altitude_ok),
            ],
            dtype=np.float32,
        )

        # Store in ring buffers.
        self._vision_buf.append(vision_result)
        self._distance_buf.append(distance_result)
        self._motor_buf.append(motor_result)
        self._audio_buf.append(audio_result)

        total_valid = sum([vision_ok, distance_ok, motor_ok, audio_ok, imu_ok, gps_ok, altitude_ok])
        _log.debug(
            "drone_read_all_complete",
            elapsed_ms=(time.monotonic() - t0) * MILLISECONDS_PER_SECOND,
            valid_sensors=total_valid,
        )

        return DroneObservationBundle(
            _timestamp=timestamp,
            _vision_features=vision_result,
            _distance_m=distance_result,
            _motor_state=motor_result,
            _audio_chunk=audio_result,
            _valid_mask=valid_mask,
            _altitude_m=altitude_result,
            _gps_position=gps_result,
            _imu_data=imu_result,
            _gps_fix=gps_ok,
            _imu_healthy=imu_ok,
            _armed=self._fc.armed,
        )

    # -- Private helpers ---------------------------------------------------

    async def _safe_vision_read(self) -> tuple[NDArray[np.float32], bool]:
        """Attempt a vision capture, returning zeros on failure."""
        try:
            features = await self._vision.capture_features()
            return features, True
        except Exception:
            _log.warning("vision_read_failed", exc_info=True)
            return np.zeros(self._cfg.camera.feature_dim, dtype=np.float32), False

    async def _safe_distance_read(self) -> tuple[float, bool]:
        """Attempt a distance read, returning max range on failure."""
        try:
            distance = await self._distance.read_distance_m()
            return distance, True
        except Exception:
            _log.warning("distance_read_failed", exc_info=True)
            return self._distance.max_range_m, False

    async def _safe_motor_read(self) -> tuple[NDArray[np.float32], bool]:
        """Attempt motor state read via MotorControlProtocol."""
        try:
            motor_state = await self._motor_controller.read_state()
            return motor_state, True
        except Exception:
            _log.warning("motor_read_failed", exc_info=True)
            return np.zeros(self._motor_state_dim, dtype=np.float32), False

    async def _safe_audio_read(self) -> tuple[NDArray[np.float32], bool]:
        """Attempt an audio chunk read, returning zeros on failure."""
        if self._microphone is None:
            return np.zeros(self._audio_chunk_size, dtype=np.float32), False
        try:
            chunk = await self._microphone.read_chunk()
            return chunk, True
        except Exception:
            _log.warning("audio_read_failed", exc_info=True)
            return np.zeros(self._audio_chunk_size, dtype=np.float32), False

    async def _safe_imu_read(self) -> tuple[NDArray[np.float32], bool]:
        """Attempt IMU data read from flight controller."""
        try:
            imu_data = await self._fc.get_imu_data()
            return imu_data, True
        except Exception:
            _log.warning("imu_read_failed", exc_info=True)
            return np.zeros(_IMU_DIM, dtype=np.float32), False

    async def _safe_gps_read(self) -> tuple[tuple[float, float, float], bool]:
        """Attempt GPS position read from flight controller."""
        try:
            gps = await self._fc.get_gps_position()
            return gps, True
        except Exception:
            _log.warning("gps_read_failed", exc_info=True)
            return (0.0, 0.0, 0.0), False

    async def _safe_altitude_read(self) -> tuple[float, bool]:
        """Attempt altitude read from flight controller."""
        try:
            alt = await self._fc.get_altitude_m()
            return alt, True
        except Exception:
            _log.warning("altitude_read_failed", exc_info=True)
            return 0.0, False
