"""Sensor manager — orchestrates all sensor reads into an observation bundle.

Reads vision, ultrasonic, ESP32 motor, and audio data concurrently, handles
failures gracefully, and maintains per-sensor ring buffers.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from mousedroid.constants import (
    DEFAULT_AUDIO_CHUNK_SIZE,
    DEFAULT_MOTOR_STATE_DIM,
    MILLISECONDS_PER_SECOND,
)
from mousedroid.logging.setup import get_logger
from mousedroid.sensing.bundle import MouseDroidObservationBundle

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mousedroid.comms.protocol import ESP32CommProtocol
    from mousedroid.config.schema import Settings
    from mousedroid.hardware.protocols import AudioProtocol, DistanceSensorProtocol, VisionProtocol

_log = get_logger(__name__)


class SensorManager:
    """Orchestrates concurrent sensor reads and produces fused bundles.

    Each sensor has a ring buffer sized according to the corresponding
    loop frequency in config.  Failed reads zero out the value and mark
    the modality invalid in the observation bundle.

    Args:
        vision: Vision driver implementing :class:`VisionProtocol`.
        distance: Distance sensor implementing :class:`DistanceSensorProtocol`.
        esp32: ESP32 comms implementing :class:`ESP32CommProtocol`.
        cfg: Root application settings.
        microphone: Optional audio driver implementing :class:`AudioProtocol`.
    """

    def __init__(
        self,
        vision: VisionProtocol,
        distance: DistanceSensorProtocol,
        esp32: ESP32CommProtocol,
        cfg: Settings,
        microphone: AudioProtocol | None = None,
    ) -> None:
        self._vision = vision
        self._distance = distance
        self._esp32 = esp32
        self._cfg = cfg
        self._microphone = microphone

        # Ring buffer sizes derived from config loop rates.
        vision_buf_size = max(1, int(cfg.loop.perception_hz))
        ultrasonic_buf_size = max(1, int(cfg.loop.ultrasonic_hz))
        motor_buf_size = max(1, int(cfg.loop.control_hz))
        audio_buf_size = max(1, int(cfg.loop.audio_hz))

        self._vision_buf: deque[NDArray[np.float32]] = deque(maxlen=vision_buf_size)
        self._distance_buf: deque[float] = deque(maxlen=ultrasonic_buf_size)
        self._motor_buf: deque[NDArray[np.float32]] = deque(maxlen=motor_buf_size)
        self._audio_buf: deque[NDArray[np.float32]] = deque(maxlen=audio_buf_size)

        # Determine audio chunk size for zero-fill on failure.
        self._audio_chunk_size = (
            microphone.chunk_size * microphone.channels
            if microphone is not None
            else DEFAULT_AUDIO_CHUNK_SIZE
        )

        _log.info(
            "sensor_manager_init",
            vision_buf=vision_buf_size,
            ultrasonic_buf=ultrasonic_buf_size,
            motor_buf=motor_buf_size,
            audio_buf=audio_buf_size,
            microphone_enabled=microphone is not None,
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

    async def read_all(self) -> MouseDroidObservationBundle:
        """Read all sensors concurrently and return a fused observation bundle.

        Sensors that fail are represented by zeroed-out data with the
        corresponding :pyattr:`valid_mask` slot set to ``0.0``.

        Returns:
            A fully-populated :class:`MouseDroidObservationBundle`.
        """
        t0 = time.monotonic()
        timestamp = t0

        # Kick off all reads concurrently.
        vision_task = asyncio.create_task(self._safe_vision_read())
        distance_task = asyncio.create_task(self._safe_distance_read())
        motor_task = asyncio.create_task(self._safe_motor_read())
        audio_task = asyncio.create_task(self._safe_audio_read())

        vision_result, vision_ok = await vision_task
        distance_result, distance_ok = await distance_task
        motor_result, motor_ok = await motor_task
        audio_result, audio_ok = await audio_task

        # Build validity mask: vision=0, ultrasonic=1, motor=2, audio=3.
        valid_mask = np.array(
            [float(vision_ok), float(distance_ok), float(motor_ok), float(audio_ok)],
            dtype=np.float32,
        )

        # Store in ring buffers.
        self._vision_buf.append(vision_result)
        self._distance_buf.append(distance_result)
        self._motor_buf.append(motor_result)
        self._audio_buf.append(audio_result)

        _log.debug(
            "read_all_complete",
            elapsed_ms=(time.monotonic() - t0) * MILLISECONDS_PER_SECOND,
            valid_sensors=int(vision_ok) + int(distance_ok) + int(motor_ok) + int(audio_ok),
        )

        return MouseDroidObservationBundle(
            _timestamp=timestamp,
            _vision_features=vision_result,
            _distance_m=distance_result,
            _motor_state=motor_result,
            _audio_chunk=audio_result,
            _valid_mask=valid_mask,
        )

    # -- Private helpers ---------------------------------------------------

    @staticmethod
    async def _safe_read(
        coro: asyncio.coroutines,
        sensor_name: str,
        default: NDArray[np.float32] | float,
    ) -> tuple[NDArray[np.float32] | float, bool]:
        """Generic safe sensor read with fallback.

        Args:
            coro: Awaitable that performs the sensor read.
            sensor_name: Human-readable sensor name for logging.
            default: Fallback value returned on failure.

        Returns:
            Tuple of (result, success_flag).
        """
        try:
            result = await coro
            return result, True
        except Exception:
            _log.warning(f"{sensor_name}_read_failed", exc_info=True)
            return default, False

    async def _safe_vision_read(self) -> tuple[NDArray[np.float32], bool]:
        """Attempt a vision capture, returning zeros on failure."""
        default = np.zeros(self._cfg.camera.feature_dim, dtype=np.float32)
        result, ok = await self._safe_read(
            self._vision.capture_features(), "vision", default,
        )
        return result, ok  # type: ignore[return-value]

    async def _safe_distance_read(self) -> tuple[float, bool]:
        """Attempt a distance read, returning max range on failure."""
        result, ok = await self._safe_read(
            self._distance.read_distance_m(), "distance", self._distance.max_range_m,
        )
        return result, ok  # type: ignore[return-value]

    async def _safe_motor_read(self) -> tuple[NDArray[np.float32], bool]:
        """Attempt an ESP32 motor/battery read, returning zeros on failure."""
        default = np.zeros(DEFAULT_MOTOR_STATE_DIM, dtype=np.float32)
        try:
            encoders, battery_v = await asyncio.gather(
                self._esp32.read_encoders(),
                self._esp32.get_battery_voltage(),
            )
            motor_state = np.array(
                [
                    encoders.left_velocity_mps,
                    encoders.right_velocity_mps,
                    encoders.heading_rad,
                    battery_v,
                ],
                dtype=np.float32,
            )
            return motor_state, True
        except Exception:
            _log.warning("motor_read_failed", exc_info=True)
            return default, False

    async def _safe_audio_read(self) -> tuple[NDArray[np.float32], bool]:
        """Attempt an audio chunk read, returning zeros on failure."""
        default = np.zeros(self._audio_chunk_size, dtype=np.float32)
        if self._microphone is None:
            return default, False

        result, ok = await self._safe_read(
            self._microphone.read_chunk(), "audio", default,
        )
        return result, ok  # type: ignore[return-value]
