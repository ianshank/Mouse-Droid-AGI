"""Sensor manager — orchestrates all sensor reads into an observation bundle.

Reads vision, ultrasonic, and ESP32 motor data concurrently, handles
failures gracefully, and maintains per-sensor ring buffers.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from mousedroid.logging.setup import get_logger
from mousedroid.sensing.bundle import MouseDroidObservationBundle

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mousedroid.comms.protocol import ESP32CommProtocol
    from mousedroid.config.schema import Settings
    from mousedroid.hardware.protocols import DistanceSensorProtocol, VisionProtocol

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
    """

    def __init__(
        self,
        vision: VisionProtocol,
        distance: DistanceSensorProtocol,
        esp32: ESP32CommProtocol,
        cfg: Settings,
    ) -> None:
        self._vision = vision
        self._distance = distance
        self._esp32 = esp32
        self._cfg = cfg

        # Ring buffer sizes derived from config loop rates.
        vision_buf_size = max(1, int(cfg.loop.perception_hz))
        ultrasonic_buf_size = max(1, int(cfg.loop.ultrasonic_hz))
        motor_buf_size = max(1, int(cfg.loop.control_hz))

        self._vision_buf: deque[NDArray[np.float32]] = deque(maxlen=vision_buf_size)
        self._distance_buf: deque[float] = deque(maxlen=ultrasonic_buf_size)
        self._motor_buf: deque[NDArray[np.float32]] = deque(maxlen=motor_buf_size)

        _log.info(
            "sensor_manager_init",
            vision_buf=vision_buf_size,
            ultrasonic_buf=ultrasonic_buf_size,
            motor_buf=motor_buf_size,
        )

    # -- Public API --------------------------------------------------------

    async def read_all(self) -> MouseDroidObservationBundle:
        """Read all sensors concurrently and return a fused observation bundle.

        Sensors that fail are represented by zeroed-out data with the
        corresponding :pyattr:`valid_mask` slot set to ``0.0``.

        Returns:
            A fully-populated :class:`MouseDroidObservationBundle`.
        """
        timestamp = time.monotonic()

        # Kick off all reads concurrently.
        vision_task = asyncio.create_task(self._safe_vision_read())
        distance_task = asyncio.create_task(self._safe_distance_read())
        motor_task = asyncio.create_task(self._safe_motor_read())

        vision_result, vision_ok = await vision_task
        distance_result, distance_ok = await distance_task
        motor_result, motor_ok = await motor_task

        # Build validity mask: vision=0, ultrasonic=1, motor=2.
        valid_mask = np.array(
            [float(vision_ok), float(distance_ok), float(motor_ok)],
            dtype=np.float32,
        )

        # Store in ring buffers.
        self._vision_buf.append(vision_result)
        self._distance_buf.append(distance_result)
        self._motor_buf.append(motor_result)

        return MouseDroidObservationBundle(
            _timestamp=timestamp,
            _vision_features=vision_result,
            _distance_m=distance_result,
            _motor_state=motor_result,
            _valid_mask=valid_mask,
        )

    # -- Private helpers ---------------------------------------------------

    async def _safe_vision_read(self) -> tuple[NDArray[np.float32], bool]:
        """Attempt a vision capture, returning zeros on failure.

        Returns:
            Tuple of (feature_vector, success_flag).
        """
        try:
            features = await self._vision.capture_features()
            return features, True
        except Exception:
            _log.warning("vision_read_failed", exc_info=True)
            return np.zeros(self._cfg.camera.feature_dim, dtype=np.float32), False

    async def _safe_distance_read(self) -> tuple[float, bool]:
        """Attempt a distance read, returning max range on failure.

        Returns:
            Tuple of (distance_m, success_flag).
        """
        try:
            distance = await self._distance.read_distance_m()
            return distance, True
        except Exception:
            _log.warning("distance_read_failed", exc_info=True)
            return self._distance.max_range_m, False

    async def _safe_motor_read(self) -> tuple[NDArray[np.float32], bool]:
        """Attempt an ESP32 motor/battery read, returning zeros on failure.

        Returns:
            Tuple of (motor_state_array, success_flag).
        """
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
            return np.zeros(4, dtype=np.float32), False
