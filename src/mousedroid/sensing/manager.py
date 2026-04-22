"""Sensor manager — orchestrates all sensor reads into an observation bundle.

Reads vision, ultrasonic, ESP32 motor, and audio data concurrently, handles
failures gracefully, and maintains per-sensor ring buffers.

Degraded motor reads:
    When the ESP32 is unresponsive (consecutive failures exceed
    ``esp32.max_consecutive_timeouts``), motor reads switch to a cached
    last-known value at reduced poll frequency (``esp32.degraded_poll_interval_s``)
    to avoid blocking the orchestrator loop.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable
from typing import TYPE_CHECKING, TypeVar

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
    from mousedroid.hardware.audio.feature_extractor import AudioFeatureExtractor
    from mousedroid.hardware.lidar.feature_extractor import LidarFeatureExtractor
    from mousedroid.hardware.protocols import (
        AudioProtocol,
        DistanceSensorProtocol,
        LidarProtocol,
        VisionProtocol,
    )

T = TypeVar("T")

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
        audio_feature_extractor: Optional audio feature extractor for mel features.
        lidar: Optional LiDAR driver implementing :class:`LidarProtocol`.
        lidar_feature_extractor: Optional LiDAR feature extractor.
    """

    def __init__(
        self,
        vision: VisionProtocol | None,
        distance: DistanceSensorProtocol | None,
        esp32: ESP32CommProtocol,
        cfg: Settings,
        microphone: AudioProtocol | None = None,
        audio_feature_extractor: AudioFeatureExtractor | None = None,
        lidar: LidarProtocol | None = None,
        lidar_feature_extractor: LidarFeatureExtractor | None = None,
    ) -> None:
        self._vision = vision
        self._distance = distance
        self._esp32 = esp32
        self._cfg = cfg
        self._microphone = microphone
        self._audio_feature_extractor = audio_feature_extractor
        self._lidar = lidar
        self._lidar_feature_extractor = lidar_feature_extractor

        # Ring buffer sizes derived from config loop rates.
        vision_buf_size = max(1, int(cfg.loop.perception_hz))
        ultrasonic_buf_size = max(1, int(cfg.loop.ultrasonic_hz))
        motor_buf_size = max(1, int(cfg.loop.control_hz))
        audio_buf_size = max(1, int(cfg.loop.audio_hz))

        self._vision_buf: deque[NDArray[np.float32]] = deque(maxlen=vision_buf_size)
        self._distance_buf: deque[float] = deque(maxlen=ultrasonic_buf_size)
        self._motor_buf: deque[NDArray[np.float32]] = deque(maxlen=motor_buf_size)
        self._audio_buf: deque[NDArray[np.float32]] = deque(maxlen=audio_buf_size)

        # LiDAR ring buffer (only allocated when lidar is present).
        if lidar is not None:
            lidar_buf_size = max(1, int(cfg.loop.lidar_hz))
            self._lidar_buf: deque[NDArray[np.float32] | None] = deque(maxlen=lidar_buf_size)
        else:
            self._lidar_buf = deque(maxlen=1)

        # Determine lidar feature size for zero-fill on failure.
        if lidar_feature_extractor is not None:
            self._lidar_feature_dim = lidar_feature_extractor.feature_dim
        elif cfg.lidar is not None:
            self._lidar_feature_dim = cfg.lidar.feature_dim
        else:
            self._lidar_feature_dim = 0

        # Determine audio output size for zero-fill on failure.
        # When a feature extractor is present the output dimension changes.
        if audio_feature_extractor is not None:
            self._audio_chunk_size = audio_feature_extractor.feature_dim
        elif microphone is not None:
            self._audio_chunk_size = microphone.chunk_size * microphone.channels
        else:
            self._audio_chunk_size = DEFAULT_AUDIO_CHUNK_SIZE

        # Motor degraded-polling state: cache last-known motor reading and
        # skip real reads when ESP32 is unresponsive.
        self._motor_consecutive_failures: int = 0
        self._motor_max_failures: int = cfg.esp32.max_consecutive_timeouts
        self._motor_degraded_interval: float = cfg.esp32.degraded_poll_interval_s
        self._motor_last_probe: float = 0.0
        self._motor_degraded: bool = False
        self._distance_fallback_m: float = cfg.safety.distance_fallback_m
        self._cached_motor_state: NDArray[np.float32] = np.zeros(
            DEFAULT_MOTOR_STATE_DIM,
            dtype=np.float32,
        )

        _log.info(
            "sensor_manager_init",
            vision_buf=vision_buf_size,
            ultrasonic_buf=ultrasonic_buf_size,
            motor_buf=motor_buf_size,
            audio_buf=audio_buf_size,
            microphone_enabled=microphone is not None,
            lidar_enabled=lidar is not None,
        )

    # -- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Start all sensor hardware."""
        if self._vision is not None:
            try:
                await self._vision.start()
            except Exception as exc:
                _log.warning("vision_start_failed_degrading", error=str(exc))
                self._vision = None
        if self._microphone is not None:
            try:
                await self._microphone.start()
            except Exception as exc:
                _log.warning("microphone_start_failed_degrading", error=str(exc))
                self._microphone = None
        if self._lidar is not None:
            try:
                await self._lidar.start()
            except Exception as exc:
                _log.warning("lidar_start_failed_degrading", error=str(exc))
                self._lidar = None

    async def stop(self) -> None:
        """Stop all sensor hardware."""
        if self._vision is not None:
            await self._vision.stop()
        if self._microphone is not None:
            await self._microphone.stop()
        if self._lidar is not None:
            await self._lidar.stop()

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
        lidar_task = asyncio.create_task(self._safe_lidar_read())

        vision_result, vision_ok = await vision_task
        distance_result, distance_ok = await distance_task
        motor_result, motor_ok = await motor_task
        audio_result, audio_ok = await audio_task
        lidar_result, lidar_ok, lidar_n_points = await lidar_task

        # Build validity mask: vision=0, ultrasonic=1, motor=2, audio=3,
        # lidar=4 (only when lidar is configured).
        if self._lidar is not None:
            valid_mask = np.array(
                [
                    float(vision_ok),
                    float(distance_ok),
                    float(motor_ok),
                    float(audio_ok),
                    float(lidar_ok),
                ],
                dtype=np.float32,
            )
        else:
            valid_mask = np.array(
                [float(vision_ok), float(distance_ok), float(motor_ok), float(audio_ok)],
                dtype=np.float32,
            )

        # Store in ring buffers.
        self._vision_buf.append(vision_result)
        self._distance_buf.append(distance_result)
        self._motor_buf.append(motor_result)
        self._audio_buf.append(audio_result)
        if self._lidar is not None:
            self._lidar_buf.append(lidar_result)

        valid_count = int(vision_ok) + int(distance_ok) + int(motor_ok) + int(audio_ok)
        if self._lidar is not None:
            valid_count += int(lidar_ok)

        _log.debug(
            "read_all_complete",
            elapsed_ms=(time.monotonic() - t0) * MILLISECONDS_PER_SECOND,
            valid_sensors=valid_count,
        )

        return MouseDroidObservationBundle(
            _timestamp=timestamp,
            _vision_features=vision_result,
            _distance_m=distance_result,
            _motor_state=motor_result,
            _audio_chunk=audio_result,
            _lidar_features=lidar_result if self._lidar is not None else None,
            _lidar_n_points=lidar_n_points if self._lidar is not None else 0,
            _valid_mask=valid_mask,
        )

    # -- Private helpers ---------------------------------------------------

    @staticmethod
    async def _safe_read(
        coro: Awaitable[T],
        sensor_name: str,
        default: T,
    ) -> tuple[T, bool]:
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
        if self._vision is None:
            return default, False
        result, ok = await self._safe_read(
            self._vision.capture_features(),
            "vision",
            default,
        )
        return result, ok

    async def _safe_distance_read(self) -> tuple[float, bool]:
        """Attempt a distance read, returning max range on failure."""
        if self._distance is None:
            return self._distance_fallback_m, False
        result, ok = await self._safe_read(
            self._distance.read_distance_m(),
            "distance",
            self._distance.max_range_m,
        )
        return result, ok

    async def _safe_motor_read(self) -> tuple[NDArray[np.float32], bool]:
        """Attempt an ESP32 motor/battery read, returning zeros on failure.

        When the ESP32 is unresponsive (>= ``_motor_max_failures`` consecutive
        failures), switches to degraded mode: returns the cached last-known
        motor state and only probes the real device every
        ``_motor_degraded_interval`` seconds.
        """
        default = np.zeros(DEFAULT_MOTOR_STATE_DIM, dtype=np.float32)

        # In degraded mode, return cached value unless probe interval has elapsed.
        if self._motor_degraded:
            now = time.monotonic()
            if (now - self._motor_last_probe) < self._motor_degraded_interval:
                _log.debug("motor_read_skipped_degraded")
                return self._cached_motor_state.copy(), False
            # Probe interval elapsed — attempt a real read below.
            self._motor_last_probe = now

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
            # Successful read — cache and exit degraded mode.
            self._cached_motor_state = motor_state.copy()
            if self._motor_degraded:
                self._motor_degraded = False
                self._motor_consecutive_failures = 0
                _log.info("motor_recovered_from_degraded")
            else:
                self._motor_consecutive_failures = 0
            return motor_state, True
        except Exception:
            _log.warning("motor_read_failed", exc_info=True)
            self._motor_consecutive_failures += 1
            if (
                not self._motor_degraded
                and self._motor_consecutive_failures >= self._motor_max_failures
            ):
                self._motor_degraded = True
                self._motor_last_probe = time.monotonic()
                _log.warning(
                    "motor_entering_degraded_mode",
                    consecutive_failures=self._motor_consecutive_failures,
                    poll_interval_s=self._motor_degraded_interval,
                )
            return default, False

    async def _safe_audio_read(self) -> tuple[NDArray[np.float32], bool]:
        """Attempt an audio chunk read with optional feature extraction.

        When an :class:`AudioFeatureExtractor` is configured, raw audio is
        transformed into mel-spectrogram features before being returned.
        Otherwise the raw audio chunk is returned directly.
        """
        default = np.zeros(self._audio_chunk_size, dtype=np.float32)
        if self._microphone is None:
            return default, False

        result, ok = await self._safe_read(
            self._microphone.read_chunk(),
            "audio",
            default,
        )
        if ok and self._audio_feature_extractor is not None:
            try:
                result = self._audio_feature_extractor.extract(result)
            except Exception:
                _log.warning("audio_feature_extraction_failed", exc_info=True)
                return np.zeros(self._audio_chunk_size, dtype=np.float32), False
        return result, ok

    async def _safe_lidar_read(
        self,
    ) -> tuple[NDArray[np.float32] | None, bool, int]:
        """Attempt a LiDAR scan read with optional feature extraction.

        When a :class:`LidarFeatureExtractor` is configured, raw scans are
        transformed into sector-binned distance features.  Returns
        ``(features, ok, n_points)`` where ``n_points`` is the number of
        raw points in the scan (``0`` when no scan).  Features are
        ``None`` when LiDAR is not configured.
        """
        if self._lidar is None:
            return None, False, 0

        default: NDArray[np.float32] | None = None
        if self._lidar_feature_dim > 0:
            default = np.ones(self._lidar_feature_dim, dtype=np.float32)

        try:
            scan = await self._lidar.read_scan()
            n_points = int(getattr(scan, "n_points", 0))
            if self._lidar_feature_extractor is not None:
                features = self._lidar_feature_extractor.extract(scan)
                return features, True, n_points
            return default, False, n_points
        except Exception:
            _log.warning("lidar_read_failed", exc_info=True)
            return default, False, 0
