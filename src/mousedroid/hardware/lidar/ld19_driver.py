"""FHL-LD19 LiDAR hardware driver.

Implements :class:`~mousedroid.hardware.protocols.LidarProtocol` using
``pyserial`` for UART communication.  All blocking serial I/O is delegated
to :func:`asyncio.to_thread` to keep the event loop responsive.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mousedroid.constants import (
    LIDAR_FRAME_SIZE,
    LIDAR_HEADER_BYTE,
    LIDAR_MM_PER_M,
    LIDAR_SCAN_TIMEOUT_MULTIPLIER,
    LIDAR_VER_LEN_BYTE,
)
from mousedroid.hardware.lidar.ld19_protocol import LD19Frame, LD19FrameParser
from mousedroid.logging.setup import get_logger
from mousedroid.sensing.lidar_scan import LidarScan, empty_scan

if TYPE_CHECKING:
    from mousedroid.config.schema import LidarConfig

_log = get_logger(__name__)


@dataclass
class LD19ReadStats:
    """Low-level serial and parser statistics for one LD19 scan read."""

    bytes_read: int = 0
    chunks_read: int = 0
    empty_reads: int = 0
    prefix_hits: int = 0
    header_search_misses: int = 0
    bytes_discarded: int = 0
    parse_failures: int = 0
    crc_failures: int = 0
    frames_parsed: int = 0
    covered_angle_deg: float = 0.0
    elapsed_s: float = 0.0


class LD19LidarDriver:
    """FHL-LD19 2D LiDAR driver implementing LidarProtocol.

    Args:
        cfg: LiDAR configuration with serial port, baud rate, and thresholds.
    """

    def __init__(self, cfg: LidarConfig) -> None:
        self._cfg = cfg
        self._serial: Any = None
        self._running = False

    # -- LidarProtocol properties -----------------------------------------

    @property
    def max_range_m(self) -> float:
        """Maximum detection range in metres."""
        return self._cfg.max_range_m

    @property
    def min_range_m(self) -> float:
        """Minimum detection range in metres."""
        return self._cfg.min_range_m

    @property
    def scan_frequency_hz(self) -> float:
        """Nominal scan rotation frequency in Hz."""
        return self._cfg.scan_frequency_hz

    # -- Lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Open the serial port and begin data acquisition."""
        self._serial = await asyncio.to_thread(self._open_serial)
        self._running = True
        _log.info(
            "ld19_started",
            serial_port=self._cfg.serial_port,
            baud_rate=self._cfg.baud_rate,
        )

    async def stop(self) -> None:
        """Close the serial port and stop data acquisition."""
        self._running = False
        if self._serial is not None:
            await asyncio.to_thread(self._close_serial)
            self._serial = None
        _log.info("ld19_stopped")

    # -- Data acquisition -------------------------------------------------

    async def read_scan(self) -> LidarScan:
        """Read a full 360-degree scan.

        Delegates blocking serial reads to a thread pool.

        Returns:
            A :class:`LidarScan` with sorted, filtered measurements.
        """
        if self._serial is None:
            _log.warning("ld19_read_scan_no_serial")
            return empty_scan()

        frames = await asyncio.to_thread(self._read_frames_blocking)
        if not frames:
            _log.warning("ld19_no_frames_received")
            return empty_scan()

        return self._assemble_scan(frames, self._cfg)

    async def read_scan_with_diagnostics(self) -> tuple[LidarScan, LD19ReadStats]:
        """Read a scan together with low-level serial and parser statistics."""
        if self._serial is None:
            _log.warning("ld19_read_scan_no_serial")
            return empty_scan(), LD19ReadStats()

        frames, stats = await asyncio.to_thread(self._read_frames_with_stats_blocking)
        if not frames:
            _log.warning("ld19_no_frames_received")
            return empty_scan(), stats

        return self._assemble_scan(frames, self._cfg), stats

    # -- Private helpers --------------------------------------------------

    def _open_serial(self) -> object:
        """Open the serial port (blocking)."""
        import serial as pyserial

        return pyserial.Serial(
            port=self._cfg.serial_port,
            baudrate=self._cfg.baud_rate,
            timeout=self._cfg.read_timeout_s,
        )

    def _close_serial(self) -> None:
        """Close the serial port (blocking)."""
        if self._serial is not None and hasattr(self._serial, "close"):
            self._serial.close()

    def _read_frames_blocking(self) -> list[LD19Frame]:
        """Read serial data until a full 360-degree rotation is captured.

        Scans the byte stream for header bytes, validates each frame, and
        accumulates frames until the angle wraps past 360 degrees.

        Returns:
            List of parsed :class:`LD19Frame` objects forming one rotation.
        """
        frames, _ = self._read_frames_with_stats_blocking()
        return frames

    def _read_frames_with_stats_blocking(self) -> tuple[list[LD19Frame], LD19ReadStats]:
        """Read serial data until a full rotation is captured, collecting stats."""
        ser = self._serial
        if ser is None:
            return [], LD19ReadStats()

        frames: list[LD19Frame] = []
        buf = bytearray()
        stats = LD19ReadStats()
        frame_prefix = bytes((LIDAR_HEADER_BYTE, LIDAR_VER_LEN_BYTE))
        start_time = time.monotonic()
        deadline = time.monotonic() + max(
            self._cfg.scan_acquisition_timeout_s,
            LIDAR_SCAN_TIMEOUT_MULTIPLIER / max(self._cfg.scan_frequency_hz, 0.1),
        )
        prev_start_angle: float | None = None
        covered_angle_deg = 0.0

        while time.monotonic() < deadline:
            chunk = ser.read(LIDAR_FRAME_SIZE * 4)
            if not chunk:
                stats.empty_reads += 1
                continue
            stats.chunks_read += 1
            stats.bytes_read += len(chunk)
            buf.extend(chunk)

            while len(buf) >= LIDAR_FRAME_SIZE:
                idx = buf.find(frame_prefix)
                if idx < 0:
                    stats.header_search_misses += 1
                    if buf and buf[-1] == LIDAR_HEADER_BYTE:
                        stats.bytes_discarded += max(0, len(buf) - 1)
                        del buf[:-1]
                    else:
                        stats.bytes_discarded += len(buf)
                        buf.clear()
                    break
                if idx > 0:
                    stats.bytes_discarded += idx
                    del buf[:idx]
                if len(buf) < LIDAR_FRAME_SIZE:
                    break

                frame_bytes = bytes(buf[:LIDAR_FRAME_SIZE])
                stats.prefix_hits += 1
                frame, failure_reason = LD19FrameParser.parse_frame_with_reason(frame_bytes)
                if frame is None:
                    stats.parse_failures += 1
                    if failure_reason == "crc":
                        stats.crc_failures += 1
                    stats.bytes_discarded += 1
                    del buf[0]
                    continue

                del buf[:LIDAR_FRAME_SIZE]
                stats.frames_parsed += 1

                if prev_start_angle is not None:
                    delta_deg = frame.start_angle_deg - prev_start_angle
                    if delta_deg < 0:
                        delta_deg += 360.0
                    covered_angle_deg += delta_deg

                prev_start_angle = frame.start_angle_deg
                frames.append(frame)

                if covered_angle_deg >= self._cfg.min_scan_coverage_deg and len(frames) > 1:
                    stats.covered_angle_deg = covered_angle_deg
                    stats.elapsed_s = time.monotonic() - start_time
                    return frames, stats

        stats.covered_angle_deg = covered_angle_deg
        stats.elapsed_s = time.monotonic() - start_time
        return frames, stats

    @staticmethod
    def _assemble_scan(frames: list[LD19Frame], cfg: LidarConfig) -> LidarScan:
        """Convert accumulated frames into a sorted, filtered LidarScan.

        Args:
            frames: Parsed frames from one rotation.
            cfg: LiDAR configuration for filtering thresholds.

        Returns:
            A :class:`LidarScan` with deduplicated, sorted measurements.
        """
        all_angles: list[float] = []
        all_distances: list[float] = []
        all_confidences: list[int] = []

        for frame in frames:
            angles = LD19FrameParser.interpolate_angles(
                frame.start_angle_deg,
                frame.end_angle_deg,
                len(frame.points),
            )
            for i, point in enumerate(frame.points):
                if point.confidence < cfg.min_confidence:
                    continue
                dist_m = point.distance_mm / LIDAR_MM_PER_M
                if dist_m < cfg.min_range_m or dist_m > cfg.max_range_m:
                    continue
                all_angles.append(float(angles[i]))
                all_distances.append(point.distance_mm)
                all_confidences.append(point.confidence)

        if not all_angles:
            return empty_scan()

        # Sort by angle for consistent scan ordering.
        order = np.argsort(np.array(all_angles, dtype=np.float32))
        angles_arr = np.array(all_angles, dtype=np.float32)[order]
        distances_arr = np.array(all_distances, dtype=np.float32)[order]
        confidences_arr = np.array(all_confidences, dtype=np.uint8)[order]

        _log.debug(
            "ld19_scan_assembled",
            n_points=len(angles_arr),
            n_frames=len(frames),
        )

        return LidarScan(
            angles_deg=angles_arr,
            distances_mm=distances_arr,
            confidences=confidences_arr,
            timestamp=time.monotonic(),
            n_points=len(angles_arr),
        )
