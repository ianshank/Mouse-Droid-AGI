"""FHL-LD19 binary frame protocol parser.

Pure-function parser with no I/O dependencies, making it highly testable.
Handles CRC8 validation, frame extraction, and angle interpolation.

Frame format (47 bytes total):
    ``0x54 | 0x2C | speed(2B LE) | start_angle(2B LE) |
    12 x (distance_mm(2B LE) + confidence(1B)) |
    end_angle(2B LE) | timestamp(2B LE) | crc8(1B)``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mousedroid.constants import (
    LIDAR_ANGLE_SCALE,
    LIDAR_CRC8_POLYNOMIAL,
    LIDAR_FRAME_SIZE,
    LIDAR_FULL_ROTATION_DEG,
    LIDAR_HEADER_BYTE,
    LIDAR_POINTS_PER_FRAME,
    LIDAR_VER_LEN_BYTE,
)


def _build_crc_table() -> tuple[int, ...]:
    """Build the CRC8 lookup table for the LD19 polynomial."""
    table: list[int] = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc << 1) ^ LIDAR_CRC8_POLYNOMIAL) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table.append(crc)
    return tuple(table)


CRC_TABLE: tuple[int, ...] = _build_crc_table()
"""Pre-computed CRC8 lookup table."""


@dataclass(frozen=True)
class LD19Point:
    """Single measurement point from an LD19 frame."""

    distance_mm: int
    confidence: int


@dataclass(frozen=True)
class LD19Frame:
    """Parsed data from a single LD19 data frame (12 measurement points)."""

    speed_deg_s: float
    start_angle_deg: float
    end_angle_deg: float
    points: tuple[LD19Point, ...]
    timestamp_ms: int


class LD19FrameParser:
    """Parse FHL-LD19 binary frames from a byte stream.

    All methods are static — no mutable state.
    """

    @staticmethod
    def crc8(data: bytes) -> int:
        """Compute CRC8 checksum over *data* using the LD19 polynomial.

        Args:
            data: Input bytes to checksum.

        Returns:
            CRC8 value in ``[0, 255]``.
        """
        crc = 0
        for byte in data:
            crc = CRC_TABLE[(crc ^ byte) & 0xFF]
        return crc

    @staticmethod
    def parse_frame(data: bytes) -> LD19Frame | None:
        """Parse a single 47-byte LD19 frame.

        Args:
            data: Exactly :data:`LIDAR_FRAME_SIZE` bytes starting at the
                header byte.

        Returns:
            Parsed :class:`LD19Frame`, or ``None`` if the data is invalid
            (wrong header, wrong length, or CRC mismatch).
        """
        frame, _ = LD19FrameParser.parse_frame_with_reason(data)
        return frame

    @staticmethod
    def parse_frame_with_reason(data: bytes) -> tuple[LD19Frame | None, str | None]:
        """Parse a single 47-byte LD19 frame with failure classification.

        Args:
            data: Exactly :data:`LIDAR_FRAME_SIZE` bytes starting at the
                header byte.

        Returns:
            Tuple of ``(frame, failure_reason)`` where ``failure_reason`` is
            ``None`` on success or one of ``length``, ``header``, or ``crc``.
        """
        if len(data) != LIDAR_FRAME_SIZE:
            return None, "length"

        if data[0] != LIDAR_HEADER_BYTE or data[1] != LIDAR_VER_LEN_BYTE:
            return None, "header"

        # Validate CRC8 (over all bytes except the last one).
        expected_crc = data[-1]
        computed_crc = LD19FrameParser.crc8(data[:-1])
        if computed_crc != expected_crc:
            return None, "crc"

        # Parse header fields (little-endian).
        speed_raw = int.from_bytes(data[2:4], "little")
        start_angle_raw = int.from_bytes(data[4:6], "little")

        # Parse 12 measurement points: each is 3 bytes (dist_mm LE + confidence).
        points: list[LD19Point] = []
        offset = 6
        for _ in range(LIDAR_POINTS_PER_FRAME):
            dist_mm = int.from_bytes(data[offset : offset + 2], "little")
            confidence = data[offset + 2]
            points.append(LD19Point(distance_mm=dist_mm, confidence=confidence))
            offset += 3

        # Parse trailer fields.
        end_angle_raw = int.from_bytes(data[offset : offset + 2], "little")
        timestamp_ms = int.from_bytes(data[offset + 2 : offset + 4], "little")

        return (
            LD19Frame(
                speed_deg_s=speed_raw * LIDAR_ANGLE_SCALE,
                start_angle_deg=start_angle_raw * LIDAR_ANGLE_SCALE,
                end_angle_deg=end_angle_raw * LIDAR_ANGLE_SCALE,
                points=tuple(points),
                timestamp_ms=timestamp_ms,
            ),
            None,
        )

    @staticmethod
    def interpolate_angles(
        start_deg: float,
        end_deg: float,
        n_points: int,
    ) -> NDArray[np.float32]:
        """Linearly interpolate angles between start and end for *n_points*.

        Handles the 360-degree wraparound case (e.g., start=350, end=10
        spans 20 degrees, not 340).

        Args:
            start_deg: Start angle in degrees.
            end_deg: End angle in degrees.
            n_points: Number of points to interpolate.

        Returns:
            Array of interpolated angles in ``[0, 360)``, shape ``(n_points,)``.
        """
        if n_points <= 0:
            return np.array([], dtype=np.float32)

        # Handle wraparound: if end < start, the span crosses 0 degrees.
        diff = end_deg - start_deg
        if diff < 0:
            diff += LIDAR_FULL_ROTATION_DEG

        step = diff / (n_points - 1) if n_points > 1 else 0.0
        angles = np.array(
            [(start_deg + step * i) % LIDAR_FULL_ROTATION_DEG for i in range(n_points)],
            dtype=np.float32,
        )
        return angles
