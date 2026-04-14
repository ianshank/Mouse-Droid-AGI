"""Tests for LD19 binary frame protocol parser — CRC, parsing, interpolation."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from mousedroid.constants import (
    LIDAR_ANGLE_SCALE,
    LIDAR_FRAME_SIZE,
    LIDAR_HEADER_BYTE,
    LIDAR_POINTS_PER_FRAME,
    LIDAR_VER_LEN_BYTE,
)
from mousedroid.hardware.lidar.ld19_protocol import LD19FrameParser


def _build_frame(
    *,
    speed_raw: int = 500,
    start_angle_raw: int = 0,
    end_angle_raw: int = 1000,
    points: list[tuple[int, int]] | None = None,
    timestamp_ms: int = 1234,
    corrupt_crc: bool = False,
    wrong_header: bool = False,
) -> bytes:
    """Build a valid 47-byte LD19 frame for testing.

    Args:
        speed_raw: Raw speed value (multiply by LIDAR_ANGLE_SCALE for deg/s).
        start_angle_raw: Raw start angle value.
        end_angle_raw: Raw end angle value.
        points: List of (distance_mm, confidence) tuples; padded to 12 if short.
        timestamp_ms: Timestamp in milliseconds.
        corrupt_crc: If True, write an invalid CRC byte.
        wrong_header: If True, use 0x00 instead of 0x54 for the header.

    Returns:
        47-byte frame.
    """
    if points is None:
        points = [(1000, 200)] * LIDAR_POINTS_PER_FRAME
    # Pad or truncate to exactly 12 points.
    while len(points) < LIDAR_POINTS_PER_FRAME:
        points.append((0, 0))
    points = points[:LIDAR_POINTS_PER_FRAME]

    header = 0x00 if wrong_header else LIDAR_HEADER_BYTE
    buf = bytearray()
    buf.append(header)
    buf.append(LIDAR_VER_LEN_BYTE)
    buf += struct.pack("<H", speed_raw)
    buf += struct.pack("<H", start_angle_raw)

    for dist_mm, confidence in points:
        buf += struct.pack("<H", dist_mm)
        buf.append(confidence)

    buf += struct.pack("<H", end_angle_raw)
    buf += struct.pack("<H", timestamp_ms)

    assert len(buf) == LIDAR_FRAME_SIZE - 1  # 46 bytes before CRC

    crc = LD19FrameParser.crc8(bytes(buf))
    if corrupt_crc:
        crc = (crc + 1) & 0xFF
    buf.append(crc)

    assert len(buf) == LIDAR_FRAME_SIZE
    return bytes(buf)


# -- CRC8 -----------------------------------------------------------------


def test_crc8_empty_data() -> None:
    """CRC8 of empty bytes is 0."""
    assert LD19FrameParser.crc8(b"") == 0


def test_crc8_deterministic() -> None:
    """Same input always produces the same CRC."""
    data = b"\x54\x2c\x00\x01\x02\x03"
    assert LD19FrameParser.crc8(data) == LD19FrameParser.crc8(data)


def test_crc8_known_vectors() -> None:
    """CRC8 produces expected values for known test vectors."""
    assert LD19FrameParser.crc8(b"\x00\x01\x02") == 0x74
    assert LD19FrameParser.crc8(b"\x03\x04\x05") == 0x4C


def test_crc8_range() -> None:
    """CRC8 is always in [0, 255]."""
    for val in [b"\xff" * 10, b"\x00" * 10, b"\xab\xcd\xef"]:
        crc = LD19FrameParser.crc8(val)
        assert 0 <= crc <= 255


# -- parse_frame -----------------------------------------------------------


def test_parse_valid_frame() -> None:
    """A well-formed 47-byte frame parses successfully."""
    frame_bytes = _build_frame(
        speed_raw=500,
        start_angle_raw=10000,  # 100.00 degrees
        end_angle_raw=11000,  # 110.00 degrees
        points=[(2000, 150)] * LIDAR_POINTS_PER_FRAME,
        timestamp_ms=5678,
    )
    frame = LD19FrameParser.parse_frame(frame_bytes)
    assert frame is not None
    assert frame.speed_deg_s == pytest.approx(500 * LIDAR_ANGLE_SCALE)
    assert frame.start_angle_deg == pytest.approx(100.0)
    assert frame.end_angle_deg == pytest.approx(110.0)
    assert frame.timestamp_ms == 5678
    assert len(frame.points) == LIDAR_POINTS_PER_FRAME
    for pt in frame.points:
        assert pt.distance_mm == 2000
        assert pt.confidence == 150


def test_parse_frame_invalid_crc_returns_none() -> None:
    """A frame with incorrect CRC returns None."""
    frame_bytes = _build_frame(corrupt_crc=True)
    assert LD19FrameParser.parse_frame(frame_bytes) is None


def test_parse_frame_short_data_returns_none() -> None:
    """Data shorter than LIDAR_FRAME_SIZE returns None."""
    assert LD19FrameParser.parse_frame(b"\x54\x2c") is None
    assert LD19FrameParser.parse_frame(b"") is None
    assert LD19FrameParser.parse_frame(b"\x00" * 10) is None


def test_parse_frame_wrong_header_returns_none() -> None:
    """A frame with wrong header byte returns None."""
    frame_bytes = _build_frame(wrong_header=True)
    assert LD19FrameParser.parse_frame(frame_bytes) is None


def test_parse_frame_various_point_values() -> None:
    """Points with varying distance and confidence are parsed correctly."""
    points = [(i * 100, i * 20) for i in range(1, LIDAR_POINTS_PER_FRAME + 1)]
    frame_bytes = _build_frame(points=points)
    frame = LD19FrameParser.parse_frame(frame_bytes)
    assert frame is not None
    for i, pt in enumerate(frame.points):
        assert pt.distance_mm == (i + 1) * 100
        assert pt.confidence == (i + 1) * 20


def test_parse_frame_zero_speed_and_angles() -> None:
    """Frame with zeroed header fields parses correctly."""
    frame_bytes = _build_frame(
        speed_raw=0,
        start_angle_raw=0,
        end_angle_raw=0,
        timestamp_ms=0,
    )
    frame = LD19FrameParser.parse_frame(frame_bytes)
    assert frame is not None
    assert frame.speed_deg_s == 0.0
    assert frame.start_angle_deg == 0.0
    assert frame.end_angle_deg == 0.0
    assert frame.timestamp_ms == 0


# -- interpolate_angles ----------------------------------------------------


def test_interpolate_uniform_distribution() -> None:
    """Angles are uniformly distributed between start and end (n-1 intervals)."""
    angles = LD19FrameParser.interpolate_angles(0.0, 120.0, 4)
    assert len(angles) == 4
    # 4 points, 3 intervals over 120°: step = 40°
    expected = np.array([0.0, 40.0, 80.0, 120.0], dtype=np.float32)
    np.testing.assert_allclose(angles, expected, atol=0.01)


def test_interpolate_wraparound() -> None:
    """Handles 360-degree wraparound (e.g. 350 -> 10 spans 20 degrees)."""
    angles = LD19FrameParser.interpolate_angles(350.0, 10.0, 4)
    assert len(angles) == 4
    # Span = 20 degrees, 3 intervals: step ≈ 6.667 degrees.
    # Expected: 350, 356.667, 3.333, 10
    expected = np.array([350.0, 356.6667, 3.3333, 10.0], dtype=np.float32)
    np.testing.assert_allclose(angles, expected, atol=0.01)


def test_interpolate_zero_points() -> None:
    """Zero points returns an empty array."""
    angles = LD19FrameParser.interpolate_angles(0.0, 10.0, 0)
    assert len(angles) == 0


def test_interpolate_single_point() -> None:
    """Single point returns just the start angle."""
    angles = LD19FrameParser.interpolate_angles(45.0, 90.0, 1)
    assert len(angles) == 1
    assert angles[0] == pytest.approx(45.0)


def test_interpolate_same_start_end() -> None:
    """When start == end, all points are at the start angle."""
    angles = LD19FrameParser.interpolate_angles(90.0, 90.0, 5)
    assert len(angles) == 5
    for a in angles:
        assert a == pytest.approx(90.0, abs=0.01)


def test_interpolate_dtype_is_float32() -> None:
    """Interpolated angles have float32 dtype."""
    angles = LD19FrameParser.interpolate_angles(0.0, 180.0, 10)
    assert angles.dtype == np.float32


def test_interpolate_values_in_range() -> None:
    """All interpolated angles are in [0, 360)."""
    angles = LD19FrameParser.interpolate_angles(340.0, 20.0, 12)
    assert np.all(angles >= 0.0)
    assert np.all(angles < 360.0)
