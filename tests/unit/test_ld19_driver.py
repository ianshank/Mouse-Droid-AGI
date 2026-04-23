"""Tests for LD19LidarDriver._assemble_scan — sorting, filtering, clamping."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from mousedroid.config.schema import LidarConfig
from mousedroid.constants import LIDAR_POINTS_PER_FRAME, LIDAR_VER_LEN_BYTE
from mousedroid.hardware.lidar.ld19_driver import LD19LidarDriver
from mousedroid.hardware.lidar.ld19_protocol import LD19Frame, LD19FrameParser, LD19Point


def _cfg(**overrides: object) -> LidarConfig:
    """Create a LidarConfig with test defaults."""
    defaults: dict[str, object] = {
        "enabled": True,
        "serial_port": "/dev/ttyUSB1",
        "baud_rate": 230400,
        "max_range_m": 12.0,
        "min_range_m": 0.15,
        "scan_frequency_hz": 10.0,
        "min_confidence": 10,
        "read_timeout_s": 0.2,
        "n_sectors": 36,
        "feature_dim": 36,
    }
    defaults.update(overrides)
    return LidarConfig(**defaults)  # type: ignore[arg-type]


def _make_frame(
    start_deg: float,
    end_deg: float,
    points: list[tuple[int, int]],
) -> LD19Frame:
    """Build an LD19Frame with the given angles and points.

    Args:
        start_deg: Start angle in degrees.
        end_deg: End angle in degrees.
        points: List of (distance_mm, confidence) tuples.

    Returns:
        LD19Frame with the specified values.
    """
    return LD19Frame(
        speed_deg_s=300.0,
        start_angle_deg=start_deg,
        end_angle_deg=end_deg,
        points=tuple(LD19Point(distance_mm=d, confidence=c) for d, c in points),
        timestamp_ms=1000,
    )


def _build_frame_bytes(start_deg: float, end_deg: float) -> bytes:
    """Build a valid LD19 frame byte payload for resync tests."""
    start_raw = round(start_deg * 100.0)
    end_raw = round(end_deg * 100.0)
    buf = bytearray()
    buf.append(0x54)
    buf.append(LIDAR_VER_LEN_BYTE)
    buf += struct.pack("<H", 500)
    buf += struct.pack("<H", start_raw)

    for _ in range(LIDAR_POINTS_PER_FRAME):
        buf += struct.pack("<H", 5000)
        buf.append(200)

    buf += struct.pack("<H", end_raw)
    buf += struct.pack("<H", 1234)
    buf.append(LD19FrameParser.crc8(bytes(buf)))
    return bytes(buf)


# -- Sorts by angle ---------------------------------------------------------


def test_assemble_scan_sorts_by_angle() -> None:
    """Output scan is sorted by angle regardless of input frame order."""
    # Frame at 180-190 degrees, then frame at 10-20 degrees.
    frame_a = _make_frame(180.0, 190.0, [(5000, 200)] * 3)
    frame_b = _make_frame(10.0, 20.0, [(5000, 200)] * 3)

    cfg = _cfg()
    scan = LD19LidarDriver._assemble_scan([frame_a, frame_b], cfg)

    # All angles should be in sorted order.
    for i in range(1, scan.n_points):
        assert scan.angles_deg[i] >= scan.angles_deg[i - 1]


# -- Filters by confidence --------------------------------------------------


def test_assemble_scan_filters_low_confidence() -> None:
    """Points with confidence below min_confidence are excluded."""
    cfg = _cfg(min_confidence=50)
    # All points at confidence=10, below threshold of 50.
    frame = _make_frame(0.0, 10.0, [(5000, 10)] * 12)
    scan = LD19LidarDriver._assemble_scan([frame], cfg)
    assert scan.n_points == 0


def test_assemble_scan_keeps_high_confidence() -> None:
    """Points with confidence >= min_confidence are kept."""
    cfg = _cfg(min_confidence=50)
    frame = _make_frame(0.0, 10.0, [(5000, 100)] * 12)
    scan = LD19LidarDriver._assemble_scan([frame], cfg)
    assert scan.n_points == 12


def test_assemble_scan_mixed_confidence() -> None:
    """Only points above min_confidence threshold are kept."""
    cfg = _cfg(min_confidence=50)
    points = [(5000, 10), (5000, 100), (5000, 49), (5000, 50)]
    frame = _make_frame(0.0, 10.0, points)
    scan = LD19LidarDriver._assemble_scan([frame], cfg)
    # Confidence 100 and 50 pass; 10 and 49 are filtered.
    assert scan.n_points == 2


# -- Clamps distance to min/max range ---------------------------------------


def test_assemble_scan_clamps_below_min_range() -> None:
    """Points with distance below min_range_m are excluded."""
    cfg = _cfg(min_range_m=0.15)  # 150 mm
    # 100 mm < 150 mm min range -> excluded
    frame = _make_frame(0.0, 10.0, [(100, 200)] * 12)
    scan = LD19LidarDriver._assemble_scan([frame], cfg)
    assert scan.n_points == 0


def test_assemble_scan_clamps_above_max_range() -> None:
    """Points with distance above max_range_m are excluded."""
    cfg = _cfg(max_range_m=12.0)  # 12000 mm
    # 15000 mm > 12000 mm max range -> excluded
    frame = _make_frame(0.0, 10.0, [(15000, 200)] * 12)
    scan = LD19LidarDriver._assemble_scan([frame], cfg)
    assert scan.n_points == 0


def test_assemble_scan_keeps_in_range() -> None:
    """Points within [min_range, max_range] are kept."""
    cfg = _cfg(min_range_m=0.15, max_range_m=12.0)
    # 5000 mm = 5.0 m, within range
    frame = _make_frame(0.0, 10.0, [(5000, 200)] * 12)
    scan = LD19LidarDriver._assemble_scan([frame], cfg)
    assert scan.n_points == 12


def test_assemble_scan_at_boundary_distances() -> None:
    """Points exactly at min and max range boundaries are kept."""
    cfg = _cfg(min_range_m=0.15, max_range_m=12.0)
    # 150 mm == 0.15 m (min), 12000 mm == 12.0 m (max)
    points = [(150, 200), (12000, 200)]
    frame = _make_frame(0.0, 10.0, points)
    scan = LD19LidarDriver._assemble_scan([frame], cfg)
    assert scan.n_points == 2


# -- Empty input -------------------------------------------------------------


def test_assemble_scan_empty_frames() -> None:
    """Empty frame list returns an empty scan."""
    cfg = _cfg()
    scan = LD19LidarDriver._assemble_scan([], cfg)
    assert scan.n_points == 0


# -- Distances in mm --------------------------------------------------------


def test_assemble_scan_distances_in_mm() -> None:
    """Output distances are in millimetres."""
    cfg = _cfg()
    frame = _make_frame(0.0, 10.0, [(5000, 200)] * 3)
    scan = LD19LidarDriver._assemble_scan([frame], cfg)
    assert np.all(scan.distances_mm == pytest.approx(5000.0))


# -- Multiple frames --------------------------------------------------------


def test_assemble_scan_multiple_frames() -> None:
    """Multiple frames are merged into a single scan."""
    cfg = _cfg()
    frames = [
        _make_frame(0.0, 10.0, [(5000, 200)] * 4),
        _make_frame(10.0, 20.0, [(6000, 200)] * 4),
        _make_frame(20.0, 30.0, [(7000, 200)] * 4),
    ]
    scan = LD19LidarDriver._assemble_scan(frames, cfg)
    assert scan.n_points == 12


def test_read_frames_blocking_waits_for_coverage_after_wrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scan starting near 360° should not stop after the first small wrap."""
    from mousedroid.hardware.lidar import ld19_driver

    class FakeSerial:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks

        def read(self, _size: int) -> bytes:
            if self._chunks:
                return self._chunks.pop(0)
            return b""

    frame_angles = [(350.0 + 10.0 * i) % 360.0 for i in range(31)]
    frames = [
        _make_frame(angle, (angle + 4.0) % 360.0, [(5000, 200)] * 12)
        for angle in frame_angles
    ]
    serialized_chunks = [(b"\x54\x2c" * (47 * 2)) for _ in range(8)]

    driver = LD19LidarDriver(_cfg(min_scan_coverage_deg=270.0))
    driver._serial = FakeSerial(serialized_chunks)

    def _next_frame(_frame_bytes: bytes) -> tuple[LD19Frame | None, str | None]:
        if frames:
            return frames.pop(0), None
        return None, "crc"

    monkeypatch.setattr(
        ld19_driver.LD19FrameParser,
        "parse_frame_with_reason",
        staticmethod(_next_frame),
    )

    scan_frames = driver._read_frames_blocking()
    scan = LD19LidarDriver._assemble_scan(scan_frames, driver._cfg)

    assert len(scan_frames) >= 28
    assert scan.n_points > 0
    assert float(scan.angles_deg.max() - scan.angles_deg.min()) >= 270.0


def test_read_frames_blocking_resyncs_after_false_header() -> None:
    """A stray 0x54 byte should not cause the reader to discard a valid frame."""
    class FakeSerial:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks

        def read(self, _size: int) -> bytes:
            if self._chunks:
                return self._chunks.pop(0)
            return b""

    valid_frame_a = _build_frame_bytes(0.0, 8.0)
    valid_frame_b = _build_frame_bytes(10.0, 18.0)
    noise = b"\x54\x00garbage-prefix"

    driver = LD19LidarDriver(
        _cfg(min_scan_coverage_deg=5.0, scan_acquisition_timeout_s=0.5),
    )
    driver._serial = FakeSerial([noise + valid_frame_a + valid_frame_b])

    scan_frames = driver._read_frames_blocking()

    assert len(scan_frames) == 2
    assert scan_frames[0].start_angle_deg == pytest.approx(0.0)
    assert scan_frames[1].start_angle_deg == pytest.approx(10.0)


def test_read_frames_blocking_reports_crc_and_resync_stats() -> None:
    """Driver diagnostics should distinguish CRC failures from successful resync."""

    class FakeSerial:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks

        def read(self, _size: int) -> bytes:
            if self._chunks:
                return self._chunks.pop(0)
            return b""

    valid_frame_a = _build_frame_bytes(0.0, 8.0)
    valid_frame_b = _build_frame_bytes(10.0, 18.0)
    corrupt_frame = valid_frame_a[:-1] + bytes(((valid_frame_a[-1] + 1) % 256,))
    noise = b"noise-before-frame"

    driver = LD19LidarDriver(
        _cfg(min_scan_coverage_deg=5.0, scan_acquisition_timeout_s=0.5),
    )
    driver._serial = FakeSerial([noise + corrupt_frame + valid_frame_a + valid_frame_b])

    scan_frames, stats = driver._read_frames_with_stats_blocking()

    assert len(scan_frames) == 2
    assert stats.frames_parsed == 2
    assert stats.parse_failures >= 1
    assert stats.crc_failures >= 1
    assert stats.prefix_hits >= 3
    assert stats.bytes_discarded >= len(noise) + 1
    assert stats.bytes_read == len(noise + corrupt_frame + valid_frame_a + valid_frame_b)
