"""LiDAR scan capture + coverage-diagnostics runtime validation helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings
    from mousedroid.hardware.lidar.ld19_driver import LD19ReadStats
    from mousedroid.sensing.lidar_scan import LidarScan


@dataclass(frozen=True)
class LidarScanDiagnostics:
    """Structured diagnostics for one LiDAR scan acquisition."""

    scan_index: int
    n_points: int
    coverage_deg: float
    validation_coverage_deg: float
    largest_gap_deg: float
    largest_gap_start_deg: float | None
    largest_gap_end_deg: float | None
    min_angle_deg: float | None
    max_angle_deg: float | None
    elapsed_s: float
    bytes_read: int
    chunks_read: int
    empty_reads: int
    prefix_hits: int
    header_search_misses: int
    bytes_discarded: int
    parse_failures: int
    crc_failures: int
    frames_parsed: int
    driver_covered_angle_deg: float
    meets_min_coverage: bool


async def read_lidar_scan(cfg: Settings) -> LidarScan | None:
    """Read a single LiDAR scan through the configured factory.

    Args:
        cfg: Fully resolved settings.

    Returns:
        Captured scan, or ``None`` when LiDAR is disabled.
    """
    from mousedroid.factory import build_lidar

    lidar = build_lidar(cfg)
    if lidar is None:
        return None

    await lidar.start()
    try:
        return await lidar.read_scan()
    finally:
        await lidar.stop()


def lidar_scan_coverage_deg(scan: LidarScan) -> float:
    """Estimate angular coverage for a LiDAR scan.

    The coverage is computed as the complement of the largest angular gap,
    which handles scans that wrap around 0 degrees.

    Args:
        scan: Captured LiDAR scan.

    Returns:
        Angular coverage in degrees in the inclusive range ``[0, 360]``.
    """
    largest_gap_deg, _, _ = lidar_scan_largest_gap_deg(scan)
    return max(0.0, 360.0 - largest_gap_deg)


def lidar_scan_validation_coverage_deg(
    scan: LidarScan,
    *,
    driver_covered_angle_deg: float | None = None,
) -> float:
    """Return the coverage metric used by smoke and runtime validation.

    Filtered point coverage can under-report healthy scans in sparse environments,
    so validation prefers the driver's frame coverage when available.

    Args:
        scan: Captured LiDAR scan.
        driver_covered_angle_deg: Frame-based coverage reported by the driver.

    Returns:
        Coverage in degrees suitable for validation thresholds.
    """
    point_coverage_deg = lidar_scan_coverage_deg(scan)
    if driver_covered_angle_deg is None:
        return point_coverage_deg

    return max(point_coverage_deg, max(0.0, float(driver_covered_angle_deg)))


def lidar_scan_largest_gap_deg(scan: LidarScan) -> tuple[float, float | None, float | None]:
    """Return the largest angular gap and its bounding angles.

    Args:
        scan: Captured LiDAR scan.

    Returns:
        Tuple of ``(largest_gap_deg, gap_start_deg, gap_end_deg)``.
        When fewer than two points are present, the gap defaults to a full
        360-degree blind spot with unknown bounds.
    """
    if scan.n_points < 2:
        return 360.0, None, None

    angles_deg = np.sort(np.asarray(scan.angles_deg, dtype=np.float32))
    wrapped_angles_deg = np.concatenate((angles_deg, angles_deg[:1] + 360.0))
    gap_sizes_deg = np.diff(wrapped_angles_deg)
    gap_idx = int(np.argmax(gap_sizes_deg))
    return (
        float(gap_sizes_deg[gap_idx]),
        float(wrapped_angles_deg[gap_idx] % 360.0),
        float(wrapped_angles_deg[gap_idx + 1] % 360.0),
    )


async def collect_lidar_diagnostics(
    cfg: Settings,
    *,
    n_scans: int = 1,
) -> list[LidarScanDiagnostics]:
    """Collect repeated LiDAR scan diagnostics through the configured driver."""
    from mousedroid.factory import build_lidar

    if n_scans <= 0:
        return []
    if cfg.lidar is None or not cfg.lidar.enabled:
        return []

    lidar = build_lidar(cfg)
    if lidar is None:
        return []

    read_with_diagnostics = getattr(lidar, "read_scan_with_diagnostics", None)
    diagnostics: list[LidarScanDiagnostics] = []

    await lidar.start()
    try:
        for scan_index in range(n_scans):
            started_at = time.monotonic()
            read_stats: LD19ReadStats | None = None

            if callable(read_with_diagnostics):
                scan, read_stats = await read_with_diagnostics()
            else:
                scan = await lidar.read_scan()

            largest_gap_deg, gap_start_deg, gap_end_deg = lidar_scan_largest_gap_deg(scan)
            coverage_deg = lidar_scan_coverage_deg(scan)
            driver_covered_angle_deg = float(getattr(read_stats, "covered_angle_deg", 0.0))
            validation_coverage_deg = lidar_scan_validation_coverage_deg(
                scan,
                driver_covered_angle_deg=(
                    driver_covered_angle_deg if driver_covered_angle_deg > 0.0 else None
                ),
            )
            diagnostics.append(
                LidarScanDiagnostics(
                    scan_index=scan_index,
                    n_points=scan.n_points,
                    coverage_deg=coverage_deg,
                    validation_coverage_deg=validation_coverage_deg,
                    largest_gap_deg=largest_gap_deg,
                    largest_gap_start_deg=gap_start_deg,
                    largest_gap_end_deg=gap_end_deg,
                    min_angle_deg=(float(np.min(scan.angles_deg)) if scan.n_points else None),
                    max_angle_deg=(float(np.max(scan.angles_deg)) if scan.n_points else None),
                    elapsed_s=float(
                        getattr(read_stats, "elapsed_s", time.monotonic() - started_at),
                    ),
                    bytes_read=int(getattr(read_stats, "bytes_read", 0)),
                    chunks_read=int(getattr(read_stats, "chunks_read", 0)),
                    empty_reads=int(getattr(read_stats, "empty_reads", 0)),
                    prefix_hits=int(getattr(read_stats, "prefix_hits", 0)),
                    header_search_misses=int(getattr(read_stats, "header_search_misses", 0)),
                    bytes_discarded=int(getattr(read_stats, "bytes_discarded", 0)),
                    parse_failures=int(getattr(read_stats, "parse_failures", 0)),
                    crc_failures=int(getattr(read_stats, "crc_failures", 0)),
                    frames_parsed=int(getattr(read_stats, "frames_parsed", 0)),
                    driver_covered_angle_deg=driver_covered_angle_deg,
                    meets_min_coverage=validation_coverage_deg >= cfg.lidar.min_scan_coverage_deg,
                ),
            )
    finally:
        await lidar.stop()

    return diagnostics
