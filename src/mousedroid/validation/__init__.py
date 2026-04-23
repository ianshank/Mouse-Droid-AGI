"""Validation helpers for Jetson smoke and hardware checks."""

from mousedroid.validation.runtime import (
    LidarScanDiagnostics,
    capture_camera_frame,
    capture_microphone_chunk,
    collect_lidar_diagnostics,
    lidar_scan_coverage_deg,
    lidar_scan_largest_gap_deg,
    lidar_scan_validation_coverage_deg,
    load_runtime_settings,
    play_speaker_tone,
    read_lidar_scan,
    resolve_runtime_config_paths,
)

__all__ = [
    "LidarScanDiagnostics",
    "capture_camera_frame",
    "capture_microphone_chunk",
    "collect_lidar_diagnostics",
    "lidar_scan_coverage_deg",
    "lidar_scan_largest_gap_deg",
    "lidar_scan_validation_coverage_deg",
    "load_runtime_settings",
    "play_speaker_tone",
    "read_lidar_scan",
    "resolve_runtime_config_paths",
]
