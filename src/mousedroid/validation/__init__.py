"""Validation helpers for Jetson smoke and hardware checks.

The heavy sensor-runtime helpers (``capture_camera_frame`` et al.) pull in
``numpy``/``cv2``/``pyaudio``. They remain importable from this package for
backwards compatibility, but are re-exported **lazily** via :pep:`562` so that
importing a pure, dependency-free sibling (``latency_stats``, ``report_store``)
does not drag the whole sensor stack — and its native extensions — into the
process. This keeps the pure modules cheap to import (operator probes, unit
tests) and avoids coupling them to hardware-only dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Bind the names for static analysers (mypy/IDEs) without importing the
    # heavy module at runtime. The runtime path goes through ``__getattr__``.
    from mousedroid.validation.runtime import (
        CameraFrameDiagnostics,
        HailoDiagnostics,
        LidarScanDiagnostics,
        PcieSsdDiagnostics,
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
        verify_hailo_accelerator,
        verify_pcie_ssd_layout,
    )

# Names re-exported (lazily) from the heavy runtime module. Listed literally so
# static tooling recognises the re-exports; ``_RUNTIME_EXPORTS`` reuses it for
# the ``__getattr__`` membership test (single source of truth).
__all__ = [
    "CameraFrameDiagnostics",
    "HailoDiagnostics",
    "LidarScanDiagnostics",
    "PcieSsdDiagnostics",
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
    "verify_hailo_accelerator",
    "verify_pcie_ssd_layout",
]
_RUNTIME_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Lazily resolve the runtime re-exports on first access (:pep:`562`)."""
    if name in _RUNTIME_EXPORTS:
        from mousedroid.validation import runtime

        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
