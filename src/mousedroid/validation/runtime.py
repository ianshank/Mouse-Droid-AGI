"""Reusable runtime validation helpers.

These helpers keep Jetson smoke tests and verification scripts aligned with
the same config overlays and factory-backed drivers used by the application.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.config.loader import load_settings
from mousedroid.factory import build_camera, build_microphone, build_speaker, build_voice_engine

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings
    from mousedroid.hardware.lidar.ld19_driver import LD19ReadStats
    from mousedroid.sensing.lidar_scan import LidarScan


_CONFIG_LIST_ENV_VARS = ("MOUSEDROID_CONFIGS", "MOUSEDROID_JETSON_CONFIGS")
_CONFIG_SINGLE_ENV_VARS = ("MOUSEDROID_CONFIG", "MOUSEDROID_JETSON_CONFIG")


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


def resolve_runtime_config_paths(
    config_paths: Sequence[Path | str] | None = None,
) -> tuple[Path, ...]:
    """Resolve runtime config overlays from explicit args or environment.

    Precedence:
        1. Explicit ``config_paths`` passed by the caller.
        2. CSV lists in ``MOUSEDROID_CONFIGS`` or ``MOUSEDROID_JETSON_CONFIGS``.
        3. Single-path ``MOUSEDROID_CONFIG`` or legacy ``MOUSEDROID_JETSON_CONFIG``.

    Args:
        config_paths: Explicit config overlay paths.

    Returns:
        Normalized config paths in precedence order.
    """
    resolved = tuple(Path(str(path)) for path in (config_paths or ()) if str(path).strip())
    if resolved:
        return resolved

    for env_var in _CONFIG_LIST_ENV_VARS:
        raw_value = os.getenv(env_var, "").strip()
        if not raw_value:
            continue
        csv_paths = [Path(part.strip()) for part in raw_value.split(",") if part.strip()]
        if csv_paths:
            return tuple(csv_paths)

    for env_var in _CONFIG_SINGLE_ENV_VARS:
        single_path = os.getenv(env_var, "").strip()
        if single_path:
            return (Path(single_path),)

    return ()


def load_runtime_settings(config_paths: Sequence[Path | str] | None = None) -> Settings:
    """Load runtime settings using the resolved config overlay list."""
    resolved_paths = resolve_runtime_config_paths(config_paths)
    return load_settings(*resolved_paths)


def camera_unavailable_reason(cfg: Settings, exc: Exception | None = None) -> str | None:
    """Return a skip-worthy reason when the Jetson camera runtime is unavailable."""
    if cfg.camera.backend != "jetson_csi":
        return None

    reasons: list[str] = []
    device_path = str(cfg.camera.device_path).strip()
    if device_path and not Path(device_path).exists():
        reasons.append(f"V4L2 device {device_path} is missing")
    if not Path("/tmp/argus_socket").exists():  # noqa: S108
        reasons.append("libargus socket /tmp/argus_socket is missing")

    if not reasons:
        return None

    if exc is not None and str(exc).strip():
        reasons.append(str(exc).strip())
    return "; ".join(reasons)


async def capture_camera_frame(cfg: Settings) -> tuple[NDArray[np.uint8], str]:
    """Capture one raw frame through the configured camera factory.

    Args:
        cfg: Fully resolved settings.

    Returns:
        Tuple of ``(frame, backend_name)``.

    Raises:
        RuntimeError: If the camera driver cannot expose a raw frame.
    """
    camera = build_camera(cfg)
    await camera.start()
    try:
        capture_raw = getattr(camera, "capture_raw_frame", None)
        if callable(capture_raw):
            frame = np.asarray(await capture_raw(), dtype=np.uint8)
        else:
            # Backward-compatible path for drivers that expose the legacy private
            # blocking helper but not the public async method.
            capture_frame = getattr(camera, "_capture_frame", None)
            if not callable(capture_frame):
                msg = "camera driver does not expose raw frame capture"
                raise RuntimeError(msg)
            frame = np.asarray(await asyncio.to_thread(capture_frame), dtype=np.uint8)
        backend_name = str(getattr(camera, "_backend", camera.__class__.__name__))
        return frame, backend_name
    finally:
        await camera.stop()


async def capture_microphone_chunk(cfg: Settings) -> NDArray[np.float32] | None:
    """Capture one chunk through the configured microphone driver.

    Returns ``None`` when the microphone is disabled in config.

    Args:
        cfg: Fully resolved settings.

    Returns:
        Captured audio chunk, or ``None`` when disabled.

    Raises:
        RuntimeError: If the configured microphone cannot open its runtime stream.
    """
    microphone = build_microphone(cfg)
    if microphone is None:
        return None

    await microphone.start()
    try:
        if getattr(microphone, "_stream", object()) is None:
            msg = "configured microphone device unavailable"
            raise RuntimeError(msg)
        chunk = await microphone.read_chunk()
        return np.asarray(chunk, dtype=np.float32)
    finally:
        await microphone.stop()


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


async def play_speaker_tone(
    cfg: Settings,
    *,
    duration_s: float = 0.3,
    frequency_hz: float = 440.0,
) -> int | None:
    """Play a short tone through the configured speaker driver.

    Args:
        cfg: Fully resolved settings.
        duration_s: Tone duration.
        frequency_hz: Tone frequency.

    Returns:
        Total number of interleaved samples written (``frames * channels``),
        or ``None`` when the speaker is disabled.

    Raises:
        RuntimeError: If the configured speaker cannot open its runtime stream.
    """
    speaker = build_speaker(cfg)
    if speaker is None:
        return None

    await speaker.start()
    try:
        if getattr(speaker, "_stream", object()) is None:
            msg = "configured speaker device unavailable"
            raise RuntimeError(msg)

        channels = max(1, int(getattr(speaker, "channels", 1)))
        min_frames = max(1, round(float(speaker.sample_rate) * duration_s))
        total_frames = max(
            speaker.chunk_size,
            math.ceil(min_frames / speaker.chunk_size) * speaker.chunk_size,
        )
        time_axis = np.arange(total_frames, dtype=np.float32) / float(speaker.sample_rate)
        mono_tone = (0.2 * np.sin(2.0 * np.pi * frequency_hz * time_axis)).astype(np.float32)
        # Interleave identical tone across channels so each frame is `channels` samples.
        interleaved = np.repeat(mono_tone, channels) if channels > 1 else mono_tone

        samples_per_chunk = speaker.chunk_size * channels
        total_samples = total_frames * channels
        for start in range(0, total_samples, samples_per_chunk):
            chunk = interleaved[start : start + samples_per_chunk]
            if chunk.shape[0] < samples_per_chunk:
                chunk = np.pad(chunk, (0, samples_per_chunk - chunk.shape[0]))
            await speaker.write_chunk(chunk)

        return total_samples
    finally:
        await speaker.stop()


async def play_rocky_voice_phrase(
    cfg: Settings,
    *,
    phrase: str = "Hello hello! Rocky ready!",
) -> tuple[int, float] | None:
    """Play a short Rocky voice phrase through the configured voice pipeline.

    Args:
        cfg: Fully resolved settings.
        phrase: Short phrase to synthesize and play.

    Returns:
        Tuple of ``(samples_written, peak_abs_sample)``, or ``None`` when the
        voice engine is disabled.

    Raises:
        RuntimeError: If the voice pipeline cannot load TTS or write to the
            configured speaker.
    """
    if not cfg.voice.enabled:
        return None

    speaker = build_speaker(cfg)
    if speaker is None:
        raise RuntimeError("configured speaker unavailable for Rocky voice")

    engine = build_voice_engine(cfg, speaker=speaker)
    if engine is None:
        raise RuntimeError("Rocky voice engine unavailable")

    await engine.start()
    try:
        if getattr(speaker, "_stream", object()) is None:
            raise RuntimeError("configured speaker device unavailable")

        tts = getattr(engine, "_tts", None)
        if tts is None:
            raise RuntimeError("voice engine missing TTS backend")

        if not cfg.mock_hardware and getattr(tts, "_voice", None) is None:
            model_path = str(cfg.voice.tts_model_path or "").strip()
            if not model_path:
                raise RuntimeError("voice.tts_model_path is not configured")
            raise RuntimeError(f"Piper voice model failed to load from {model_path}")

        samples = np.asarray(await tts.synthesize(phrase), dtype=np.float32)
        if samples.size == 0:
            raise RuntimeError("Rocky voice TTS returned no samples")

        peak_abs = float(np.max(np.abs(samples)))
        if not cfg.mock_hardware and peak_abs <= 1e-6:
            raise RuntimeError("Rocky voice TTS returned silent audio")

        write_samples = getattr(engine, "_write_samples", None)
        if not callable(write_samples):
            raise RuntimeError("voice engine missing sample writer")
        await write_samples(samples)

        return int(samples.size), peak_abs
    finally:
        await engine.stop()
