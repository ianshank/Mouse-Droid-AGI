"""Reusable runtime validation helpers.

These helpers keep Jetson smoke tests and verification scripts aligned with
the same config overlays and factory-backed drivers used by the application.
"""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray

from mousedroid.config.loader import load_settings
from mousedroid.factory import build_camera, build_microphone, build_speaker

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings
    from mousedroid.sensing.lidar_scan import LidarScan


_CONFIG_LIST_ENV_VARS = ("MOUSEDROID_CONFIGS", "MOUSEDROID_JETSON_CONFIGS")


def resolve_runtime_config_paths(
    config_paths: Sequence[Path | str] | None = None,
) -> tuple[Path, ...]:
    """Resolve runtime config overlays from explicit args or environment.

    Precedence:
        1. Explicit ``config_paths`` passed by the caller.
        2. CSV lists in ``MOUSEDROID_CONFIGS`` or ``MOUSEDROID_JETSON_CONFIGS``.
        3. Single-path ``MOUSEDROID_CONFIG``.

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

    single_path = os.getenv("MOUSEDROID_CONFIG", "").strip()
    if single_path:
        return (Path(single_path),)

    return ()


def load_runtime_settings(config_paths: Sequence[Path | str] | None = None) -> Settings:
    """Load runtime settings using the resolved config overlay list."""
    resolved_paths = resolve_runtime_config_paths(config_paths)
    return load_settings(*resolved_paths)


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
        capture_frame = getattr(camera, "_capture_frame", None)
        if not callable(capture_frame):
            msg = "camera driver does not expose raw frame capture"
            raise RuntimeError(msg)

        frame = await asyncio.to_thread(capture_frame)
        backend_name = str(getattr(camera, "_backend", camera.__class__.__name__))
        return np.asarray(frame, dtype=np.uint8), backend_name
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
    if scan.n_points < 2:
        return 0.0

    angles_deg = np.sort(np.asarray(scan.angles_deg, dtype=np.float32))
    wrapped_angles_deg = np.concatenate((angles_deg, angles_deg[:1] + 360.0))
    largest_gap_deg = float(np.max(np.diff(wrapped_angles_deg)))
    return max(0.0, 360.0 - largest_gap_deg)


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
        Number of samples written, or ``None`` when the speaker is disabled.

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

        min_samples = max(1, int(round(float(speaker.sample_rate) * duration_s)))
        total_samples = max(
            speaker.chunk_size,
            math.ceil(min_samples / speaker.chunk_size) * speaker.chunk_size,
        )
        time_axis = np.arange(total_samples, dtype=np.float32) / float(speaker.sample_rate)
        tone = (0.2 * np.sin(2.0 * np.pi * frequency_hz * time_axis)).astype(np.float32)

        for start in range(0, total_samples, speaker.chunk_size):
            chunk = tone[start : start + speaker.chunk_size]
            if chunk.shape[0] < speaker.chunk_size:
                chunk = np.pad(chunk, (0, speaker.chunk_size - chunk.shape[0]))
            await speaker.write_chunk(chunk)

        return total_samples
    finally:
        await speaker.stop()