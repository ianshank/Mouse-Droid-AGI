"""Reusable runtime validation helpers.

These helpers keep Jetson smoke tests and verification scripts aligned with
the same config overlays and factory-backed drivers used by the application.

This package is split by hardware domain (camera, Hailo accelerator, PCIe
SSD, audio, LiDAR, plus shared config/logging plumbing) but re-exports every
symbol that used to live in the flat ``runtime.py`` module from this
``__init__``, so ``from mousedroid.validation.runtime import X`` and
``from mousedroid.validation import runtime; runtime.X`` keep working
identically for every existing caller.
"""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from mousedroid.common.imports import module_importable
from mousedroid.config.loader import load_settings
from mousedroid.factory import build_camera, build_microphone, build_speaker, build_voice_engine
from mousedroid.logging.setup import get_logger

from ._audio import capture_microphone_chunk, play_rocky_voice_phrase, play_speaker_tone
from ._camera import (
    CameraFrameDiagnostics,
    _encode_camera_frame_jpeg,
    _resolve_raw_frame_capture,
    _snapshot_jpeg_quality,
    camera_unavailable_reason,
    capture_camera_frame,
)
from ._hailo import HailoDiagnostics, _discover_hef_role_fields, verify_hailo_accelerator
from ._lidar import (
    LidarScanDiagnostics,
    collect_lidar_diagnostics,
    lidar_scan_coverage_deg,
    lidar_scan_largest_gap_deg,
    lidar_scan_validation_coverage_deg,
    read_lidar_scan,
)
from ._shared import (
    _ARGUS_SOCKET_PATH,
    _CONFIG_LIST_ENV_VARS,
    _CONFIG_SINGLE_ENV_VARS,
    _DEFAULT_SMOKE_PHRASE,
    _collect_configured_runtime_paths,
    _log,
    _subprocess_timeout_s,
    load_runtime_settings,
    resolve_runtime_config_paths,
)
from ._storage import (
    PcieSsdDiagnostics,
    _nvme_device_for,
    _nvme_partition_for,
    _resolve_pcie_ssd_mount,
    verify_pcie_ssd_layout,
)

__all__ = [
    "TYPE_CHECKING",
    "_ARGUS_SOCKET_PATH",
    "_CONFIG_LIST_ENV_VARS",
    "_CONFIG_SINGLE_ENV_VARS",
    "_DEFAULT_SMOKE_PHRASE",
    "Awaitable",
    "Callable",
    "CameraFrameDiagnostics",
    "HailoDiagnostics",
    "LidarScanDiagnostics",
    "NDArray",
    "Path",
    "PcieSsdDiagnostics",
    "Sequence",
    "_collect_configured_runtime_paths",
    "_discover_hef_role_fields",
    "_encode_camera_frame_jpeg",
    "_log",
    "_nvme_device_for",
    "_nvme_partition_for",
    "_resolve_pcie_ssd_mount",
    "_resolve_raw_frame_capture",
    "_snapshot_jpeg_quality",
    "_subprocess_timeout_s",
    "asyncio",
    "build_camera",
    "build_microphone",
    "build_speaker",
    "build_voice_engine",
    "camera_unavailable_reason",
    "capture_camera_frame",
    "capture_microphone_chunk",
    "cast",
    "collect_lidar_diagnostics",
    "dataclass",
    "field",
    "get_logger",
    "lidar_scan_coverage_deg",
    "lidar_scan_largest_gap_deg",
    "lidar_scan_validation_coverage_deg",
    "load_runtime_settings",
    "load_settings",
    "math",
    "module_importable",
    "np",
    "os",
    "play_rocky_voice_phrase",
    "play_speaker_tone",
    "read_lidar_scan",
    "resolve_runtime_config_paths",
    "subprocess",
    "time",
    "verify_hailo_accelerator",
    "verify_pcie_ssd_layout",
]
