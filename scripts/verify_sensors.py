#!/usr/bin/env python3
"""Standalone sensor verification script for MouseDroid Jetson hardware.

This script validates the deployed runtime configuration by loading the same
YAML overlays and factory-backed drivers used by the application.

Usage:
        python scripts/verify_sensors.py [--config config/jetson_production.yaml ...]
        python scripts/verify_sensors.py --sensor camera --config config/jetson_production.yaml
        python scripts/verify_sensors.py --json

Exit code is non-zero if any requested sensor reports FAIL.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

# Make the src tree importable when run from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_FAILURES: list[str] = []
_RESULTS: dict[str, dict[str, object]] = {}

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _ok(label: str, detail: str = "", *, sensor: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {_GREEN}[PASS]{_RESET} {label}{suffix}")
    if sensor:
        _RESULTS.setdefault(sensor, {})["status"] = "pass"
        if detail:
            _RESULTS[sensor]["detail"] = detail


def _fail(label: str, detail: str = "", *, sensor: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {_RED}[FAIL]{_RESET} {label}{suffix}")
    _FAILURES.append(label)
    if sensor:
        _RESULTS.setdefault(sensor, {})["status"] = "fail"
        if detail:
            _RESULTS[sensor]["error"] = detail


def _skip(label: str, reason: str = "", *, sensor: str = "") -> None:
    suffix = f"  ({reason})" if reason else ""
    print(f"  {_YELLOW}[SKIP]{_RESET} {label}{suffix}")
    if sensor:
        _RESULTS.setdefault(sensor, {})["status"] = "skip"
        if reason:
            _RESULTS[sensor]["reason"] = reason


def _section(title: str) -> None:
    bar = "=" * (len(title) + 4)
    print(f"\n{bar}\n  {title}\n{bar}")


from mousedroid.config.schema import Settings
from mousedroid.factory import build_distance_sensor
from mousedroid.validation.runtime import (
    capture_camera_frame,
    capture_microphone_chunk,
    lidar_scan_coverage_deg,
    load_runtime_settings,
    play_speaker_tone,
    read_lidar_scan,
    resolve_runtime_config_paths,
)


def check_camera(cfg: Settings) -> None:
    """Verify the configured camera can capture one frame."""
    _section("Video (Camera)")

    t0 = time.monotonic()
    try:
        frame, backend_name = asyncio.run(capture_camera_frame(cfg))
    except Exception as exc:
        _fail("camera capture", str(exc), sensor="camera")
        return

    elapsed = time.monotonic() - t0
    height, width = int(frame.shape[0]), int(frame.shape[1])
    expected_height = cfg.camera.resolution_height
    expected_width = cfg.camera.resolution_width

    if height != expected_height or width != expected_width:
        _fail(
            "frame shape",
            f"expected ({expected_height}, {expected_width}), got ({height}, {width})",
            sensor="camera",
        )
        return

    _ok(
        "frame capture",
        f"{width}x{height} via {backend_name} in {elapsed:.2f}s, dtype={frame.dtype}",
        sensor="camera",
    )


def check_audio(cfg: Settings) -> None:
    """Verify the configured microphone can capture one chunk."""
    _section("Audio (USB Microphone)")

    if cfg.microphone is None or not cfg.microphone.enabled:
        _skip("USB microphone", "disabled in config", sensor="audio")
        return

    try:
        import pyaudio
    except ImportError:
        _fail("pyaudio import", "library not installed", sensor="audio")
        return

    pa = pyaudio.PyAudio()
    input_devices: list[str] = []
    try:
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) > 0:
                input_devices.append(f"[{index}] {info['name']}")
    finally:
        pa.terminate()

    if not input_devices:
        _fail("USB mic detected", "no input devices found", sensor="audio")
        return

    t0 = time.monotonic()
    try:
        chunk = asyncio.run(capture_microphone_chunk(cfg))
    except Exception as exc:
        _fail("UsbMicrophone chunk capture", str(exc), sensor="audio")
        return

    if chunk is None:
        _skip("UsbMicrophone chunk capture", "disabled in config", sensor="audio")
        return

    elapsed = time.monotonic() - t0
    expected_shape = (cfg.microphone.chunk_size * cfg.microphone.channels,)
    if chunk.dtype != np.float32:
        _fail("chunk dtype", f"expected float32, got {chunk.dtype}", sensor="audio")
        return
    if chunk.shape != expected_shape:
        _fail(
            "chunk shape",
            f"expected {expected_shape}, got {chunk.shape}",
            sensor="audio",
        )
        return

    _ok(
        "UsbMicrophone chunk capture",
        f"devices={'; '.join(input_devices)}, shape={chunk.shape}, {elapsed:.2f}s",
        sensor="audio",
    )


def check_ultrasonic(cfg: Settings) -> None:
    """Verify the configured ultrasonic sensor can read a distance."""
    _section("Ultrasonic (HC-SR04)")

    if cfg.ultrasonic is None:
        _skip("Ultrasonic sensor", "disabled in config", sensor="ultrasonic")
        return

    try:
        import Jetson.GPIO  # noqa: F401
    except ImportError:
        _fail("Jetson.GPIO import", "library not installed", sensor="ultrasonic")
        return

    sensor = build_distance_sensor(cfg)
    t0 = time.monotonic()
    try:
        distance_m = asyncio.run(sensor.read_distance_m())
    except Exception as exc:
        _fail("HcSr04 distance read", str(exc), sensor="ultrasonic")
        return

    elapsed = time.monotonic() - t0
    if not (sensor.min_range_m <= distance_m <= sensor.max_range_m):
        _fail(
            "distance range",
            f"{distance_m:.3f} m outside [{sensor.min_range_m}, {sensor.max_range_m}] m",
            sensor="ultrasonic",
        )
        return

    _ok(
        "HcSr04 distance read",
        f"{distance_m:.3f} m in {elapsed:.2f}s",
        sensor="ultrasonic",
    )


def check_lidar(cfg: Settings) -> None:
    """Verify the configured LiDAR can produce one scan."""
    _section("LiDAR (LD19)")

    if cfg.lidar is None or not cfg.lidar.enabled:
        _skip("LiDAR sensor", "disabled in config", sensor="lidar")
        return

    t0 = time.monotonic()
    try:
        scan = asyncio.run(read_lidar_scan(cfg))
    except Exception as exc:
        _fail("LiDAR scan read", str(exc), sensor="lidar")
        return

    if scan is None:
        _skip("LiDAR driver", "build_lidar returned None", sensor="lidar")
        return

    elapsed = time.monotonic() - t0
    if scan.n_points < 10:
        _fail(
            "LiDAR scan points",
            f"only {scan.n_points} points (expected 10+)",
            sensor="lidar",
        )
        return

    coverage_deg = lidar_scan_coverage_deg(scan)
    min_coverage_deg = cfg.lidar.min_scan_coverage_deg
    if coverage_deg < min_coverage_deg:
        _fail(
            "LiDAR scan coverage",
            f"{coverage_deg:.1f}\N{DEGREE SIGN} below {min_coverage_deg:.1f}\N{DEGREE SIGN} threshold",
            sensor="lidar",
        )
        return

    _ok(
        "LiDAR scan read",
        (
            f"{scan.n_points} points on {cfg.lidar.serial_port} in {elapsed:.2f}s, "
            f"coverage={coverage_deg:.1f}\N{DEGREE SIGN}"
        ),
        sensor="lidar",
    )


def check_speaker(cfg: Settings) -> None:
    """Verify the configured speaker can play a short tone."""
    _section("Speaker (Audio Output)")

    if cfg.speaker is None or not cfg.speaker.enabled:
        _skip("Speaker", "disabled in config", sensor="speaker")
        return

    t0 = time.monotonic()
    try:
        written_samples = asyncio.run(play_speaker_tone(cfg))
    except Exception as exc:
        _fail("Speaker play", str(exc), sensor="speaker")
        return

    if written_samples is None:
        _skip("Speaker driver", "disabled in config", sensor="speaker")
        return

    elapsed = time.monotonic() - t0
    _ok(
        "Speaker play",
        f"{written_samples} samples at {cfg.speaker.sample_rate} Hz in {elapsed:.2f}s",
        sensor="speaker",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_ALL_SENSORS = ("camera", "audio", "ultrasonic", "lidar", "speaker")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify access to MouseDroid hardware sensors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  python scripts/verify_sensors.py\n"
        "  python scripts/verify_sensors.py --sensor camera\n"
        "  python scripts/verify_sensors.py --sensor lidar\n"
        "  python scripts/verify_sensors.py --json\n",
    )
    ap.add_argument(
        "--config",
        type=Path,
        nargs="*",
        default=[],
        help="YAML config overlay files (defaults to MOUSEDROID_CONFIG* env vars)",
    )
    ap.add_argument(
        "--sensor",
        choices=["all", *_ALL_SENSORS],
        default="all",
        help="Which sensor to verify (default: all)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (machine-readable)",
    )
    args = ap.parse_args()

    cfg_paths = resolve_runtime_config_paths(args.config)
    cfg = load_runtime_settings(args.config)

    if not args.json:
        print("\nMouseDroid Sensor Verification")
        print(f"Running on Python {sys.version.split()[0]}")
        print(f"Project root: {_PROJECT_ROOT}")
        if cfg_paths:
            print("Config overlays:")
            for cfg_path in cfg_paths:
                print(f"  - {cfg_path}")
        else:
            print("Config overlays: default.yaml only")

    sensor_checks: dict[str, object] = {
        "camera": check_camera,
        "audio": check_audio,
        "ultrasonic": check_ultrasonic,
        "lidar": check_lidar,
        "speaker": check_speaker,
    }

    for name, check_fn in sensor_checks.items():
        if args.sensor in ("all", name):
            check_fn(cfg)

    # Summary
    selected_sensors = [sensor for sensor in _ALL_SENSORS if args.sensor in ("all", sensor)]
    if args.json:
        import json

        result = {
            "sensors": _RESULTS,
            "failures": _FAILURES,
            "passed": sum(
                1 for sensor in selected_sensors if _RESULTS.get(sensor, {}).get("status") == "pass"
            ),
            "total": len(selected_sensors),
        }
        print(json.dumps(result, indent=2))
    else:
        _section("Summary")
        total = len(selected_sensors)
        failed = len(_FAILURES)
        passed = total - failed

        print(f"  {passed}/{total} sensors OK")
        if _FAILURES:
            print(f"  {_RED}Failed:{_RESET} {', '.join(_FAILURES)}")
        else:
            print(f"  {_GREEN}All sensors accessible.{_RESET}")

    if _FAILURES:
        sys.exit(1)


if __name__ == "__main__":
    main()
