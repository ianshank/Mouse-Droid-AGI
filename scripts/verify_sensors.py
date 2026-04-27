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
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Make the src tree importable when run from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from mousedroid.config.schema import Settings  # noqa: E402
from mousedroid.factory import build_distance_sensor  # noqa: E402
from mousedroid.validation.runtime import (  # noqa: E402
    capture_camera_frame,
    capture_microphone_chunk,
    collect_lidar_diagnostics,
    load_runtime_settings,
    play_rocky_voice_phrase,
    play_speaker_tone,
    resolve_runtime_config_paths,
)

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


def _diagnose_camera_host(cfg: Settings) -> str | None:
    """Return an actionable diagnosis when no /dev/video* node is present.

    Inspects the configured ``camera.device_path`` plus any sibling video
    nodes, and (when running on a Jetson host) tails the ``nvargus-daemon``
    journal for the most recent IMX/CSI probe error. Returns ``None`` when
    the device path exists or diagnosis is not applicable; otherwise returns
    a single-line, operator-actionable string suitable for ``_fail`` detail.

    The function is best-effort: any subprocess or filesystem failure is
    swallowed so the caller still surfaces the original capture exception.
    """
    import subprocess

    device_path = Path(getattr(cfg.camera, "device_path", "/dev/video0"))
    siblings = sorted(Path("/dev").glob("video*"))
    if device_path.exists() or siblings:
        return None

    notes: list[str] = [f"{device_path} not present and no /dev/video* node enumerated"]

    media_nodes = sorted(Path("/dev").glob("media*"))
    if media_nodes:
        notes.append(f"media nodes present: {[str(p) for p in media_nodes]}")

    try:
        journal = subprocess.run(
            ["journalctl", "-u", "nvargus-daemon", "--no-pager", "-n", "60"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        for line in reversed(journal.stdout.splitlines()):
            lower = line.lower()
            if any(token in lower for token in ("imx", "probe of", "i2c read probe", "modulenotpresent")):
                notes.append(f"nvargus: {line.strip()}")
                break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    notes.append(
        "ACTION: confirm camera model matches device-tree overlay "
        "(check /boot/extlinux/extlinux.conf), reseat ribbon, and "
        "verify camera.device_path in jetson_production.yaml"
    )
    return " | ".join(notes)


def check_camera(cfg: Settings) -> None:
    """Verify the configured camera can capture one frame."""
    _section("Video (Camera)")

    t0 = time.monotonic()
    try:
        frame, backend_name = asyncio.run(capture_camera_frame(cfg))
    except Exception as exc:
        diagnosis = _diagnose_camera_host(cfg)
        detail = f"{exc} :: {diagnosis}" if diagnosis else str(exc)
        _fail("camera capture", detail, sensor="camera")
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
        import Jetson.GPIO  # noqa: F401  # pyright: ignore[reportMissingModuleSource]
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


def check_lidar(cfg: Settings, *, repeat: int = 1) -> None:
    """Verify the configured LiDAR can produce one scan."""
    _section("LiDAR (LD19)")

    if cfg.lidar is None or not cfg.lidar.enabled:
        _skip("LiDAR sensor", "disabled in config", sensor="lidar")
        return

    try:
        diagnostics = asyncio.run(collect_lidar_diagnostics(cfg, n_scans=max(1, repeat)))
    except Exception as exc:
        _fail("LiDAR scan read", str(exc), sensor="lidar")
        return

    if not diagnostics:
        _skip("LiDAR driver", "build_lidar returned None", sensor="lidar")
        return

    min_coverage_deg = cfg.lidar.min_scan_coverage_deg
    for diag in diagnostics:
        print(
            "  "
            f"scan[{diag.scan_index}] points={diag.n_points} "
            f"coverage={diag.coverage_deg:.1f}\N{DEGREE SIGN} "
            f"validation={diag.validation_coverage_deg:.1f}\N{DEGREE SIGN} "
            f"largest_gap={diag.largest_gap_deg:.1f}\N{DEGREE SIGN} "
            f"gap_window={diag.largest_gap_start_deg}->{diag.largest_gap_end_deg} "
            f"frames={diag.frames_parsed} parse_failures={diag.parse_failures} "
            f"crc_failures={diag.crc_failures} bytes_read={diag.bytes_read}"
        )

    _RESULTS.setdefault("lidar", {})["samples"] = [asdict(diag) for diag in diagnostics]

    min_points = min(diag.n_points for diag in diagnostics)
    min_coverage = min(diag.coverage_deg for diag in diagnostics)
    max_coverage = max(diag.coverage_deg for diag in diagnostics)
    min_validation_coverage = min(diag.validation_coverage_deg for diag in diagnostics)
    max_validation_coverage = max(diag.validation_coverage_deg for diag in diagnostics)
    total_parse_failures = sum(diag.parse_failures for diag in diagnostics)
    total_crc_failures = sum(diag.crc_failures for diag in diagnostics)

    if min_points < 10:
        _fail(
            "LiDAR scan points",
            f"minimum {min_points} points across {len(diagnostics)} scan(s) (expected 10+)",
            sensor="lidar",
        )
        return

    if min_validation_coverage < min_coverage_deg:
        _fail(
            "LiDAR scan coverage",
            (
                f"minimum validation coverage {min_validation_coverage:.1f}\N{DEGREE SIGN} "
                f"below {min_coverage_deg:.1f}\N{DEGREE SIGN} "
                f"threshold over {len(diagnostics)} scan(s); "
                f"point_coverage={min_coverage:.1f}-{max_coverage:.1f}\N{DEGREE SIGN}; "
                f"parse_failures={total_parse_failures}, crc_failures={total_crc_failures}"
            ),
            sensor="lidar",
        )
        return

    _ok(
        "LiDAR scan read",
        (
            f"{len(diagnostics)} scan(s) on {cfg.lidar.serial_port}; "
            f"coverage={min_coverage:.1f}-{max_coverage:.1f}\N{DEGREE SIGN}; "
            f"validation={min_validation_coverage:.1f}-{max_validation_coverage:.1f}\N{DEGREE SIGN}; "
            f"parse_failures={total_parse_failures}, crc_failures={total_crc_failures}"
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


def check_voice(cfg: Settings) -> None:
    """Verify the Rocky voice pipeline can synthesize and play speech."""
    _section("Rocky Voice (TTS + Speaker)")

    if not cfg.voice.enabled:
        _skip("Rocky voice", "disabled in config", sensor="voice")
        return

    t0 = time.monotonic()
    try:
        result = asyncio.run(play_rocky_voice_phrase(cfg))
    except Exception as exc:
        _fail("Rocky voice play", str(exc), sensor="voice")
        return

    if result is None:
        _skip("Rocky voice", "disabled in config", sensor="voice")
        return

    written_samples, peak_abs = result
    elapsed = time.monotonic() - t0
    _ok(
        "Rocky voice play",
        f"{written_samples} samples peak={peak_abs:.3f} in {elapsed:.2f}s",
        sensor="voice",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_ALL_SENSORS = ("camera", "audio", "ultrasonic", "lidar", "speaker", "voice")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify access to MouseDroid hardware sensors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  python scripts/verify_sensors.py\n"
        "  python scripts/verify_sensors.py --sensor camera\n"
        "  python scripts/verify_sensors.py --sensor lidar\n"
        "  python scripts/verify_sensors.py --sensor voice\n"
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
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat LiDAR scans N times when --sensor lidar is selected (default: 1)",
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
        "voice": check_voice,
    }

    for name, check_fn in sensor_checks.items():
        if args.sensor in ("all", name):
            if name == "lidar":
                check_lidar(cfg, repeat=max(1, args.repeat))
            else:
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
