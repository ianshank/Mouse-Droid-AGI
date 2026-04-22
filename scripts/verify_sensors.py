#!/usr/bin/env python3
"""Standalone sensor verification script for MouseDroid Jetson hardware.

Checks access to the primary input and output devices:
  - Video      : IMX500 camera via picamera2 (jetson_utils fallback)
  - Audio      : USB microphone via pyaudio / UsbMicrophone driver
  - Ultrasonic : HC-SR04 via Jetson.GPIO
  - LiDAR      : FHL-LD19 via pyserial / LD19LidarDriver
  - Speaker    : USB speaker via pyaudio / UsbSpeaker driver

Usage:
    python scripts/verify_sensors.py \\
        [--sensor {camera,audio,ultrasonic,lidar,speaker,all}]

Exit code is non-zero if any sensor reports FAIL.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Make the src tree importable when run from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_FAILURES: list[str] = []

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _ok(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {_GREEN}[PASS]{_RESET} {label}{suffix}")


def _fail(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {_RED}[FAIL]{_RESET} {label}{suffix}")
    _FAILURES.append(label)


def _skip(label: str, reason: str = "") -> None:
    suffix = f"  ({reason})" if reason else ""
    print(f"  {_YELLOW}[SKIP]{_RESET} {label}{suffix}")


def _section(title: str) -> None:
    bar = "=" * (len(title) + 4)
    print(f"\n{bar}\n  {title}\n{bar}")


# ---------------------------------------------------------------------------
# 1. Video — IMX500 / picamera2
# ---------------------------------------------------------------------------

def check_camera() -> None:
    _section("Video (Camera)")

    # -- picamera2 import
    try:
        from picamera2 import Picamera2
    except ImportError:
        _skip("picamera2 import", "library not installed — trying jetson_utils fallback")
        _check_camera_jetson_utils()
        return

    _ok("picamera2 import")

    # -- open and capture
    t0 = time.monotonic()
    try:
        cam = Picamera2()
        cfg = cam.create_still_configuration(main={"size": (640, 480)})
        cam.configure(cfg)
        cam.start()
        time.sleep(0.5)  # let AE/AWB settle
        frame = cam.capture_array()
        cam.stop()
        cam.close()
    except Exception as exc:
        _fail("camera open/capture", str(exc))
        return

    elapsed = time.monotonic() - t0
    h, w = frame.shape[0], frame.shape[1]
    if h != 480 or w != 640:
        _fail("frame shape", f"expected (480, 640), got ({h}, {w})")
        return

    _ok("frame capture", f"{w}x{h} in {elapsed:.2f}s, dtype={frame.dtype}")


def _check_camera_jetson_utils() -> None:
    try:
        import jetson_utils
    except ImportError:
        _fail("camera access", "neither picamera2 nor jetson_utils is installed")
        return

    _ok("jetson_utils import")
    t0 = time.monotonic()
    try:
        cam = jetson_utils.videoSource(
            "csi://0", argv=["--input-width=640", "--input-height=480"]
        )
        cuda_img = cam.Capture()
        frame = jetson_utils.cudaToNumpy(cuda_img) if cuda_img is not None else None
    except Exception as exc:
        _fail("jetson_utils camera capture", str(exc))
        return

    if frame is None:
        _fail("jetson_utils frame", "Capture() returned None")
        return

    elapsed = time.monotonic() - t0
    h, w = frame.shape[0], frame.shape[1]
    if h != 480 or w != 640:
        _fail("frame shape", f"expected (480, 640), got ({h}, {w})")
        return

    _ok("frame capture (jetson_utils)", f"{w}x{h} in {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# 2. Audio — USB microphone / pyaudio
# ---------------------------------------------------------------------------

def check_audio() -> None:
    _section("Audio (USB Microphone)")

    # -- pyaudio import
    try:
        import pyaudio
    except ImportError:
        _fail("pyaudio import", "library not installed")
        return

    _ok("pyaudio import")

    # -- enumerate input devices
    pa = pyaudio.PyAudio()
    input_devices: list[str] = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            input_devices.append(f"[{i}] {info['name']}")
    pa.terminate()

    if not input_devices:
        _fail("USB mic detected", "no input devices found — microphone not connected?")
        return

    _ok("input device(s) found", "; ".join(input_devices))

    # -- driver-level capture (uses UsbMicrophone if available, else raw pyaudio)
    asyncio.run(_async_capture_check())


async def _async_capture_check() -> None:
    try:
        from mousedroid.config.schema import MicrophoneConfig
        from mousedroid.hardware.audio.usb_microphone import UsbMicrophone
    except ImportError:
        _skip(
            "UsbMicrophone driver",
            "mousedroid package not importable — falling back to raw capture",
        )
        _raw_pyaudio_capture()
        return

    cfg = MicrophoneConfig(chunk_size=512, sample_rate=16000, channels=1)
    mic = UsbMicrophone(cfg)
    t0 = time.monotonic()
    try:
        await mic.start()
        chunk = await mic.read_chunk()
        await mic.stop()
    except Exception as exc:
        # Device not found or unavailable — fall back to raw pyaudio so dev
        # environments without the target USB mic still exercise the path.
        _skip("UsbMicrophone driver", f"driver unavailable: {exc} — trying raw capture")
        _raw_pyaudio_capture()
        return

    elapsed = time.monotonic() - t0
    import numpy as np

    if chunk.dtype != np.float32:
        _fail("chunk dtype", f"expected float32, got {chunk.dtype}")
        return
    if chunk.shape != (512,):
        _fail("chunk shape", f"expected (512,), got {chunk.shape}")
        return

    _ok("UsbMicrophone chunk capture", f"shape={chunk.shape}, dtype={chunk.dtype}, {elapsed:.2f}s")


def _raw_pyaudio_capture() -> None:
    import numpy as np
    import pyaudio

    chunk_size = 512
    sample_rate = 16000

    pa = pyaudio.PyAudio()

    # Collect all usable input device indices.
    input_indices = [
        i
        for i in range(pa.get_device_count())
        if pa.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
    ]

    if not input_indices:
        pa.terminate()
        _fail("raw pyaudio capture", "no input device found")
        return

    last_exc: Exception | None = None
    raw: bytes | None = None
    used_index: int | None = None
    used_fmt: str = "float32"
    t0 = time.monotonic()

    # Try each device with paFloat32, then paInt16 fallback.
    for idx in input_indices:
        for fmt, fmt_name in (
            (pyaudio.paFloat32, "float32"),
            (pyaudio.paInt16, "int16"),
        ):
            try:
                t0 = time.monotonic()
                stream = pa.open(
                    format=fmt,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    input_device_index=idx,
                    frames_per_buffer=chunk_size,
                )
                raw = stream.read(chunk_size, exception_on_overflow=False)
                stream.stop_stream()
                stream.close()
                used_index = idx
                used_fmt = fmt_name
                break
            except Exception as exc:
                last_exc = exc
        if raw is not None:
            break

    # Save the device name before terminating PyAudio.
    device_name = (
        pa.get_device_info_by_index(used_index)["name"]
        if used_index is not None
        else "?"
    )
    pa.terminate()

    if raw is None:
        _fail("raw pyaudio capture", f"all input devices failed; last error: {last_exc}")
        return

    elapsed = time.monotonic() - t0
    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.shape != (chunk_size,):
        _fail("raw capture shape", f"expected ({chunk_size},), got {arr.shape}")
        return

    detail = (
        f"device=[{used_index}]{device_name}, "
        f"fmt={used_fmt}, shape={arr.shape}, {elapsed:.2f}s"
    )
    _ok("raw pyaudio capture", detail)


# ---------------------------------------------------------------------------
# 3. Ultrasonic — HC-SR04 / Jetson.GPIO
# ---------------------------------------------------------------------------

def check_ultrasonic() -> None:
    _section("Ultrasonic (HC-SR04)")

    # -- GPIO import (Jetson.GPIO raises non-ImportError at load time when
    #    it cannot determine the Jetson model, e.g. inside some containers).
    try:
        import Jetson.GPIO  # noqa: F401
    except ImportError:
        _fail("Jetson.GPIO import", "library not installed — not running on Jetson?")
        return
    except Exception as exc:
        _fail("Jetson.GPIO import", f"library failed to initialize: {exc}")
        return

    _ok("Jetson.GPIO import")

    # -- driver-level distance read
    asyncio.run(_async_ultrasonic_check())


async def _async_ultrasonic_check() -> None:
    try:
        from mousedroid.config.schema import UltrasonicConfig
        from mousedroid.hardware.sensors.ultrasonic import HcSr04
    except ImportError:
        _skip("HcSr04 driver", "mousedroid package not importable — falling back to raw GPIO")
        _raw_gpio_distance_check()
        return

    cfg = UltrasonicConfig(trigger_pin=23, echo_pin=24)
    sensor = HcSr04(cfg)
    t0 = time.monotonic()
    try:
        await sensor.start()
        distance_m = await sensor.read_distance_m()
        await sensor.stop()
    except Exception as exc:
        _fail("HcSr04 distance read", str(exc))
        return

    elapsed = time.monotonic() - t0

    if not isinstance(distance_m, float):
        _fail("distance type", f"expected float, got {type(distance_m).__name__}")
        return
    if not (0.0 <= distance_m <= cfg.max_range_m):
        _fail("distance range", f"{distance_m:.3f} m outside [0, {cfg.max_range_m}] m")
        return

    _ok("HcSr04 distance read", f"{distance_m:.3f} m in {elapsed:.2f}s")


def _raw_gpio_distance_check() -> None:
    """Low-level GPIO pulse measurement without the driver layer."""
    import Jetson.GPIO as GPIO

    trigger_pin = 23
    echo_pin = 24
    max_range_m = 4.0
    speed_of_sound = 343.0  # m/s at 20 °C

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(trigger_pin, GPIO.OUT)
    GPIO.setup(echo_pin, GPIO.IN)
    GPIO.output(trigger_pin, False)
    time.sleep(0.05)

    try:
        # Send 10 µs trigger pulse
        GPIO.output(trigger_pin, True)
        time.sleep(0.00001)
        GPIO.output(trigger_pin, False)

        t0 = time.monotonic()
        deadline = t0 + 0.1
        pulse_start = pulse_end = t0

        # Wait for echo to go HIGH
        while GPIO.input(echo_pin) == 0:
            pulse_start = time.monotonic()
            if pulse_start > deadline:
                _fail("raw GPIO echo start", "timeout waiting for echo HIGH")
                return

        # Wait for echo to go LOW
        while GPIO.input(echo_pin) == 1:
            pulse_end = time.monotonic()
            if pulse_end > deadline:
                _fail("raw GPIO echo end", "timeout waiting for echo LOW")
                return

        pulse_duration = pulse_end - pulse_start
        distance_m = (pulse_duration * speed_of_sound) / 2.0

        if not (0.0 <= distance_m <= max_range_m):
            _fail("raw GPIO distance", f"{distance_m:.3f} m outside [0, {max_range_m}] m")
            return

        _ok("raw GPIO distance pulse", f"{distance_m:.3f} m")

    finally:
        GPIO.cleanup()


# ---------------------------------------------------------------------------
# 4. LiDAR — FHL-LD19 / pyserial
# ---------------------------------------------------------------------------

def check_lidar() -> None:
    _section("LiDAR (FHL-LD19)")

    try:
        import serial  # noqa: F401
    except ImportError:
        _fail("pyserial import", "library not installed")
        return

    _ok("pyserial import")

    try:
        from mousedroid.config.schema import LidarConfig
        from mousedroid.hardware.lidar.ld19_driver import LD19LidarDriver
    except ImportError as exc:
        _fail("LD19LidarDriver import", f"mousedroid package not importable: {exc}")
        return

    cfg = LidarConfig()
    asyncio.run(_async_lidar_check(cfg, LD19LidarDriver))


async def _async_lidar_check(cfg: object, driver_cls: object) -> None:
    driver = driver_cls(cfg)  # type: ignore[operator]
    t0 = time.monotonic()
    try:
        try:
            await driver.start()
        except Exception as exc:
            _skip(
                "LD19 serial open",
                f"port {cfg.serial_port} unavailable: {exc}",  # type: ignore[attr-defined]
            )
            return

        try:
            _ok("LD19 serial open", f"port={cfg.serial_port}")  # type: ignore[attr-defined]
            scan = await driver.read_scan()
        finally:
            await driver.stop()
    except Exception as exc:
        _fail("LD19 read_scan", str(exc))
        return

    elapsed = time.monotonic() - t0
    if scan.n_points <= 0:
        _fail("LD19 scan content", "no points returned — is the LiDAR spinning?")
        return

    import numpy as np

    angles = scan.angles_deg
    distances_m = scan.distances_mm / 1000.0
    angle_span = float(np.asarray(angles).max() - np.asarray(angles).min())

    if angle_span < 270.0:
        _fail("LD19 angle coverage", f"{angle_span:.1f}° below 270° threshold")
        return

    dmin = float(np.asarray(distances_m).min())
    dmax = float(np.asarray(distances_m).max())
    if dmin < cfg.min_range_m or dmax > cfg.max_range_m:  # type: ignore[attr-defined]
        _fail(
            "LD19 distance range",
            f"[{dmin:.3f}, {dmax:.3f}] m outside "
            f"[{cfg.min_range_m}, {cfg.max_range_m}] m",  # type: ignore[attr-defined]
        )
        return

    _ok(
        "LD19 scan",
        f"n={scan.n_points}, span={angle_span:.1f}°, "
        f"dist=[{dmin:.2f}, {dmax:.2f}] m in {elapsed:.2f}s",
    )


# ---------------------------------------------------------------------------
# 5. Speaker — USB speaker / pyaudio
# ---------------------------------------------------------------------------

def check_speaker() -> None:
    _section("Audio Output (USB Speaker)")

    try:
        import pyaudio
    except ImportError:
        _fail("pyaudio import", "library not installed")
        return

    _ok("pyaudio import")

    try:
        from mousedroid.config.schema import SpeakerConfig
        from mousedroid.hardware.audio.usb_speaker import UsbSpeaker
    except ImportError as exc:
        _fail("UsbSpeaker import", f"mousedroid package not importable: {exc}")
        return

    cfg = SpeakerConfig()
    needle = cfg.device_name.lower()

    pa = pyaudio.PyAudio()
    matches: list[str] = []
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = str(info.get("name", ""))
            if int(info.get("maxOutputChannels", 0)) > 0 and needle in name.lower():
                matches.append(f"[{i}] {name}")
    finally:
        pa.terminate()

    if not matches:
        _skip(
            "USB speaker detected",
            f"no output device matching '{cfg.device_name}'",
        )
        return

    _ok("USB speaker detected", "; ".join(matches))

    asyncio.run(_async_speaker_check(cfg, UsbSpeaker))


async def _async_speaker_check(cfg: object, speaker_cls: object) -> None:
    import numpy as np

    speaker = speaker_cls(cfg)  # type: ignore[operator]
    t0 = time.monotonic()
    try:
        try:
            await speaker.start()
        except Exception as exc:
            _skip("UsbSpeaker start", f"{exc}")
            return

        if getattr(speaker, "_stream", None) is None:
            _skip(
                "UsbSpeaker stream",
                f"device '{cfg.device_name}' not opened (graceful no-op)",  # type: ignore[attr-defined]
            )
            await speaker.stop()
            return

        try:
            freq_hz = 440.0
            duration_chunks = 4
            chunk_size = cfg.chunk_size  # type: ignore[attr-defined]
            sample_rate = cfg.sample_rate  # type: ignore[attr-defined]
            total = chunk_size * duration_chunks
            t = np.arange(total, dtype=np.float32) / float(sample_rate)
            tone = (0.1 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)

            for start in range(0, total, chunk_size):
                chunk = tone[start : start + chunk_size]
                await speaker.write_chunk(chunk)
        finally:
            await speaker.stop()
    except Exception as exc:
        _fail("UsbSpeaker write_chunk", str(exc))
        return

    elapsed = time.monotonic() - t0
    _ok("UsbSpeaker sine tone", f"440 Hz, 4 chunks in {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify access to MouseDroid hardware sensors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  python scripts/verify_sensors.py\n"
        "  python scripts/verify_sensors.py --sensor camera\n"
        "  python scripts/verify_sensors.py --sensor audio\n"
        "  python scripts/verify_sensors.py --sensor ultrasonic\n"
        "  python scripts/verify_sensors.py --sensor lidar\n"
        "  python scripts/verify_sensors.py --sensor speaker\n",
    )
    parser.add_argument(
        "--sensor",
        choices=["all", "camera", "audio", "ultrasonic", "lidar", "speaker"],
        default="all",
        help="Which sensor to verify (default: all)",
    )
    args = parser.parse_args()

    print("\nMouseDroid Sensor Verification")
    print(f"Running on Python {sys.version.split()[0]}")
    print(f"Project root: {_PROJECT_ROOT}")

    if args.sensor in ("all", "camera"):
        check_camera()
    if args.sensor in ("all", "audio"):
        check_audio()
    if args.sensor in ("all", "ultrasonic"):
        check_ultrasonic()
    if args.sensor in ("all", "lidar"):
        check_lidar()
    if args.sensor in ("all", "speaker"):
        check_speaker()

    # Summary
    _section("Summary")
    total = sum(
        1
        for s in ("camera", "audio", "ultrasonic", "lidar", "speaker")
        if args.sensor in ("all", s)
    )
    failed = len(_FAILURES)
    passed = total - failed

    print(f"  {passed}/{total} sensors OK")
    if _FAILURES:
        print(f"  {_RED}Failed:{_RESET} {', '.join(_FAILURES)}")
        sys.exit(1)
    else:
        print(f"  {_GREEN}All sensors accessible.{_RESET}")


if __name__ == "__main__":
    main()
