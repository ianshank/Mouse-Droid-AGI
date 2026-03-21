#!/usr/bin/env python3
"""Standalone sensor verification script for MouseDroid Jetson hardware.

Checks access to the three primary input sensors:
  - Video   : IMX500 camera via picamera2 (jetson_utils fallback)
  - Audio   : USB microphone via pyaudio / UsbMicrophone driver
  - Ultrasonic : HC-SR04 via Jetson.GPIO

Usage:
    python scripts/verify_sensors.py [--sensor {camera,audio,ultrasonic,all}]

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
        _fail("UsbMicrophone capture", str(exc))
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
    t0 = time.monotonic()
    try:
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size,
        )
        raw = stream.read(chunk_size, exception_on_overflow=False)
        stream.stop_stream()
        stream.close()
    except Exception as exc:
        pa.terminate()
        _fail("raw pyaudio capture", str(exc))
        return
    finally:
        pa.terminate()

    elapsed = time.monotonic() - t0
    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.shape != (chunk_size,):
        _fail("raw capture shape", f"expected ({chunk_size},), got {arr.shape}")
        return

    _ok("raw pyaudio capture", f"shape={arr.shape}, dtype={arr.dtype}, {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# 3. Ultrasonic — HC-SR04 / Jetson.GPIO
# ---------------------------------------------------------------------------

def check_ultrasonic() -> None:
    _section("Ultrasonic (HC-SR04)")

    # -- GPIO import
    try:
        import Jetson.GPIO  # noqa: F401
    except ImportError:
        _fail("Jetson.GPIO import", "library not installed — not running on Jetson?")
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
        "  python scripts/verify_sensors.py --sensor ultrasonic\n",
    )
    parser.add_argument(
        "--sensor",
        choices=["all", "camera", "audio", "ultrasonic"],
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

    # Summary
    _section("Summary")
    total = sum(
        1
        for s in ("camera", "audio", "ultrasonic")
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
