#!/usr/bin/env python3
"""MouseDroid sensor streaming server.

Runs on the Jetson Nano and streams live sensor data to any browser on the LAN.

  Camera   → MJPEG stream at  /stream/camera
  Ultrasonic → Server-Sent Events at /stream/ultrasonic
  Dashboard  → Single-page UI at  /

Usage:
    python scripts/sensor_stream_server.py [--host 0.0.0.0] [--port 8765]

Then open  http://<jetson-ip>:8765  on this Windows machine.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

# Make the src tree importable when run from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="MouseDroid Sensor Stream")

# ---------------------------------------------------------------------------
# Camera — MJPEG multipart stream
# ---------------------------------------------------------------------------

async def _camera_frames() -> AsyncIterator[bytes]:
    """Yield MJPEG boundary frames as fast as the camera allows (~30 fps cap)."""
    try:
        from picamera2 import Picamera2

        cam = Picamera2()
        cfg = cam.create_video_configuration(main={"size": (640, 480)})
        cam.configure(cfg)
        cam.start()
        await asyncio.sleep(0.5)  # let AE settle

        try:
            while True:
                frame = cam.capture_array()
                # Encode to JPEG using PIL (always available with picamera2)
                from PIL import Image

                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80)
                jpg = buf.getvalue()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
                await asyncio.sleep(1 / 30)  # cap at 30 fps
        finally:
            cam.stop()
            cam.close()

    except ImportError:
        # jetson_utils fallback
        try:
            import jetson_utils
            from PIL import Image

            cam = jetson_utils.videoSource(
                "csi://0", argv=["--input-width=640", "--input-height=480"]
            )
            while True:
                cuda_img = cam.Capture()
                if cuda_img is None:
                    await asyncio.sleep(0.033)
                    continue
                frame = jetson_utils.cudaToNumpy(cuda_img)
                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80)
                jpg = buf.getvalue()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
                await asyncio.sleep(1 / 30)
        except ImportError:
            log.error("No camera library available (picamera2 or jetson_utils)")
            # Send a single "unavailable" placeholder frame
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (640, 80), color=(40, 40, 40))
            draw = ImageDraw.Draw(img)
            draw.text((10, 25), "Camera unavailable", fill=(200, 60, 60))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.getvalue() + b"\r\n"
            )


@app.get("/stream/camera")
async def camera_stream() -> StreamingResponse:
    return StreamingResponse(
        _camera_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Ultrasonic — Server-Sent Events
# ---------------------------------------------------------------------------

async def _ultrasonic_events() -> AsyncIterator[str]:
    """Yield SSE distance readings at ~10 Hz."""
    try:
        from mousedroid.config.schema import UltrasonicConfig
        from mousedroid.hardware.sensors.ultrasonic import HcSr04

        cfg = UltrasonicConfig(trigger_pin=23, echo_pin=24)
        sensor = HcSr04(cfg)
        await sensor.start()
        try:
            while True:
                try:
                    dist = await asyncio.wait_for(sensor.read_distance_m(), timeout=0.2)
                    yield f"data: {dist:.4f}\n\n"
                except asyncio.TimeoutError:
                    yield "data: timeout\n\n"
                except Exception as exc:
                    yield f"data: error:{exc}\n\n"
                await asyncio.sleep(0.1)
        finally:
            await sensor.stop()

    except ImportError:
        # Raw GPIO fallback
        try:
            import Jetson.GPIO as GPIO

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(23, GPIO.OUT)
            GPIO.setup(24, GPIO.IN)
            try:
                while True:
                    dist = await asyncio.get_event_loop().run_in_executor(
                        None, _raw_gpio_pulse
                    )
                    yield f"data: {dist:.4f}\n\n"
                    await asyncio.sleep(0.1)
            finally:
                GPIO.cleanup()
        except ImportError:
            log.error("Jetson.GPIO not available")
            while True:
                yield "data: unavailable\n\n"
                await asyncio.sleep(1.0)


def _raw_gpio_pulse() -> float:
    """Blocking GPIO pulse measurement; runs in executor thread."""
    import Jetson.GPIO as GPIO

    speed = 343.0
    GPIO.output(23, True)
    time.sleep(0.00001)
    GPIO.output(23, False)

    deadline = time.monotonic() + 0.1
    while GPIO.input(24) == 0:
        pulse_start = time.monotonic()
        if pulse_start > deadline:
            return -1.0
    while GPIO.input(24) == 1:
        pulse_end = time.monotonic()
        if pulse_end > deadline:
            return -1.0
    return ((pulse_end - pulse_start) * speed) / 2.0


@app.get("/stream/ultrasonic")
async def ultrasonic_stream() -> StreamingResponse:
    return StreamingResponse(
        _ultrasonic_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Audio level — Server-Sent Events (peak amplitude per chunk)
# ---------------------------------------------------------------------------

async def _audio_events() -> AsyncIterator[str]:
    """Yield SSE peak audio levels (0.0-1.0) at ~10 Hz."""
    import numpy as np

    try:
        from mousedroid.config.schema import MicrophoneConfig
        from mousedroid.hardware.audio.usb_microphone import UsbMicrophone

        cfg = MicrophoneConfig(chunk_size=512, sample_rate=16000, channels=1)
        mic = UsbMicrophone(cfg)
        await mic.start()
        try:
            while True:
                chunk = await mic.read_chunk()
                peak = float(np.abs(chunk).max())
                yield f"data: {peak:.6f}\n\n"
                await asyncio.sleep(0.05)
        finally:
            await mic.stop()

    except ImportError:
        # Raw pyaudio fallback
        try:
            import pyaudio

            pa = pyaudio.PyAudio()
            input_idx = next(
                (
                    i
                    for i in range(pa.get_device_count())
                    if pa.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
                ),
                None,
            )
            if input_idx is None:
                pa.terminate()
                while True:
                    yield "data: unavailable\n\n"
                    await asyncio.sleep(1.0)
                return

            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=input_idx,
                frames_per_buffer=512,
            )
            try:
                while True:
                    raw = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: stream.read(512, exception_on_overflow=False)
                    )
                    arr = np.frombuffer(raw, dtype=np.float32)
                    peak = float(np.abs(arr).max())
                    yield f"data: {peak:.6f}\n\n"
                    await asyncio.sleep(0.05)
            finally:
                stream.stop_stream()
                stream.close()
                pa.terminate()

        except ImportError:
            while True:
                yield "data: unavailable\n\n"
                await asyncio.sleep(1.0)


@app.get("/stream/audio")
async def audio_stream() -> StreamingResponse:
    return StreamingResponse(
        _audio_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MouseDroid Sensor Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #111; color: #eee; font-family: monospace; padding: 1rem; }
  h1 { color: #4af; margin-bottom: 1rem; font-size: 1.4rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .card { background: #1e1e1e; border: 1px solid #333; border-radius: 6px; padding: 1rem; }
  .card h2 { color: #8cf; font-size: 1rem; margin-bottom: 0.5rem; }
  .card.full { grid-column: 1 / -1; }
  img#camera { width: 100%; border-radius: 4px; background: #000; min-height: 200px; }
  .value { font-size: 2rem; color: #4f4; margin: 0.5rem 0; }
  .value.warn { color: #fa4; }
  .value.err  { color: #f44; }
  .unit { font-size: 0.8rem; color: #888; }
  #audio-bar-wrap { background: #333; border-radius: 3px; height: 20px; margin-top: 0.5rem; }
  #audio-bar { background: #4f4; height: 20px; border-radius: 3px; width: 0%; transition: width 0.05s; }
  .status { font-size: 0.75rem; color: #666; margin-top: 0.3rem; }
</style>
</head>
<body>
<h1>&#128287; MouseDroid Sensor Dashboard</h1>
<div class="grid">

  <div class="card full">
    <h2>&#127909; Camera (IMX500 / picamera2)</h2>
    <img id="camera" src="/stream/camera" alt="camera stream">
    <div class="status" id="cam-status">Connecting…</div>
  </div>

  <div class="card">
    <h2>&#128308; Ultrasonic (HC-SR04)</h2>
    <div class="value" id="dist-val">--</div>
    <div class="unit">metres</div>
    <div class="status" id="dist-status">Connecting…</div>
  </div>

  <div class="card">
    <h2>&#127908; Microphone (USB)</h2>
    <div class="value" id="audio-val">--</div>
    <div class="unit">peak amplitude</div>
    <div id="audio-bar-wrap"><div id="audio-bar"></div></div>
    <div class="status" id="audio-status">Connecting…</div>
  </div>

</div>
<script>
  // Camera — just an img tag; browser handles MJPEG natively
  const camImg = document.getElementById('camera');
  camImg.onload  = () => { document.getElementById('cam-status').textContent = 'streaming'; };
  camImg.onerror = () => { document.getElementById('cam-status').textContent = 'stream error'; };

  // Ultrasonic SSE
  const sse_us = new EventSource('/stream/ultrasonic');
  const distEl = document.getElementById('dist-val');
  const distSt = document.getElementById('dist-status');
  sse_us.onmessage = (e) => {
    const v = e.data;
    distSt.textContent = new Date().toLocaleTimeString();
    if (v === 'unavailable') { distEl.textContent = 'N/A'; distEl.className = 'value err'; return; }
    if (v === 'timeout')     { distEl.textContent = 'timeout'; distEl.className = 'value warn'; return; }
    if (v.startsWith('error')) { distEl.textContent = v; distEl.className = 'value err'; return; }
    const m = parseFloat(v);
    distEl.textContent = isNaN(m) ? v : m.toFixed(3);
    distEl.className = m > 0 ? 'value' : 'value warn';
  };
  sse_us.onerror = () => { distSt.textContent = 'SSE error'; };

  // Audio SSE
  const sse_au = new EventSource('/stream/audio');
  const audioEl = document.getElementById('audio-val');
  const audioBar = document.getElementById('audio-bar');
  const audioSt = document.getElementById('audio-status');
  sse_au.onmessage = (e) => {
    const v = e.data;
    audioSt.textContent = new Date().toLocaleTimeString();
    if (v === 'unavailable') { audioEl.textContent = 'N/A'; audioEl.className = 'value err'; return; }
    const p = parseFloat(v);
    if (!isNaN(p)) {
      audioEl.textContent = p.toFixed(4);
      audioEl.className = 'value';
      audioBar.style.width = Math.min(p * 100, 100) + '%';
      audioBar.style.background = p > 0.8 ? '#f44' : p > 0.4 ? '#fa4' : '#4f4';
    }
  };
  sse_au.onerror = () => { audioSt.textContent = 'SSE error'; };
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(_DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MouseDroid sensor streaming server (run on Jetson)."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    args = parser.parse_args()

    log.info("Starting MouseDroid sensor stream server on %s:%d", args.host, args.port)
    log.info("Open  http://<jetson-ip>:%d  in your browser", args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
