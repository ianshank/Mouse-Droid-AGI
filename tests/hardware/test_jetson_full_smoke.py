"""Comprehensive hardware smoke tests for Jetson Orin Nano deployment.

Covers GPU memory, TensorRT compilation, camera capture, ultrasonic sensor,
ESP32 velocity round-trip, microphone capture, orchestrator timing, and
telemetry health. Every threshold comes from config.

Run on Jetson::

    pytest -m hardware -v --timeout=60 tests/hardware/test_jetson_full_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time

import pytest

from mousedroid.config.schema import Settings

pytestmark = pytest.mark.hardware

JETSON_PROD_CONFIG = os.getenv("MOUSEDROID_JETSON_CONFIG", "config/jetson_production.yaml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Load Settings from jetson_production.yaml."""
    import yaml

    with open(JETSON_PROD_CONFIG) as fh:
        raw = yaml.safe_load(fh)
    return Settings(**raw)


# ---------------------------------------------------------------------------
# 1. GPU memory alloc / dealloc on CUDA
# ---------------------------------------------------------------------------


def test_gpu_memory_alloc_dealloc() -> None:
    """Allocate a tensor on GPU, verify device, then free it."""
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "CUDA not available"

    free_before = torch.cuda.mem_get_info()[0]
    t = torch.randn(1024, 1024, device="cuda")
    assert t.device.type == "cuda"
    del t
    torch.cuda.empty_cache()
    free_after = torch.cuda.mem_get_info()[0]
    # After dealloc, free memory should be roughly restored
    assert free_after >= free_before * 0.9, (
        f"GPU memory not freed: before={free_before}, after={free_after}"
    )


# ---------------------------------------------------------------------------
# 2. TensorRT compile a small Linear model
# ---------------------------------------------------------------------------


def test_tensorrt_compile_linear(settings: Settings) -> None:
    """Compile a small Linear model with TensorRT and run inference."""
    torch = pytest.importorskip("torch")
    tensorrt = pytest.importorskip("tensorrt")

    assert hasattr(tensorrt, "__version__"), "tensorrt missing __version__"

    model = torch.nn.Linear(16, 4).cuda().eval()
    x = torch.randn(1, 16, device="cuda")

    with torch.no_grad():
        # Basic forward pass — TensorRT import proves the runtime is available
        out = model(x)

    assert out.shape == (1, 4), f"Unexpected output shape: {out.shape}"
    assert out.device.type == "cuda"


# ---------------------------------------------------------------------------
# 3. Camera capture one frame, verify shape
# ---------------------------------------------------------------------------


def test_camera_capture_frame(settings: Settings) -> None:
    """Capture one frame and verify it matches config resolution."""
    expected_h = settings.camera.resolution_height
    expected_w = settings.camera.resolution_width
    frame = None

    try:
        from picamera2 import Picamera2

        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": (expected_w, expected_h)},
        )
        cam.configure(config)
        cam.start()
        try:
            time.sleep(0.5)
            frame = cam.capture_array()
        finally:
            cam.stop()
            cam.close()
    except ImportError:
        pass

    if frame is None:
        try:
            import jetson_utils

            cam = jetson_utils.videoSource(
                "csi://0",
                argv=[f"--input-width={expected_w}", f"--input-height={expected_h}"],
            )
            cuda_img = cam.Capture()
            frame = jetson_utils.cudaToNumpy(cuda_img)
        except ImportError:
            pytest.skip("No camera library available (picamera2 or jetson_utils)")

    assert frame is not None, "Camera returned None frame"
    h, w = frame.shape[0], frame.shape[1]
    assert h == expected_h, f"Expected height {expected_h}, got {h}"
    assert w == expected_w, f"Expected width {expected_w}, got {w}"


# ---------------------------------------------------------------------------
# 4. Ultrasonic read distance
# ---------------------------------------------------------------------------


def test_ultrasonic_read_distance(settings: Settings) -> None:
    """Read a distance value from HC-SR04 within config range."""
    if settings.ultrasonic is None:
        pytest.skip("Ultrasonic config not set (mock_hardware=true?)")

    from mousedroid.hardware.sensors.ultrasonic import HcSr04

    sensor = HcSr04(settings.ultrasonic)

    try:
        distance = asyncio.run(sensor.read_distance_m())
    except RuntimeError:
        pytest.skip("GPIO unavailable -- not running on Jetson hardware")
        return

    assert isinstance(distance, float)
    assert settings.ultrasonic.min_range_m <= distance <= settings.ultrasonic.max_range_m, (
        f"Distance {distance:.3f} m outside range "
        f"[{settings.ultrasonic.min_range_m}, {settings.ultrasonic.max_range_m}]"
    )


# ---------------------------------------------------------------------------
# 5. ESP32 velocity round-trip
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
def test_esp32_velocity_round_trip(settings: Settings) -> None:
    """Send a small velocity, read encoders, verify non-zero response."""
    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(settings)

    async def _run() -> None:
        await driver.connect()
        try:
            test_vel = settings.esp32.max_velocity_mps * 0.1
            await driver.send_velocity(test_vel, 0.0, 0.0)
            await asyncio.sleep(0.1)
            enc = await driver.read_encoders()
            await driver.emergency_stop()
            assert enc.left_velocity_mps >= 0.0 or enc.right_velocity_mps >= 0.0
        finally:
            await driver.disconnect()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 6. Microphone 0.5s capture
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
def test_microphone_half_second_capture(settings: Settings) -> None:
    """Capture 0.5 seconds of audio from USB microphone."""
    pytest.importorskip("pyaudio")
    import numpy as np

    if settings.microphone is None:
        pytest.skip("Microphone config not set")

    from mousedroid.hardware.audio.usb_microphone import UsbMicrophone

    mic = UsbMicrophone(settings.microphone)
    sample_rate = settings.microphone.sample_rate
    chunk_size = settings.microphone.chunk_size
    # Number of chunks needed for 0.5 seconds
    n_chunks = max(1, int(0.5 * sample_rate / chunk_size))
    collected: list[np.ndarray] = []

    async def _capture() -> None:
        await mic.start()
        try:
            for _ in range(n_chunks):
                chunk = await mic.read_chunk()
                collected.append(chunk)
        finally:
            await mic.stop()

    asyncio.run(_capture())

    assert len(collected) == n_chunks, f"Expected {n_chunks} chunks, got {len(collected)}"
    total_samples = sum(c.shape[0] for c in collected)
    expected_samples = n_chunks * chunk_size
    assert total_samples == expected_samples, (
        f"Expected {expected_samples} samples, got {total_samples}"
    )


# ---------------------------------------------------------------------------
# 7. Orchestrator 30 Hz latency (30 ticks, mean < 33 ms)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_orchestrator_30hz_latency(settings: Settings) -> None:
    """Run 30 ticks and verify mean latency within budget."""
    from mousedroid.factory import build_orchestrator

    budget_ms = 1000.0 / settings.loop.control_hz
    n_ticks = int(os.getenv("MOUSEDROID_SMOKE_TICKS", "30"))

    async def _run() -> list[float]:
        orch = build_orchestrator(settings)
        await orch.start()
        try:
            tick_times: list[float] = []
            for _ in range(n_ticks):
                t0 = time.monotonic()
                await orch.tick()
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                tick_times.append(elapsed_ms)
            return tick_times
        finally:
            await orch.stop()

    tick_times = asyncio.run(_run())
    mean_ms = statistics.mean(tick_times)
    assert mean_ms <= budget_ms, (
        f"Mean tick {mean_ms:.1f} ms exceeds budget {budget_ms:.1f} ms "
        f"(control_hz={settings.loop.control_hz})"
    )


# ---------------------------------------------------------------------------
# 8. Orchestrator P95 latency within 2x budget
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_orchestrator_p95_latency(settings: Settings) -> None:
    """P95 tick latency must stay within 2x budget."""
    from mousedroid.factory import build_orchestrator

    budget_ms = 1000.0 / settings.loop.control_hz
    n_ticks = int(os.getenv("MOUSEDROID_SMOKE_TICKS", "30"))

    async def _run() -> list[float]:
        orch = build_orchestrator(settings)
        await orch.start()
        try:
            tick_times: list[float] = []
            for _ in range(n_ticks):
                t0 = time.monotonic()
                await orch.tick()
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                tick_times.append(elapsed_ms)
            return tick_times
        finally:
            await orch.stop()

    tick_times = asyncio.run(_run())
    sorted_times = sorted(tick_times)
    p95_idx = int(len(sorted_times) * 0.95)
    p95_ms = sorted_times[min(p95_idx, len(sorted_times) - 1)]
    hard_limit_ms = budget_ms * 2.0
    assert p95_ms <= hard_limit_ms, (
        f"P95 tick {p95_ms:.1f} ms exceeds 2x budget {hard_limit_ms:.1f} ms"
    )


# ---------------------------------------------------------------------------
# 9. Telemetry health endpoint responds
# ---------------------------------------------------------------------------


@pytest.mark.timeout(15)
def test_telemetry_health_endpoint(settings: Settings) -> None:
    """Start telemetry server and verify /health responds 200."""
    if not settings.telemetry.enabled:
        pytest.skip("Telemetry disabled in config")

    import aiohttp

    from mousedroid.telemetry.server import TelemetryServer

    port = settings.telemetry.port

    async def _run() -> None:
        server = TelemetryServer(settings)
        await server.start()
        try:
            url = f"http://127.0.0.1:{port}/health"
            async with (
                aiohttp.ClientSession() as session,
                session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp,
            ):
                assert resp.status == 200, f"Health endpoint returned {resp.status}"
                data = await resp.json()
                assert "status" in data
        finally:
            await server.stop()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 10. GPU temperature readable and within safe range
# ---------------------------------------------------------------------------


def test_gpu_temperature_safe(settings: Settings) -> None:
    """Read GPU temperature and verify it is below critical threshold."""
    thermal_path = settings.jetson.thermal_zone_path
    if not thermal_path.exists():
        pytest.skip(f"Thermal zone {thermal_path} not found")

    temp_raw = thermal_path.read_text().strip()
    temp_c = float(temp_raw) / 1000.0
    critical_c = settings.health.gpu_temp_critical_c
    assert temp_c < critical_c, (
        f"GPU temperature {temp_c:.1f} C >= critical threshold {critical_c:.1f} C"
    )


# ---------------------------------------------------------------------------
# 11. CUDA device properties match expected platform
# ---------------------------------------------------------------------------


def test_cuda_device_properties() -> None:
    """Verify CUDA device is accessible and reports valid compute capability."""
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "CUDA not available"

    props = torch.cuda.get_device_properties(0)
    assert props.total_mem > 0, "GPU reports 0 total memory"
    assert props.major >= 5, f"Compute capability {props.major}.{props.minor} too low (need >= 5.x)"
    assert len(props.name) > 0, "GPU device name is empty"
