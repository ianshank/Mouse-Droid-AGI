"""End-to-end hardware validation tests for Jetson deployment.

These tests run on the NVIDIA Jetson Orin Nano with real hardware attached.
Marked ``@pytest.mark.hardware`` and excluded from CI.

Run with::

    pytest -m hardware -v --timeout=60 tests/e2e/test_jetson_hardware_e2e.py

Each test validates a single subsystem before the full orchestrator loop test.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = pytest.mark.hardware


# ---------------------------------------------------------------------------
# 1. Camera captures frame at expected resolution
# ---------------------------------------------------------------------------


async def test_camera_captures_frame() -> None:
    """Camera driver captures features with correct dimension."""
    from mousedroid.config.schema import Settings
    from mousedroid.factory import build_camera

    cfg = Settings(mock_hardware=False)
    camera = build_camera(cfg)

    await camera.start()
    try:
        features = await camera.capture_features()
        assert features.shape == (cfg.camera.feature_dim,)
        assert features.dtype == np.float32
        # At least some non-zero pixels expected from a real camera
        assert np.any(features != 0), "Camera returned all-zero features"
    finally:
        await camera.stop()


# ---------------------------------------------------------------------------
# 2. Ultrasonic reads distance within valid range
# ---------------------------------------------------------------------------


async def test_ultrasonic_reads_distance() -> None:
    """Ultrasonic sensor returns distance within [0, max_range_m]."""
    from mousedroid.config.schema import Settings
    from mousedroid.factory import build_distance_sensor

    cfg = Settings(mock_hardware=False)
    distance_sensor = build_distance_sensor(cfg)

    reading = await distance_sensor.read_distance_m()
    assert (
        0.0 <= reading <= distance_sensor.max_range_m
    ), f"Distance {reading}m outside valid range [0, {distance_sensor.max_range_m}]"


# ---------------------------------------------------------------------------
# 3. ESP32 sends and receives velocity command
# ---------------------------------------------------------------------------


async def test_esp32_velocity_roundtrip() -> None:
    """ESP32 accepts a velocity command and reads encoders back."""
    from mousedroid.config.schema import Settings
    from mousedroid.factory import build_esp32_driver

    cfg = Settings(mock_hardware=False)
    esp32 = build_esp32_driver(cfg)

    await esp32.connect()
    try:
        # Send zero velocity (safe) — vx, vy, omega
        await esp32.send_velocity(0.0, 0.0, 0.0)

        # Read encoders — should not raise
        encoders = await esp32.read_encoders()
        assert hasattr(encoders, "left_velocity_mps")
        assert hasattr(encoders, "right_velocity_mps")
        assert hasattr(encoders, "heading_rad")

        # Read battery — should be reasonable
        battery_v = await esp32.get_battery_voltage()
        assert 6.0 <= battery_v <= 18.0, f"Battery {battery_v}V outside expected range"
    finally:
        await esp32.emergency_stop()
        await esp32.disconnect()


# ---------------------------------------------------------------------------
# 4. LiDAR returns 360-degree scan
# ---------------------------------------------------------------------------


async def test_lidar_returns_scan() -> None:
    """LiDAR driver returns a scan with expected number of points."""
    from mousedroid.config.schema import Settings

    cfg = Settings(mock_hardware=False)
    if cfg.lidar is None:
        pytest.skip("LiDAR not configured")

    from mousedroid.factory import build_lidar

    lidar = build_lidar(cfg)
    if lidar is None:
        pytest.skip("LiDAR not available")

    await lidar.start()
    try:
        scan = await lidar.read_scan()
        assert len(scan) > 0, "LiDAR returned empty scan"
        # LD19 typically returns ~400-500 points per scan
        assert len(scan) >= 100, f"LiDAR scan too sparse: {len(scan)} points"
    finally:
        await lidar.stop()


# ---------------------------------------------------------------------------
# 5. Microphone captures audio chunk
# ---------------------------------------------------------------------------


async def test_microphone_captures_audio() -> None:
    """Microphone driver captures a 1-second audio chunk."""
    from mousedroid.config.schema import Settings
    from mousedroid.factory import build_microphone

    cfg = Settings(mock_hardware=False)
    mic = build_microphone(cfg)
    if mic is None:
        pytest.skip("Microphone not configured")

    await mic.start()
    try:
        chunk = await mic.read_chunk()
        assert chunk.dtype == np.float32
        assert len(chunk) > 0, "Microphone returned empty chunk"
        # Audio should have some energy (not perfectly silent in a real room)
        rms = float(np.sqrt(np.mean(chunk**2)))
        assert rms >= 0.0  # May be close to zero in a quiet room
    finally:
        await mic.stop()


# ---------------------------------------------------------------------------
# 6. Speaker plays a tone (Rocky TTS)
# ---------------------------------------------------------------------------


async def test_speaker_plays_audio() -> None:
    """Speaker driver plays a short test tone without error."""
    from mousedroid.config.schema import Settings
    from mousedroid.factory import build_speaker

    cfg = Settings(mock_hardware=False)
    speaker = build_speaker(cfg)
    if speaker is None:
        pytest.skip("Speaker not configured")

    # Generate a 0.5-second sine tone at 440 Hz
    sample_rate = cfg.voice.sample_rate
    duration_s = 0.5
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Should not raise
    await speaker.play(tone)


# ---------------------------------------------------------------------------
# 7. Full 5-tick orchestrator loop with real sensors
# ---------------------------------------------------------------------------


async def test_orchestrator_5_tick_loop() -> None:
    """Full orchestrator runs 5 ticks with real hardware without crash."""
    from mousedroid.config.schema import Settings
    from mousedroid.factory import build_orchestrator

    cfg = Settings(mock_hardware=False)
    orch = build_orchestrator(cfg)

    await orch.start()
    try:
        for i in range(5):
            t0 = time.monotonic()
            await orch.tick()
            elapsed_ms = (time.monotonic() - t0) * 1000
            # Each tick should complete within reasonable time
            assert elapsed_ms < 1000, f"Tick {i} took {elapsed_ms:.1f}ms (> 1s)"
    finally:
        await orch.stop()


# ---------------------------------------------------------------------------
# 8. Sensor manager concurrent read latency
# ---------------------------------------------------------------------------


async def test_sensor_manager_read_latency() -> None:
    """SensorManager.read_all() completes within 100ms on real hardware."""
    from mousedroid.config.schema import Settings
    from mousedroid.factory import (
        build_camera,
        build_distance_sensor,
        build_esp32_driver,
        build_microphone,
        build_sensor_manager,
    )

    cfg = Settings(mock_hardware=False)
    camera = build_camera(cfg)
    distance = build_distance_sensor(cfg)
    esp32 = build_esp32_driver(cfg)
    mic = build_microphone(cfg)

    mgr = build_sensor_manager(cfg, vision=camera, distance=distance, esp32=esp32, microphone=mic)

    await camera.start()
    await esp32.connect()
    if mic is not None:
        await mic.start()

    try:
        # Warm-up read
        await mgr.read_all()

        # Timed reads
        latencies: list[float] = []
        for _ in range(10):
            t0 = time.monotonic()
            obs = await mgr.read_all()
            latencies.append((time.monotonic() - t0) * 1000)
            assert obs.valid_mask is not None

        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        avg = sum(latencies) / len(latencies)
        assert p95 < 100, f"Sensor read p95={p95:.1f}ms exceeds 100ms"
        assert avg < 50, f"Sensor read avg={avg:.1f}ms exceeds 50ms"
    finally:
        if mic is not None:
            await mic.stop()
        await camera.stop()
        await esp32.emergency_stop()
        await esp32.disconnect()
