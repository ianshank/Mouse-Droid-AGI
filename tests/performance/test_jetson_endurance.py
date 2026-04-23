"""Endurance tests for Jetson deployment.

Runs the full orchestrator loop for an extended period and validates:
- GPU temperature stays below critical threshold
- No OOM errors (RSS stable within 10%)
- Loop time p95 < 33ms at 30Hz target
- No crashes or uncaught exceptions

Marked ``@pytest.mark.hardware`` and ``@pytest.mark.slow``.

Run with::

    pytest -m "hardware and slow" -v --timeout=600 tests/performance/
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

pytestmark = [pytest.mark.hardware, pytest.mark.slow]

# Duration is configurable via env var (default 60s for CI, 300s for full validation)
ENDURANCE_DURATION_S = float(os.getenv("MOUSEDROID_ENDURANCE_DURATION_S", "60"))


def _get_rss_mb() -> float:
    """Get current process RSS in MB (Linux only)."""
    try:
        import resource

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        return rusage.ru_maxrss / 1024.0  # Convert KB to MB
    except ImportError:
        return 0.0


def _get_gpu_temp_c() -> float:
    """Get GPU temperature in Celsius (Jetson tegra thermal zone)."""
    thermal_paths = [
        "/sys/devices/virtual/thermal/thermal_zone1/temp",
        "/sys/devices/virtual/thermal/thermal_zone2/temp",
    ]
    for path in thermal_paths:
        try:
            with open(path) as f:
                return int(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError):
            continue
    return 0.0


# ---------------------------------------------------------------------------
# Endurance: 30Hz loop for configurable duration
# ---------------------------------------------------------------------------


async def test_endurance_30hz_loop(runtime_settings) -> None:
    """Run orchestrator at 30Hz for extended duration.

    Validates:
    - GPU temp < gpu_critical_temp_c (from config, default 85°C)
    - Loop time p95 < 33ms
    - RSS growth < 10% from start to end
    - No uncaught exceptions
    """
    from mousedroid.factory import build_orchestrator

    cfg = runtime_settings
    if cfg.mock_hardware:
        pytest.skip("30 Hz deadline endurance requires non-mock hardware")

    orch = build_orchestrator(cfg)

    gpu_critical_temp = cfg.safety.gpu_critical_temp_c
    target_hz = cfg.loop.control_hz
    target_loop_ms = 1000.0 / target_hz
    loop_times_ms: list[float] = []
    gpu_temps: list[float] = []
    error_count = 0

    await orch.start()

    try:
        rss_start_mb = _get_rss_mb()
        start_time = time.monotonic()
        tick_count = 0

        while (time.monotonic() - start_time) < ENDURANCE_DURATION_S:
            tick_start = time.monotonic()

            try:
                await orch.tick()
            except Exception:
                error_count += 1
                # Continue running — we want to measure stability

            elapsed_ms = (time.monotonic() - tick_start) * 1000.0
            loop_times_ms.append(elapsed_ms)

            # Sample GPU temp every 100 ticks
            tick_count += 1
            if tick_count % 100 == 0:
                temp = _get_gpu_temp_c()
                if temp > 0:
                    gpu_temps.append(temp)

            # Rate-limit to 30Hz
            target_elapsed = 1.0 / target_hz
            actual_elapsed = time.monotonic() - tick_start
            if actual_elapsed < target_elapsed:
                await asyncio.sleep(target_elapsed - actual_elapsed)

        rss_end_mb = _get_rss_mb()

    finally:
        await orch.stop()

    # --------------- Assertions ---------------

    assert len(loop_times_ms) > 0, "No ticks completed"

    # Loop time p95
    sorted_times = sorted(loop_times_ms)
    p95_idx = int(0.95 * len(sorted_times))
    p95_ms = sorted_times[min(p95_idx, len(sorted_times) - 1)]

    assert p95_ms < target_loop_ms, (
        f"Loop p95={p95_ms:.1f}ms exceeds target {target_loop_ms:.1f}ms"
    )

    # GPU temperature
    if gpu_temps:
        max_gpu_temp = max(gpu_temps)
        assert max_gpu_temp < gpu_critical_temp, (
            f"GPU temp {max_gpu_temp:.1f}°C exceeds critical {gpu_critical_temp}°C"
        )

    # Memory stability (allow 10% growth)
    if rss_start_mb > 0 and rss_end_mb > 0:
        growth_pct = (rss_end_mb - rss_start_mb) / rss_start_mb * 100
        assert growth_pct < 10.0, (
            f"RSS grew {growth_pct:.1f}% ({rss_start_mb:.0f}MB → {rss_end_mb:.0f}MB)"
        )

    # Error count
    error_rate = error_count / len(loop_times_ms) * 100
    assert error_rate < 1.0, (
        f"Error rate {error_rate:.2f}% ({error_count}/{len(loop_times_ms)} ticks)"
    )


# ---------------------------------------------------------------------------
# Voice endurance: Rocky speaks throughout without crash
# ---------------------------------------------------------------------------


async def test_voice_endurance(runtime_settings) -> None:
    """Voice engine handles repeated events without crash or memory leak."""
    from mousedroid.factory import build_speaker, build_voice_engine

    cfg = runtime_settings
    speaker = build_speaker(cfg)
    voice = build_voice_engine(cfg, speaker=speaker)
    if voice is None:
        pytest.skip("Voice engine not configured")

    await voice.start()
    try:
        rss_start = _get_rss_mb()

        for i in range(100):
            await voice.speak("obstacle_detected", {"distance_m": 0.5 + i * 0.01})

        rss_end = _get_rss_mb()

        if rss_start > 0 and rss_end > 0:
            growth_pct = (rss_end - rss_start) / rss_start * 100
            assert growth_pct < 5.0, f"Voice RSS grew {growth_pct:.1f}%"
    finally:
        await voice.stop()


# ---------------------------------------------------------------------------
# Sensor recovery endurance: repeated recovery cycles
# ---------------------------------------------------------------------------


async def test_sensor_recovery_endurance(runtime_settings) -> None:
    """Sensor manager handles repeated recovery cycles without crash."""
    from mousedroid.factory import (
        build_camera,
        build_distance_sensor,
        build_esp32_driver,
        build_microphone,
        build_sensor_manager,
    )

    cfg = runtime_settings
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
        for _ in range(10):
            recovered = await mgr.recovery_attempt()
            assert recovered >= 0

            # Verify sensors still work after recovery
            obs = await mgr.read_all()
            assert obs.valid_mask is not None
    finally:
        if mic is not None:
            await mic.stop()
        await camera.stop()
        await esp32.emergency_stop()
        await esp32.disconnect()
