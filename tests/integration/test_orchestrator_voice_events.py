"""Integration tests for orchestrator voice lifecycle and enriched events.

Phase 3: Validates startup/shutdown/error voice events reach the speaker,
and that voice context is enriched with LiDAR distance and audio RMS.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from mousedroid.config.schema import Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle


def _make_observation(
    cfg: Settings,
    *,
    distance_m: float = 1.5,
    lidar_features: np.ndarray | None = None,
    audio_chunk: np.ndarray | None = None,
) -> MouseDroidObservationBundle:
    """Create an observation bundle with optional LiDAR and audio data."""
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
        _distance_m=distance_m,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=audio_chunk if audio_chunk is not None else np.zeros(1024, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        _lidar_features=lidar_features,
    )


def _make_voiced_orchestrator(
    *,
    emergency: bool = False,
    forward_clearance_ok: bool = True,
    lidar_features: np.ndarray | None = None,
    audio_chunk: np.ndarray | None = None,
) -> MouseDroidOrchestrator:
    """Create orchestrator with a mock voice engine."""
    cfg = Settings(mock_hardware=True)

    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )

    agent = MagicMock()
    agent.name = "test_agent"
    agent.act.return_value = torch.tensor([0.1, 0.0, 0.0])

    safety_ctx = SafetyContext(
        is_emergency=emergency,
        forward_clearance_ok=forward_clearance_ok,
    )
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = safety_ctx

    esp32 = AsyncMock()

    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _make_observation(
        cfg,
        lidar_features=lidar_features,
        audio_chunk=audio_chunk,
    )
    sensor_manager.recovery_attempt.return_value = 0

    voice_engine = AsyncMock()

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        voice_engine=voice_engine,
    )


# ---------------------------------------------------------------------------
# Startup / Shutdown lifecycle events
# ---------------------------------------------------------------------------


async def test_startup_voice_event_fires_on_start() -> None:
    """start() fires 'startup' voice event after engine starts."""
    orch = _make_voiced_orchestrator()
    await orch.start()

    # Voice engine should have been started and spoken
    orch._voice_engine.start.assert_awaited_once()
    orch._voice_engine.speak.assert_awaited()

    # First speak call should be 'startup'
    first_call = orch._voice_engine.speak.call_args_list[0]
    assert first_call[0][0] == "startup"

    await orch.stop()


async def test_shutdown_voice_event_fires_on_stop() -> None:
    """stop() fires 'shutdown' voice event before engine stops."""
    orch = _make_voiced_orchestrator()
    await orch.start()
    orch._voice_engine.speak.reset_mock()

    await orch.stop()

    # Should have spoken 'shutdown' before stopping engine
    orch._voice_engine.speak.assert_awaited()
    shutdown_call = orch._voice_engine.speak.call_args_list[0]
    assert shutdown_call[0][0] == "shutdown"
    orch._voice_engine.stop.assert_awaited_once()


async def test_startup_without_voice_engine_no_crash() -> None:
    """start() without voice engine does not crash."""
    cfg = Settings(mock_hardware=True)
    orch = MouseDroidOrchestrator(
        world_model=MagicMock(),
        agents=[MagicMock(name="a")],
        safety_monitor=MagicMock(),
        esp32=AsyncMock(),
        sensor_manager=AsyncMock(),
        cfg=cfg,
    )
    await orch.start()
    assert orch._running is True
    await orch.stop()


# ---------------------------------------------------------------------------
# Emergency stop → voice event
# ---------------------------------------------------------------------------


async def test_emergency_stop_fires_voice_event() -> None:
    """Emergency tick fires 'emergency_stop' voice event."""
    orch = _make_voiced_orchestrator(emergency=True)
    await orch.tick()

    orch._voice_engine.speak.assert_awaited()
    event_arg = orch._voice_engine.speak.call_args[0][0]
    assert event_arg == "emergency_stop"


# ---------------------------------------------------------------------------
# Tick error → voice event
# ---------------------------------------------------------------------------


async def test_tick_error_fires_voice_event() -> None:
    """run() fires 'error' voice event when tick throws exception."""
    orch = _make_voiced_orchestrator()
    orch._running = True
    call_count = 0

    async def failing_tick() -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("sensor failure")
        orch._running = False

    orch.tick = failing_tick  # type: ignore[assignment]
    await orch.run()

    # Error voice event should have fired
    events = [c[0][0] for c in orch._voice_engine.speak.call_args_list]
    assert "error" in events


# ---------------------------------------------------------------------------
# Voice context enrichment with LiDAR and audio
# ---------------------------------------------------------------------------


async def test_voice_event_includes_lidar_min_dist() -> None:
    """Voice event context includes lidar_min_dist_m when LiDAR features present."""
    lidar_features = np.array([2.0, 0.5, 1.0, 3.0], dtype=np.float32)
    orch = _make_voiced_orchestrator(
        forward_clearance_ok=False,
        lidar_features=lidar_features,
    )

    await orch.tick()

    # Should have called speak with obstacle_detected
    orch._voice_engine.speak.assert_awaited()
    context = orch._voice_engine.speak.call_args[0][1]
    assert "lidar_min_dist_m" in context
    assert context["lidar_min_dist_m"] == pytest.approx(0.5, abs=1e-5)


async def test_voice_event_includes_audio_rms() -> None:
    """Voice event context includes audio_level_rms when audio chunk present."""
    # Create a known audio signal: constant 0.5 => RMS = 0.5
    audio_chunk = np.full(1024, 0.5, dtype=np.float32)
    orch = _make_voiced_orchestrator(
        forward_clearance_ok=False,
        audio_chunk=audio_chunk,
    )

    await orch.tick()

    orch._voice_engine.speak.assert_awaited()
    context = orch._voice_engine.speak.call_args[0][1]
    assert "audio_level_rms" in context
    assert context["audio_level_rms"] == pytest.approx(0.5, abs=1e-5)


async def test_voice_event_no_lidar_no_crash() -> None:
    """Voice event without LiDAR features does not include lidar_min_dist_m."""
    orch = _make_voiced_orchestrator(forward_clearance_ok=False)

    await orch.tick()

    orch._voice_engine.speak.assert_awaited()
    context = orch._voice_engine.speak.call_args[0][1]
    assert "lidar_min_dist_m" not in context
    assert "distance_m" in context


async def test_voice_event_zero_audio_rms() -> None:
    """Voice event with silent audio chunk has audio_level_rms ≈ 0."""
    audio_chunk = np.zeros(1024, dtype=np.float32)
    orch = _make_voiced_orchestrator(
        forward_clearance_ok=False,
        audio_chunk=audio_chunk,
    )

    await orch.tick()

    orch._voice_engine.speak.assert_awaited()
    context = orch._voice_engine.speak.call_args[0][1]
    assert "audio_level_rms" in context
    assert context["audio_level_rms"] == pytest.approx(0.0, abs=1e-7)


async def test_voice_lifecycle_exception_handled() -> None:
    """_voice_lifecycle exception does not crash orchestrator."""
    orch = _make_voiced_orchestrator()
    orch._voice_engine.speak.side_effect = RuntimeError("TTS crash")

    # start() should not raise despite voice failure
    await orch.start()
    assert orch._running is True

    await orch.stop()
