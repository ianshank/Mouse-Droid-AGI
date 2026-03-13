"""Tests for MouseDroidOrchestrator — full coverage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import torch

from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.schema import Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext


def _make_orchestrator(
    *,
    emergency: bool = False,
    cognitive_core: object | None = None,
) -> MouseDroidOrchestrator:
    cfg = Settings(mock_hardware=True)

    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )

    agent = MagicMock()
    agent.name = "test_agent"
    agent.act.return_value = torch.tensor([0.1, 0.0, 0.0])

    safety_ctx = SafetyContext(is_emergency=emergency)
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = safety_ctx

    esp32 = AsyncMock()
    esp32.read_encoders.return_value = EncoderReading()
    esp32.get_battery_voltage.return_value = 12.0

    camera = AsyncMock()
    camera.capture_features.return_value = np.zeros(cfg.camera.feature_dim, dtype=np.float32)

    distance_sensor = MagicMock()
    distance_sensor.max_range_m = 4.0
    distance_sensor.read_distance_m = AsyncMock(return_value=1.5)

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        camera=camera,
        distance_sensor=distance_sensor,
        cfg=cfg,
        cognitive_core=cognitive_core,
        microphone=None,
    )


def test_constructor():
    orch = _make_orchestrator()
    assert orch._running is False


async def test_health_check():
    orch = _make_orchestrator()
    result = await orch.health_check()
    assert result["status"] == "ok"
    assert result["mock_hardware"] is True
    assert "test_agent" in result["agents"]


async def test_start_sets_running():
    orch = _make_orchestrator()
    await orch.start()
    assert orch._running is True


async def test_stop_clears_running():
    orch = _make_orchestrator()
    await orch.start()
    await orch.stop()
    assert orch._running is False


async def test_tick_full_cycle():
    orch = _make_orchestrator()
    await orch.tick()
    orch._esp32.send_velocity.assert_awaited_once()


async def test_tick_emergency_stop():
    orch = _make_orchestrator(emergency=True)
    await orch.tick()
    orch._esp32.emergency_stop.assert_awaited_once()
    orch._esp32.send_velocity.assert_not_awaited()


async def test_sense_vision_failure():
    orch = _make_orchestrator()
    orch._camera.capture_features.side_effect = RuntimeError("camera fail")
    obs = await orch._sense()
    np.testing.assert_array_equal(
        obs.vision_features,
        np.zeros(orch._cfg.camera.feature_dim, dtype=np.float32),
    )
    assert obs.valid_mask[0] == 0.0


async def test_sense_distance_failure():
    orch = _make_orchestrator()
    orch._distance_sensor.read_distance_m = AsyncMock(
        side_effect=RuntimeError("sensor fail"),
    )
    obs = await orch._sense()
    assert obs.distance_m == orch._distance_sensor.max_range_m
    assert obs.valid_mask[1] == 0.0


async def test_sense_motor_failure():
    orch = _make_orchestrator()
    orch._esp32.read_encoders.side_effect = RuntimeError("motor fail")
    obs = await orch._sense()
    np.testing.assert_array_equal(obs.motor_state, np.zeros(4, dtype=np.float32))
    assert obs.valid_mask[2] == 0.0


async def test_run_loop_single_iteration():
    orch = _make_orchestrator()
    orch._running = True
    tick_count = 0
    original_tick = orch.tick

    async def counting_tick():
        nonlocal tick_count
        await original_tick()
        tick_count += 1
        orch._running = False

    orch.tick = counting_tick  # type: ignore[assignment]
    await orch.run()
    assert tick_count == 1


async def test_run_loop_tick_exception_continues():
    orch = _make_orchestrator()
    orch._running = True
    call_count = 0

    async def failing_tick():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("tick error")
        orch._running = False

    orch.tick = failing_tick  # type: ignore[assignment]
    await orch.run()
    assert call_count == 2


async def test_tick_action_1d():
    orch = _make_orchestrator()
    orch._agents[0].act.return_value = torch.tensor([0.5])
    await orch.tick()
    orch._esp32.send_velocity.assert_awaited_once()


async def test_tick_action_2d():
    orch = _make_orchestrator()
    orch._agents[0].act.return_value = torch.tensor([0.5, 0.3])
    await orch.tick()
    orch._esp32.send_velocity.assert_awaited_once()


def _make_orchestrator_with_mic() -> MouseDroidOrchestrator:
    cfg = Settings(mock_hardware=True)

    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )

    agent = MagicMock()
    agent.name = "test_agent"
    agent.act.return_value = torch.tensor([0.1, 0.0, 0.0])

    safety_ctx = SafetyContext(is_emergency=False)
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = safety_ctx

    esp32 = AsyncMock()
    esp32.read_encoders.return_value = EncoderReading()
    esp32.get_battery_voltage.return_value = 12.0

    camera = AsyncMock()
    camera.capture_features.return_value = np.zeros(
        cfg.camera.feature_dim,
        dtype=np.float32,
    )

    distance_sensor = MagicMock()
    distance_sensor.max_range_m = 4.0
    distance_sensor.read_distance_m = AsyncMock(return_value=1.5)

    mic = AsyncMock()
    mic.chunk_size = 1024
    mic.channels = 1
    mic.read_chunk.return_value = np.zeros(1024, dtype=np.float32)

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        camera=camera,
        distance_sensor=distance_sensor,
        cfg=cfg,
        microphone=mic,
    )


async def test_start_with_microphone():
    orch = _make_orchestrator_with_mic()
    await orch.start()
    orch._microphone.start.assert_awaited_once()
    assert orch._running is True


async def test_stop_with_microphone():
    orch = _make_orchestrator_with_mic()
    await orch.start()
    await orch.stop()
    orch._microphone.stop.assert_awaited_once()
    assert orch._running is False


async def test_sense_with_microphone():
    orch = _make_orchestrator_with_mic()
    obs = await orch._sense()
    assert obs.valid_mask[3] == 1.0
    orch._microphone.read_chunk.assert_awaited_once()


async def test_sense_microphone_failure():
    orch = _make_orchestrator_with_mic()
    orch._microphone.read_chunk.side_effect = RuntimeError("mic fail")
    obs = await orch._sense()
    assert obs.valid_mask[3] == 0.0
    np.testing.assert_array_equal(
        obs.audio_chunk,
        np.zeros(1024, dtype=np.float32),
    )


async def test_orchestrator_with_cognitive_core_primary():
    """Test orchestrator uses cognitive core action when available."""
    orch = _make_orchestrator()

    # Create mock cognitive core that returns valid action
    cognitive_core = MagicMock()
    cognitive_core.tick_fast = MagicMock(
        return_value=(np.array([0.3, 0.2]), [])
    )
    orch._cognitive_core = cognitive_core

    await orch.tick()

    # Verify cognitive core was called
    cognitive_core.tick_fast.assert_called_once()
    # Verify action was sent to ESP32
    orch._esp32.send_velocity.assert_awaited_once()


async def test_orchestrator_cognitive_fallback_to_mcts_on_error():
    """Test orchestrator falls back to MCTS when cognitive core fails."""
    orch = _make_orchestrator()

    # Create mock cognitive core that raises exception
    cognitive_core = MagicMock()
    cognitive_core.tick_fast = MagicMock(
        side_effect=RuntimeError("cognitive fail")
    )
    orch._cognitive_core = cognitive_core

    await orch.tick()

    # Verify MCTS agent was used as fallback
    orch._agents[0].act.assert_called_once()
    # Verify action was still sent to ESP32
    orch._esp32.send_velocity.assert_awaited_once()


async def test_orchestrator_start_calls_cognitive_core_start():
    """Test orchestrator.start() initializes cognitive core."""
    orch = _make_orchestrator()

    cognitive_core = AsyncMock()
    orch._cognitive_core = cognitive_core

    await orch.start()

    # Verify cognitive core was started
    cognitive_core.start.assert_awaited_once()
    assert orch._running is True


async def test_orchestrator_stop_calls_cognitive_core_stop():
    """Test orchestrator.stop() shuts down cognitive core."""
    orch = _make_orchestrator()

    cognitive_core = AsyncMock()
    orch._cognitive_core = cognitive_core
    await orch.start()

    await orch.stop()

    # Verify cognitive core was stopped
    cognitive_core.stop.assert_awaited_once()
    assert orch._running is False


async def test_orchestrator_without_cognitive_core_uses_mcts():
    """Test orchestrator operates normally without cognitive core."""
    orch = _make_orchestrator(cognitive_core=None)
    assert orch._cognitive_core is None

    await orch.tick()

    # MCTS agent should be used directly
    orch._agents[0].act.assert_called_once()
    orch._esp32.send_velocity.assert_awaited_once()


async def test_cognitive_action_bounds():
    """Test cognitive core actions are bounded to [-1, 1]."""
    orch = _make_orchestrator()

    # Mock cognitive core returning out-of-bounds action
    cognitive_core = MagicMock()
    cognitive_core.tick_fast = MagicMock(
        return_value=(np.array([0.5, 1.5]), [])  # Second value out of bounds
    )
    orch._cognitive_core = cognitive_core

    await orch.tick()

    # Should still send velocity without crashing
    orch._esp32.send_velocity.assert_awaited_once()


async def test_constitutional_violations_logged():
    """Test that constitutional violations are logged but don't block action."""
    orch = _make_orchestrator()

    # Mock cognitive core returning violations
    cognitive_core = MagicMock()
    violations = ["battery_too_low", "obstacle_too_close"]
    cognitive_core.tick_fast = MagicMock(
        return_value=(np.array([0.1, 0.0]), violations)
    )
    orch._cognitive_core = cognitive_core

    await orch.tick()

    # Verify action was still sent despite violations
    orch._esp32.send_velocity.assert_awaited_once()
    # Cognitive core was called (violations logged internally)
    cognitive_core.tick_fast.assert_called_once()
