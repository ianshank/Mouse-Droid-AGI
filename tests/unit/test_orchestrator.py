"""Tests for MouseDroidOrchestrator — full coverage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import torch

from mousedroid.config.schema import Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle


def _make_observation(cfg: Settings) -> MouseDroidObservationBundle:
    """Create a default observation bundle for testing."""
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(1024, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


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

    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _make_observation(cfg)

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        cognitive_core=cognitive_core,
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


async def test_tick_delegates_to_sensor_manager():
    orch = _make_orchestrator()
    await orch.tick()
    orch._sensor_manager.read_all.assert_awaited_once()


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


async def test_orchestrator_with_cognitive_core_primary():
    """Test orchestrator uses cognitive core action when available."""
    orch = _make_orchestrator()

    # Create mock cognitive core that returns valid action
    cognitive_core = MagicMock()
    cognitive_core.tick_fast = MagicMock(return_value=(np.array([0.3, 0.2]), []))
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
    cognitive_core.tick_fast = MagicMock(side_effect=RuntimeError("cognitive fail"))
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
    """Test orchestrator handles out-of-bounds cognitive core actions without crashing."""
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
    cognitive_core.tick_fast = MagicMock(return_value=(np.array([0.1, 0.0]), violations))
    orch._cognitive_core = cognitive_core

    await orch.tick()

    # Verify action was still sent despite violations
    orch._esp32.send_velocity.assert_awaited_once()
    # Cognitive core was called (violations logged internally)
    cognitive_core.tick_fast.assert_called_once()


async def test_update_world_model():
    """Test _update_world_model updates latent state."""
    orch = _make_orchestrator()
    cfg = orch._cfg
    obs = _make_observation(cfg)
    orch._update_world_model(obs)
    orch._world_model.observe_step.assert_called_once()


async def test_execute_action():
    """Test _execute_action sends scaled velocity to ESP32."""
    orch = _make_orchestrator()
    action = torch.tensor([0.5, 0.3, 0.2])
    await orch._execute_action(action)
    orch._esp32.send_velocity.assert_awaited_once()
    args = orch._esp32.send_velocity.call_args[0]
    max_v = orch._cfg.esp32.max_velocity_mps
    max_omega = orch._cfg.esp32.max_omega_rads
    assert abs(args[0] - 0.5 * max_v) < 1e-6
    assert abs(args[1] - 0.3 * max_v) < 1e-6
    assert abs(args[2] - 0.2 * max_omega) < 1e-6


async def test_normalize_cognitive_action_padding():
    """Test _normalize_cognitive_action pads short actions."""
    orch = _make_orchestrator()
    action_np = np.array([0.5], dtype=np.float32)
    result = orch._normalize_cognitive_action(action_np)
    assert result.shape == (orch._cfg.model.action_dim,)
    assert float(result[0]) == 0.5
    assert float(result[1]) == 0.0


async def test_normalize_cognitive_action_truncation():
    """Test _normalize_cognitive_action truncates long actions."""
    orch = _make_orchestrator()
    action_np = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    result = orch._normalize_cognitive_action(action_np)
    assert result.shape == (orch._cfg.model.action_dim,)
