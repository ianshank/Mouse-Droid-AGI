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
    sensor_manager.recovery_attempt.return_value = 0

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


async def test_try_cognitive_action_state_vec_padding():
    """Test _try_cognitive_action pads state_vec when smaller than belief_dim."""
    orch = _make_orchestrator()

    # Set hidden state smaller than belief_dim to trigger padding branch
    belief_dim = orch._cfg.model.belief_dim
    small_dim = belief_dim // 2
    orch._h = torch.zeros(1, small_dim)

    cognitive_core = MagicMock()
    cognitive_core.tick_fast = MagicMock(return_value=(np.array([0.1, 0.0, 0.0]), []))
    orch._cognitive_core = cognitive_core

    obs = _make_observation(orch._cfg)
    result = orch._try_cognitive_action(obs, 10.0)

    assert result is not None
    # Verify cognitive core was called with padded state
    call_args = cognitive_core.tick_fast.call_args[0][0]
    assert call_args["state"].shape == (belief_dim,)


async def test_try_cognitive_action_passes_full_bdi_state() -> None:
    """Test _try_cognitive_action preserves the full latent state for BDI."""
    orch = _make_orchestrator()
    hidden_dim = orch._cfg.model.hidden_dim
    orch._h = torch.arange(hidden_dim, dtype=torch.float32).reshape(1, hidden_dim)

    cognitive_core = MagicMock()
    cognitive_core.tick_fast = MagicMock(return_value=(np.array([0.1, 0.0, 0.0]), []))
    orch._cognitive_core = cognitive_core

    obs = _make_observation(orch._cfg)
    result = orch._try_cognitive_action(obs, 10.0)

    assert result is not None
    call_args = cognitive_core.tick_fast.call_args[0][0]
    assert call_args["state"].shape == (orch._cfg.model.belief_dim,)
    assert call_args["bdi_state"].shape == (hidden_dim,)
    np.testing.assert_array_equal(
        call_args["bdi_state"],
        np.arange(hidden_dim, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Voice engine integration tests
# ---------------------------------------------------------------------------


def _make_orchestrator_with_voice(
    *,
    emergency: bool = False,
    forward_clearance_ok: bool = True,
    battery_voltage: float = 12.0,
    gpu_temp_c: float = 40.0,
) -> MouseDroidOrchestrator:
    """Create orchestrator with a mock voice engine."""
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

    safety_ctx = SafetyContext(
        is_emergency=emergency,
        forward_clearance_ok=forward_clearance_ok,
        battery_voltage=battery_voltage,
        gpu_temp_c=gpu_temp_c,
    )
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = safety_ctx

    esp32 = AsyncMock()

    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _make_observation(cfg)
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


async def test_voice_engine_started_on_start():
    """start() calls voice_engine.start()."""
    orch = _make_orchestrator_with_voice()
    await orch.start()
    orch._voice_engine.start.assert_awaited_once()


async def test_voice_engine_stopped_on_stop():
    """stop() calls voice_engine.stop()."""
    orch = _make_orchestrator_with_voice()
    await orch.start()
    await orch.stop()
    orch._voice_engine.stop.assert_awaited_once()


async def test_voice_event_none_engine_no_crash():
    """_voice_event with voice_engine=None does not crash."""
    orch = _make_orchestrator()  # No voice engine
    obs = _make_observation(orch._cfg)
    await orch._voice_event("emergency_stop", obs)  # Should not raise


async def test_voice_observe_obstacle():
    """_voice_observe fires obstacle_detected when clearance is bad."""
    orch = _make_orchestrator_with_voice(forward_clearance_ok=False)
    obs = _make_observation(orch._cfg)
    ctx = SafetyContext(forward_clearance_ok=False)
    await orch._voice_observe(obs, ctx)
    orch._voice_engine.speak.assert_awaited()
    event_arg = orch._voice_engine.speak.call_args[0][0]
    assert event_arg == "obstacle_detected"


async def test_voice_observe_low_battery():
    """_voice_observe fires low_battery when voltage below threshold."""
    orch = _make_orchestrator_with_voice(battery_voltage=9.0)
    obs = _make_observation(orch._cfg)
    ctx = SafetyContext(battery_voltage=9.0)
    await orch._voice_observe(obs, ctx)
    orch._voice_engine.speak.assert_awaited()
    event_arg = orch._voice_engine.speak.call_args[0][0]
    assert event_arg == "low_battery"


async def test_voice_observe_gpu_overheat():
    """_voice_observe fires error event when GPU temp exceeds warning."""
    orch = _make_orchestrator_with_voice(gpu_temp_c=80.0)
    obs = _make_observation(orch._cfg)
    ctx = SafetyContext(gpu_temp_c=80.0)
    await orch._voice_observe(obs, ctx)
    orch._voice_engine.speak.assert_awaited()
    event_arg = orch._voice_engine.speak.call_args[0][0]
    assert event_arg == "error"


async def test_voice_observe_all_clear_no_event():
    """_voice_observe does not fire events when all is well."""
    orch = _make_orchestrator_with_voice()
    obs = _make_observation(orch._cfg)
    ctx = SafetyContext()
    await orch._voice_observe(obs, ctx)
    orch._voice_engine.speak.assert_not_awaited()


async def test_emergency_tick_fires_voice_event():
    """Emergency stop tick fires voice emergency_stop event."""
    orch = _make_orchestrator_with_voice(emergency=True)
    await orch.tick()
    orch._voice_engine.speak.assert_awaited()
    event_arg = orch._voice_engine.speak.call_args[0][0]
    assert event_arg == "emergency_stop"


async def test_voice_event_exception_handled():
    """Voice engine exception in tick does not crash orchestrator."""
    orch = _make_orchestrator_with_voice(forward_clearance_ok=False)
    orch._voice_engine.speak.side_effect = RuntimeError("voice failed")
    obs = _make_observation(orch._cfg)
    ctx = SafetyContext(forward_clearance_ok=False)
    await orch._voice_observe(obs, ctx)  # Should not raise


async def test_voice_observe_uses_config_battery_threshold():
    """_voice_observe uses safety config battery_warn_v, not hardcoded value."""
    orch = _make_orchestrator_with_voice()
    threshold = orch._cfg.safety.battery_warn_v
    # Voltage just below threshold should trigger
    obs = _make_observation(orch._cfg)
    ctx = SafetyContext(battery_voltage=threshold - 0.1)
    await orch._voice_observe(obs, ctx)
    orch._voice_engine.speak.assert_awaited()
    # Voltage at threshold should NOT trigger
    orch._voice_engine.reset_mock()
    ctx_ok = SafetyContext(battery_voltage=threshold + 0.1)
    await orch._voice_observe(obs, ctx_ok)
    orch._voice_engine.speak.assert_not_awaited()
