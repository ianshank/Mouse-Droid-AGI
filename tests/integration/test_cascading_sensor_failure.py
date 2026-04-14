"""Integration tests for cascading sensor failure and recovery.

Phase 4: Validates multi-sensor failure handling, emergency stop triggers,
graceful degradation, and sensor recovery protocol.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from mousedroid.config.schema import SafetyConfig, Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle


def _make_observation(
    cfg: Settings,
    *,
    valid_mask: np.ndarray | None = None,
) -> MouseDroidObservationBundle:
    """Create an observation with a configurable valid_mask."""
    if valid_mask is None:
        valid_mask = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(1024, dtype=np.float32),
        _valid_mask=valid_mask,
    )


def _make_orchestrator(
    *,
    valid_mask: np.ndarray | None = None,
    is_emergency: bool = False,
    valid_sensor_count: int = 3,
    recovery_attempts: int = 1,
) -> MouseDroidOrchestrator:
    """Create orchestrator with configurable sensor failure scenario."""
    cfg = Settings(
        mock_hardware=True,
        safety=SafetyConfig(
            sensor_recovery_attempts=recovery_attempts,
            sensor_recovery_delay_s=0.01,
        ),
    )

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
        is_emergency=is_emergency,
        valid_sensor_count=valid_sensor_count,
    )
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = safety_ctx

    esp32 = AsyncMock()

    sensor_manager = AsyncMock()
    obs = _make_observation(cfg, valid_mask=valid_mask)
    sensor_manager.read_all.return_value = obs
    sensor_manager.recovery_attempt.return_value = 0

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Config backward compatibility
# ---------------------------------------------------------------------------


def test_safety_config_recovery_defaults() -> None:
    """Safety config has backward-compatible recovery defaults."""
    cfg = SafetyConfig()
    assert cfg.sensor_recovery_attempts == 1
    assert cfg.sensor_recovery_delay_s == pytest.approx(0.5)


def test_safety_config_zero_recovery_attempts() -> None:
    """Recovery can be disabled with sensor_recovery_attempts=0."""
    cfg = SafetyConfig(sensor_recovery_attempts=0)
    assert cfg.sensor_recovery_attempts == 0


# ---------------------------------------------------------------------------
# Single sensor failure — graceful degradation
# ---------------------------------------------------------------------------


async def test_single_sensor_failure_no_emergency() -> None:
    """Single sensor failure does not trigger emergency (min_valid_sensors=2)."""
    # Vision fails, distance + motor still valid = 2 valid
    valid_mask = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    orch = _make_orchestrator(
        valid_mask=valid_mask,
        is_emergency=False,
        valid_sensor_count=2,
    )

    await orch.tick()

    # Normal tick — no emergency stop
    orch._esp32.emergency_stop.assert_not_awaited()
    orch._esp32.send_velocity.assert_awaited_once()


# ---------------------------------------------------------------------------
# Dual sensor failure — emergency triggered
# ---------------------------------------------------------------------------


async def test_dual_sensor_failure_triggers_emergency() -> None:
    """Two sensor failures trigger emergency stop."""
    valid_mask = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    orch = _make_orchestrator(
        valid_mask=valid_mask,
        is_emergency=True,
        valid_sensor_count=1,
    )

    await orch.tick()

    orch._esp32.emergency_stop.assert_awaited_once()


# ---------------------------------------------------------------------------
# All sensors fail — emergency triggered within 1 tick
# ---------------------------------------------------------------------------


async def test_all_sensors_fail_emergency_within_one_tick() -> None:
    """All sensors failing triggers emergency stop within a single tick."""
    valid_mask = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    orch = _make_orchestrator(
        valid_mask=valid_mask,
        is_emergency=True,
        valid_sensor_count=0,
    )

    await orch.tick()

    orch._esp32.emergency_stop.assert_awaited_once()


# ---------------------------------------------------------------------------
# Sensor recovery protocol
# ---------------------------------------------------------------------------


async def test_recovery_attempted_on_sensor_emergency() -> None:
    """Recovery is attempted when emergency is due to sensor degradation."""
    orch = _make_orchestrator(
        is_emergency=True,
        valid_sensor_count=1,  # Below default min_valid_sensors=2
        recovery_attempts=1,
    )

    await orch.tick()

    # Recovery should have been attempted
    orch._sensor_manager.recovery_attempt.assert_awaited()


async def test_recovery_not_attempted_when_disabled() -> None:
    """Recovery is not attempted when sensor_recovery_attempts=0."""
    orch = _make_orchestrator(
        is_emergency=True,
        valid_sensor_count=1,
        recovery_attempts=0,
    )

    await orch.tick()

    # Recovery should NOT have been attempted
    orch._sensor_manager.recovery_attempt.assert_not_awaited()
    # But emergency stop should still fire
    orch._esp32.emergency_stop.assert_awaited_once()


async def test_recovery_not_attempted_for_non_sensor_emergency() -> None:
    """Recovery is not attempted when valid sensors are sufficient."""
    orch = _make_orchestrator(
        is_emergency=True,
        valid_sensor_count=3,  # Above min_valid_sensors=2
        recovery_attempts=1,
    )

    await orch.tick()

    # No recovery needed — emergency was from other cause (e.g., obstacle)
    orch._sensor_manager.recovery_attempt.assert_not_awaited()
    orch._esp32.emergency_stop.assert_awaited_once()


async def test_successful_recovery_avoids_emergency_stop() -> None:
    """Successful sensor recovery prevents emergency stop."""
    orch = _make_orchestrator(
        is_emergency=True,
        valid_sensor_count=1,
        recovery_attempts=2,
    )

    # After recovery, sensors are back
    orch._sensor_manager.recovery_attempt.return_value = 2
    # Second evaluate returns non-emergency
    orch._safety_monitor.evaluate.side_effect = [
        SafetyContext(is_emergency=True, valid_sensor_count=1),
        SafetyContext(is_emergency=False, valid_sensor_count=3),
    ]

    await orch.tick()

    # Recovery was attempted and succeeded
    orch._sensor_manager.recovery_attempt.assert_awaited()
    # Emergency stop should NOT have fired
    orch._esp32.emergency_stop.assert_not_awaited()
    # Normal action should have been sent
    orch._esp32.send_velocity.assert_awaited_once()


async def test_failed_recovery_still_triggers_emergency() -> None:
    """Failed sensor recovery still triggers emergency stop."""
    orch = _make_orchestrator(
        is_emergency=True,
        valid_sensor_count=0,
        recovery_attempts=2,
    )

    # Recovery fails — returns 0 recovered
    orch._sensor_manager.recovery_attempt.return_value = 0

    await orch.tick()

    # Recovery was attempted
    assert orch._sensor_manager.recovery_attempt.await_count == 2
    # But still emergency-stopped
    orch._esp32.emergency_stop.assert_awaited_once()


# ---------------------------------------------------------------------------
# LiDAR stale + camera valid — system continues
# ---------------------------------------------------------------------------


async def test_lidar_stale_camera_valid_continues() -> None:
    """LiDAR failure with camera valid allows continued operation."""
    # Vision valid, distance valid, motor valid, audio off, lidar off
    valid_mask = np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    orch = _make_orchestrator(
        valid_mask=valid_mask,
        is_emergency=False,
        valid_sensor_count=3,
    )

    await orch.tick()

    orch._esp32.emergency_stop.assert_not_awaited()
    orch._esp32.send_velocity.assert_awaited_once()


# ---------------------------------------------------------------------------
# Sensor recovery in SensorManager (unit-like test)
# ---------------------------------------------------------------------------


async def test_sensor_manager_recovery_attempt() -> None:
    """SensorManager.recovery_attempt() attempts to restart sensors."""
    from mousedroid.sensing.manager import SensorManager

    cfg = Settings(mock_hardware=True)
    vision = AsyncMock()
    distance = AsyncMock()
    esp32 = AsyncMock()

    # Build real SensorManager with mocks
    mgr = SensorManager(
        vision=vision,
        distance=distance,
        esp32=esp32,
        cfg=cfg,
    )

    recovered = await mgr.recovery_attempt()
    # Vision was stop/started and re-read
    vision.stop.assert_awaited_once()
    vision.start.assert_awaited_once()
    assert recovered >= 0


async def test_sensor_manager_recovery_handles_exception() -> None:
    """SensorManager.recovery_attempt() handles sensor restart failure."""
    from mousedroid.sensing.manager import SensorManager

    cfg = Settings(mock_hardware=True)
    vision = AsyncMock()
    vision.stop.side_effect = RuntimeError("hardware dead")
    distance = AsyncMock()
    esp32 = AsyncMock()

    mgr = SensorManager(
        vision=vision,
        distance=distance,
        esp32=esp32,
        cfg=cfg,
    )

    # Should not raise
    recovered = await mgr.recovery_attempt()
    assert recovered >= 0
