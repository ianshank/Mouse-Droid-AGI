"""Unit tests for orchestrator tick timeout and exception emergency stop.

Validates that the run() loop:
1. Calls emergency_stop when tick() exceeds tick_timeout_s
2. Calls emergency_stop when tick() raises an unhandled exception
3. Notifies the watchdog after each successful tick
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import torch

from mousedroid.config.schema import Settings


def _make_orchestrator(
    tick_timeout_s: float = 0.1,
    watchdog: object | None = None,
) -> tuple:
    """Build a minimal MouseDroidOrchestrator with mocked subsystems.

    Returns:
        (orchestrator, esp32_mock) tuple.
    """
    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    cfg = Settings(mock_hardware=True)
    # Override tick_timeout via object attribute (Pydantic allows this)
    cfg.loop.tick_timeout_s = tick_timeout_s

    wm = MagicMock()
    wm.observe_step = MagicMock(return_value=(
        torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.latent_dim),
        0.0,
    ))

    agent = MagicMock()
    agent.name = "mock_agent"
    agent.act = MagicMock(return_value=torch.zeros(cfg.model.action_dim))

    monitor = MagicMock()
    safety_ctx = MagicMock()
    safety_ctx.is_emergency = False
    safety_ctx.forward_clearance_ok = True
    safety_ctx.battery_voltage = 12.0
    safety_ctx.gpu_temp_c = 50.0
    monitor.evaluate = MagicMock(return_value=safety_ctx)

    esp32 = AsyncMock()
    esp32.connect = AsyncMock()
    esp32.disconnect = AsyncMock()
    esp32.emergency_stop = AsyncMock()
    esp32.send_velocity = AsyncMock()

    sensor_manager = AsyncMock()
    observation = MagicMock()
    observation.distance_m = 2.0
    observation.motor_state = np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32)
    observation.vision_features = np.zeros(256, dtype=np.float32)
    sensor_manager.read_all = AsyncMock(return_value=observation)
    sensor_manager.start = AsyncMock()
    sensor_manager.stop = AsyncMock()

    orch = MouseDroidOrchestrator(
        world_model=wm,
        agents=[agent],
        safety_monitor=monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        watchdog=watchdog,
    )

    return orch, esp32


class TestTickTimeout:
    """Tick timeout triggers emergency stop."""

    async def test_timeout_triggers_estop(self) -> None:
        """When tick() exceeds tick_timeout_s, emergency_stop must fire."""
        orch, esp32 = _make_orchestrator(tick_timeout_s=0.05)

        # Replace tick with a version that hangs indefinitely
        async def hanging_tick() -> None:
            await asyncio.sleep(999)

        orch.tick = hanging_tick  # type: ignore[assignment]
        orch._running = True

        # Run one loop iteration then stop
        async def stop_after_delay() -> None:
            await asyncio.sleep(0.2)
            orch._running = False

        await asyncio.gather(orch.run(), stop_after_delay())

        esp32.emergency_stop.assert_awaited()


class TestExceptionEmergencyStop:
    """Exception in tick() triggers emergency stop."""

    async def test_exception_triggers_estop(self) -> None:
        """When tick() raises, emergency_stop must fire."""
        orch, esp32 = _make_orchestrator(tick_timeout_s=5.0)

        async def failing_tick() -> None:
            msg = "sensor driver crashed"
            raise RuntimeError(msg)

        orch.tick = failing_tick  # type: ignore[assignment]
        orch._running = True

        async def stop_after_delay() -> None:
            await asyncio.sleep(0.1)
            orch._running = False

        await asyncio.gather(orch.run(), stop_after_delay())

        esp32.emergency_stop.assert_awaited()

    async def test_estop_failure_does_not_propagate(self) -> None:
        """If emergency_stop itself fails, run() must not crash."""
        orch, esp32 = _make_orchestrator(tick_timeout_s=5.0)
        esp32.emergency_stop = AsyncMock(side_effect=OSError("serial disconnected"))

        async def failing_tick() -> None:
            msg = "boom"
            raise RuntimeError(msg)

        orch.tick = failing_tick  # type: ignore[assignment]
        orch._running = True

        async def stop_after_delay() -> None:
            await asyncio.sleep(0.1)
            orch._running = False

        # Must not raise despite both tick and emergency_stop failing
        await asyncio.gather(orch.run(), stop_after_delay())


class TestWatchdogNotify:
    """Watchdog.notify() is called after each successful tick."""

    async def test_watchdog_called_on_success(self) -> None:
        """After a successful tick, watchdog.notify() must be called."""
        watchdog = MagicMock()
        watchdog.notify = MagicMock()

        orch, esp32 = _make_orchestrator(tick_timeout_s=5.0, watchdog=watchdog)
        orch._running = True

        tick_count = 0
        original_tick = orch.tick

        async def counting_tick() -> None:
            nonlocal tick_count
            await original_tick()
            tick_count += 1
            if tick_count >= 3:
                orch._running = False

        orch.tick = counting_tick  # type: ignore[assignment]

        await orch.run()

        assert watchdog.notify.call_count >= 3

    async def test_watchdog_not_called_on_error(self) -> None:
        """After a failed tick, watchdog.notify() must NOT be called."""
        watchdog = MagicMock()
        watchdog.notify = MagicMock()

        orch, esp32 = _make_orchestrator(tick_timeout_s=5.0, watchdog=watchdog)
        orch._running = True

        async def failing_tick() -> None:
            orch._running = False
            msg = "crash"
            raise RuntimeError(msg)

        orch.tick = failing_tick  # type: ignore[assignment]

        await orch.run()

        watchdog.notify.assert_not_called()
