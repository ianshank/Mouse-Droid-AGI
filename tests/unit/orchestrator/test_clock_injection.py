"""Tests for ClockProtocol injection into MouseDroidOrchestrator.

Verifies that:
* The orchestrator defaults to RealClock when no clock is supplied.
* A MockClock can be injected and controls simulated time without
  any wall-clock delays.
* Ticks fire correctly when the MockClock is advanced.
* The consolidation loop wakes on MockClock advances.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import torch

from mousedroid.common.time.protocol import MockClock, RealClock
from mousedroid.config.schema import Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext


def _make_orchestrator(
    cfg: Settings,
    clock: MockClock | None = None,
) -> MouseDroidOrchestrator:
    """Build a minimal orchestrator with all optional subsystems absent."""
    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )
    agent = MagicMock()
    agent.name = "mock_agent"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)

    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)

    sensor_manager = AsyncMock()
    sensor_manager.read_all = AsyncMock(return_value=MagicMock())

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=AsyncMock(),
        sensor_manager=sensor_manager,
        cfg=cfg,
        clock=clock,
    )


class TestClockDefaultsToRealClock:
    """When no clock is supplied, orchestrator uses RealClock."""

    def test_default_clock_is_real_clock(self) -> None:
        cfg = Settings(mock_hardware=True)
        orch = _make_orchestrator(cfg)
        assert isinstance(orch._clock, RealClock)

    def test_injected_mock_clock_is_stored(self) -> None:
        cfg = Settings(mock_hardware=True)
        clock = MockClock(start=0.0)
        orch = _make_orchestrator(cfg, clock=clock)
        assert orch._clock is clock


class TestMockClockControlsTicks:
    """MockClock.advance() drives the orchestrator's time without wall delays."""

    async def test_run_loop_sleeps_on_mock_clock(self) -> None:
        cfg = Settings(mock_hardware=True)
        clock = MockClock(start=0.0)
        orch = _make_orchestrator(cfg, clock=clock)

        tick_count = 0
        original_tick = orch.tick

        async def counting_tick() -> None:
            nonlocal tick_count
            tick_count += 1
            await original_tick()

        orch.tick = counting_tick  # type: ignore[method-assign]

        orch._running = True

        run_task = asyncio.create_task(orch.run())
        await asyncio.sleep(0)  # let run() reach its first clock.sleep

        # Advance past two full control periods (30 Hz → ~33 ms each)
        control_period = 1.0 / cfg.loop.control_hz
        clock.advance(control_period * 2 + 0.001)
        await asyncio.sleep(0)

        orch._running = False
        clock.advance(control_period + 0.1)  # unblock final sleep
        await asyncio.sleep(0)

        run_task.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await run_task

        assert tick_count >= 1

    async def test_no_wall_clock_time_consumed(self) -> None:
        """Running the orchestrator loop with MockClock is nearly instant."""
        import time as _time

        cfg = Settings(mock_hardware=True)
        clock = MockClock(start=0.0)
        orch = _make_orchestrator(cfg, clock=clock)

        orch._running = True

        run_task = asyncio.create_task(orch.run())
        await asyncio.sleep(0)

        wall_before = _time.monotonic()
        # Advance 10 seconds of simulated time
        clock.advance(10.0)
        await asyncio.sleep(0)
        wall_elapsed = _time.monotonic() - wall_before

        orch._running = False
        clock.advance(1.0)
        await asyncio.sleep(0)

        run_task.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await run_task

        # 10 simulated seconds should complete in well under 1 real second
        assert wall_elapsed < 1.0

    async def test_consolidation_loop_wakes_on_mock_clock(self) -> None:
        """The background consolidation loop uses the injected clock."""
        cfg = Settings(mock_hardware=True)
        clock = MockClock(start=0.0)
        orch = _make_orchestrator(cfg, clock=clock)

        consolidate_calls = 0

        memory_tier = MagicMock()
        memory_tier.semantic = MagicMock()
        memory_tier.semantic.size = 0

        def fake_consolidate() -> int:
            nonlocal consolidate_calls
            consolidate_calls += 1
            return 0

        memory_tier.consolidation = MagicMock()
        memory_tier.consolidation.consolidate = fake_consolidate
        orch._memory_tier = memory_tier

        loop_task = asyncio.create_task(orch._consolidation_loop())
        await asyncio.sleep(0)  # let loop reach its clock.sleep

        interval = cfg.memory.consolidation_interval_s
        clock.advance(interval + 0.01)
        await asyncio.sleep(0)

        loop_task.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await loop_task

        assert consolidate_calls >= 1
