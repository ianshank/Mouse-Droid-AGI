"""Golden characterization tests for the slow-cadence background loops.

Pins the exact structlog event sequence emitted by ``_on_device_update_loop``,
``_growth_distill_loop``, and ``_consolidation_loop`` BEFORE the
``_run_slow_cadence_loop`` extraction (Item 3a of the skills/hooks build-out
plan) — and must still pass byte-for-byte identically after it. This repo's
own documented methodology for this class of refactor (AGENTS.md rule #9;
ADR-014 did the same for ``render_prometheus`` before decomposing it): capture
the golden log stream first, then confirm the refactor changes nothing about
what gets logged or when.

Each loop is driven through exactly one cycle via an injected ``MockClock``
(no wall-clock waiting), with the coordinator/memory-tier collaborator
replaced by a small controlled fake so the captured events are the LOOP's
own, undiluted by a real coordinator's internal logging.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Coroutine
from typing import Any

import pytest
import structlog.testing

from mousedroid.common.time.protocol import MockClock
from mousedroid.config.schema import Settings
from mousedroid.factory import build_orchestrator


async def _run_one_cycle(
    coro: Coroutine[Any, Any, None], clock: MockClock, interval: float
) -> list[str]:
    """Drive ``coro`` (a slow-cadence loop) through one sleep-wake cycle.

    Advances ``clock`` past ``interval`` so the loop's single
    ``await self._clock.sleep(interval)`` resolves, lets the resulting cycle
    (and any ``asyncio.to_thread`` work inside it) settle, then cancels the
    still-running infinite loop and returns the captured event names in order.
    """
    with structlog.testing.capture_logs() as logs:
        task = asyncio.create_task(coro)
        await asyncio.sleep(0)  # let the loop reach its first clock.sleep
        clock.advance(interval + 0.01)
        # A short REAL sleep (not a bare tick): the consolidation cycle runs
        # its work via asyncio.to_thread, which needs an actual OS thread
        # pool round-trip to complete, not just one event-loop tick.
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return [entry["event"] for entry in logs]


@pytest.mark.asyncio
class TestOnDeviceUpdateLoopEventStream:
    async def test_successful_cycle(self) -> None:
        cfg = Settings.model_validate(
            {"mock_hardware": True, "on_device_learning": {"enabled": True}}
        )
        clock = MockClock(start=0.0)
        orch = build_orchestrator(cfg)
        orch._clock = clock  # type: ignore[attr-defined]

        calls = 0

        async def _maybe_update() -> None:
            nonlocal calls
            calls += 1

        orch._on_device_coordinator.maybe_update = _maybe_update  # type: ignore[attr-defined]

        events = await _run_one_cycle(
            orch._on_device_update_loop(),  # type: ignore[attr-defined]
            clock,
            cfg.on_device_learning.check_interval_s,  # type: ignore[union-attr]
        )
        assert events == ["on_device_update_loop_started"]
        assert calls == 1

    async def test_failed_cycle_logs_warning(self) -> None:
        cfg = Settings.model_validate(
            {"mock_hardware": True, "on_device_learning": {"enabled": True}}
        )
        clock = MockClock(start=0.0)
        orch = build_orchestrator(cfg)
        orch._clock = clock  # type: ignore[attr-defined]

        async def _maybe_update() -> None:
            raise RuntimeError("boom")

        orch._on_device_coordinator.maybe_update = _maybe_update  # type: ignore[attr-defined]

        events = await _run_one_cycle(
            orch._on_device_update_loop(),  # type: ignore[attr-defined]
            clock,
            cfg.on_device_learning.check_interval_s,  # type: ignore[union-attr]
        )
        assert events == ["on_device_update_loop_started", "on_device_update_cycle_failed"]


@pytest.mark.asyncio
class TestGrowthDistillLoopEventStream:
    async def test_successful_cycle(self) -> None:
        cfg = Settings.model_validate(
            {"mock_hardware": True, "vla": {"backend": "mock"}, "growth": {"enabled": True}}
        )
        clock = MockClock(start=0.0)
        orch = build_orchestrator(cfg)
        orch._clock = clock  # type: ignore[attr-defined]

        calls = 0

        async def _maybe_distill() -> None:
            nonlocal calls
            calls += 1

        orch._growth_coordinator.maybe_distill = _maybe_distill  # type: ignore[attr-defined]

        events = await _run_one_cycle(
            orch._growth_distill_loop(),  # type: ignore[attr-defined]
            clock,
            cfg.growth.check_interval_s,  # type: ignore[union-attr]
        )
        assert events == ["growth_distill_loop_started"]
        assert calls == 1

    async def test_failed_cycle_logs_warning(self) -> None:
        cfg = Settings.model_validate(
            {"mock_hardware": True, "vla": {"backend": "mock"}, "growth": {"enabled": True}}
        )
        clock = MockClock(start=0.0)
        orch = build_orchestrator(cfg)
        orch._clock = clock  # type: ignore[attr-defined]

        async def _maybe_distill() -> None:
            raise RuntimeError("boom")

        orch._growth_coordinator.maybe_distill = _maybe_distill  # type: ignore[attr-defined]

        events = await _run_one_cycle(
            orch._growth_distill_loop(),  # type: ignore[attr-defined]
            clock,
            cfg.growth.check_interval_s,  # type: ignore[union-attr]
        )
        assert events == ["growth_distill_loop_started", "growth_distill_cycle_failed"]


@pytest.mark.asyncio
class TestConsolidationLoopEventStream:
    def _memory_tier(self, *, consolidate_result: int = 0, raises: bool = False) -> Any:
        from unittest.mock import MagicMock

        memory_tier = MagicMock()
        memory_tier.semantic = MagicMock()
        memory_tier.semantic.size = 0

        def _consolidate() -> int:
            if raises:
                raise RuntimeError("boom")
            return consolidate_result

        memory_tier.consolidation = MagicMock()
        memory_tier.consolidation.consolidate = _consolidate
        return memory_tier

    async def test_successful_cycle_with_records_logs_debug_complete(self) -> None:
        cfg = Settings.model_validate({"mock_hardware": True})
        clock = MockClock(start=0.0)
        orch = build_orchestrator(cfg)
        orch._clock = clock  # type: ignore[attr-defined]
        orch._memory_tier = self._memory_tier(consolidate_result=1)  # type: ignore[attr-defined]

        events = await _run_one_cycle(
            orch._consolidation_loop(),  # type: ignore[attr-defined]
            clock,
            cfg.memory.consolidation_interval_s,
        )
        assert events == ["consolidation_loop_started", "consolidation_cycle_complete"]

    async def test_successful_cycle_with_no_records_logs_nothing_extra(self) -> None:
        cfg = Settings.model_validate({"mock_hardware": True})
        clock = MockClock(start=0.0)
        orch = build_orchestrator(cfg)
        orch._clock = clock  # type: ignore[attr-defined]
        orch._memory_tier = self._memory_tier(consolidate_result=0)  # type: ignore[attr-defined]

        events = await _run_one_cycle(
            orch._consolidation_loop(),  # type: ignore[attr-defined]
            clock,
            cfg.memory.consolidation_interval_s,
        )
        assert events == ["consolidation_loop_started"]

    async def test_failed_cycle_logs_warning(self) -> None:
        cfg = Settings.model_validate({"mock_hardware": True})
        clock = MockClock(start=0.0)
        orch = build_orchestrator(cfg)
        orch._clock = clock  # type: ignore[attr-defined]
        orch._memory_tier = self._memory_tier(raises=True)  # type: ignore[attr-defined]

        events = await _run_one_cycle(
            orch._consolidation_loop(),  # type: ignore[attr-defined]
            clock,
            cfg.memory.consolidation_interval_s,
        )
        assert events == ["consolidation_loop_started", "consolidation_cycle_failed"]
