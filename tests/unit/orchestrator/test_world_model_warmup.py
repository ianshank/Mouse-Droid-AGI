"""Off-loop world-model warmup (openspec task 2.4).

The failure this prevents is specific and unrecoverable. ``observe_step`` on the
ONNX engine builds its runtime session lazily on first call
(``dual_stream_rssm_onnx.py:223-224``); ``_update_world_model`` is **synchronous**
(``orchestrator/_world_model_state_mixin.py:28``); and ``run()`` wraps each tick
in ``asyncio.wait_for(self.tick(), tick_timeout_s)`` whose timeout path calls
``emergency_stop()`` (``_lifecycle_mixin.py:420-430``). ``wait_for`` cannot
preempt a synchronous call, so a cold TensorRT engine build on the first tick
does not merely make that tick slow — it blows the deadline and e-stops the
rover, then does it again on the next tick.

So the fix is not "make warmup faster", it is "make warmup happen somewhere a
long wait is harmless". These tests pin that:

* a ``Warmable`` engine is warmed inside ``start()``, before the loop runs;
* the warmup happens on a **worker thread**, so the event loop (and with it the
  shutdown signal handlers) stays live through a multi-second build;
* a warmup slower than ``tick_timeout_s`` still leaves the first tick inside
  budget — the direct statement of the bug;
* the default PyTorch path is untouched, because those engines are not
  ``Warmable`` at all.

``time.sleep`` is used deliberately rather than ``asyncio.sleep``: a TensorRT
build is a blocking syscall, and an ``await``-able fake would pass even against
the broken implementation this test exists to reject.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog.testing
import torch

from mousedroid.config.schema import ModelConfig, Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.world_model.protocol import WarmableProtocol


class _StubWorldModel:
    """Minimal deterministic engine with no warmup capability."""

    def __init__(self) -> None:
        self.observe_calls = 0

    def observe_step(
        self,
        observation: Any,
        prev_action: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        self.observe_calls += 1
        return 0.9 * h + 0.05, 0.9 * z - 0.05, h.clone(), 0.1

    def imagine_step(
        self, action: torch.Tensor, h: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return h, z, torch.zeros(1, 1)


class _WarmableStubWorldModel(_StubWorldModel):
    """Engine whose ``warmup`` blocks, like a cold TensorRT engine build.

    Records enough to prove *where* and *when* the warmup ran: the thread it
    executed on, and whether any ``observe_step`` was served while still cold.
    """

    def __init__(self, *, delay_s: float = 0.0, fail: bool = False) -> None:
        super().__init__()
        self._delay_s = delay_s
        self._fail = fail
        self.warmup_calls = 0
        self.warmup_thread: int | None = None
        self.warm = False
        self.observed_while_cold = 0

    def warmup(self) -> None:
        """Block for ``delay_s``, then mark the engine warm. Idempotent."""
        self.warmup_calls += 1
        if self.warm:
            return
        self.warmup_thread = threading.get_ident()
        time.sleep(self._delay_s)
        if self._fail:
            msg = "engine build failed"
            raise RuntimeError(msg)
        self.warm = True

    def observe_step(
        self,
        observation: Any,
        prev_action: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if not self.warm:
            self.observed_while_cold += 1
        return super().observe_step(observation, prev_action, h, z)


def _settings(*, tick_timeout_s: float | None = None) -> Settings:
    cfg = Settings(mock_hardware=True)
    if tick_timeout_s is not None:
        cfg.loop.tick_timeout_s = tick_timeout_s
    return cfg


def _build_orch(cfg: Settings, world_model: object) -> MouseDroidOrchestrator:
    """Build an orchestrator whose ``start()`` and ``tick()`` touch no hardware."""
    agent = MagicMock(name="agent")
    agent.name = "mock_agent"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)
    safety_monitor = MagicMock(name="safety_monitor")
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)
    sensor_manager = AsyncMock(name="sensor_manager")
    sensor_manager.read_all = AsyncMock(return_value=MagicMock())
    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=AsyncMock(name="esp32"),
        sensor_manager=sensor_manager,
        cfg=cfg,
    )


class TestCapabilityDetection:
    """Only engines that declare ``warmup`` are warmed."""

    def test_the_warmable_stub_satisfies_the_protocol(self) -> None:
        assert isinstance(_WarmableStubWorldModel(), WarmableProtocol)

    def test_an_engine_without_warmup_does_not(self) -> None:
        assert not isinstance(_StubWorldModel(), WarmableProtocol)

    def test_the_torch_engine_is_not_warmable(self) -> None:
        """The default ``engine: torch`` path must be byte-identical to before."""
        from mousedroid.world_model.rssm import RSSM

        model = RSSM(
            ModelConfig(
                vision_dim=16,
                ultrasonic_dim=1,
                ultrasonic_proj_dim=4,
                motor_state_dim=4,
                hidden_dim=32,
                latent_dim=8,
                action_dim=2,
                obs_dim=16,
                vision_proj_dim=8,
                motor_proj_dim=4,
                cfc_hidden_dim=0,
            )
        )
        assert not isinstance(model, WarmableProtocol)


@pytest.mark.asyncio
class TestStartWarmsTheEngine:
    """``start()`` is where the wait belongs."""

    async def test_start_calls_warmup(self) -> None:
        wm = _WarmableStubWorldModel()
        await _build_orch(_settings(), wm).start()
        assert wm.warmup_calls == 1

    async def test_the_engine_is_warm_before_start_returns(self) -> None:
        wm = _WarmableStubWorldModel(delay_s=0.05)
        await _build_orch(_settings(), wm).start()
        assert wm.warm, "start() must not return with a cold engine"

    async def test_warmup_runs_on_a_worker_thread(self) -> None:
        """``asyncio.to_thread``, not a direct call — the event loop must stay free."""
        wm = _WarmableStubWorldModel(delay_s=0.01)
        await _build_orch(_settings(), wm).start()
        assert wm.warmup_thread is not None
        assert wm.warmup_thread != threading.get_ident(), (
            "a synchronous warmup on the event loop thread would block the "
            "shutdown handlers for the whole engine build"
        )

    async def test_an_engine_without_warmup_starts_cleanly(self) -> None:
        wm = _StubWorldModel()
        orch = _build_orch(_settings(), wm)
        await orch.start()
        assert orch._running is True

    async def test_the_event_loop_stays_responsive_during_warmup(self) -> None:
        """A concurrent task must make progress while the engine builds."""
        wm = _WarmableStubWorldModel(delay_s=0.2)
        orch = _build_orch(_settings(), wm)
        heartbeats = 0

        async def _heartbeat() -> None:
            nonlocal heartbeats
            while True:
                await asyncio.sleep(0.01)
                heartbeats += 1

        beat = asyncio.create_task(_heartbeat())
        try:
            await orch.start()
        finally:
            beat.cancel()
        assert heartbeats > 1, (
            "the event loop was starved during warmup, so a shutdown signal "
            "arriving mid-build would not have been serviced"
        )

    async def test_warmup_is_logged_with_its_duration(self) -> None:
        wm = _WarmableStubWorldModel(delay_s=0.01)
        with structlog.testing.capture_logs() as logs:
            await _build_orch(_settings(), wm).start()
        events = [entry["event"] for entry in logs]
        assert "world_model_warmup_starting" in events
        completions = [entry for entry in logs if entry["event"] == "world_model_warmup_complete"]
        assert len(completions) == 1
        assert float(completions[0]["seconds"]) >= 0.0


@pytest.mark.asyncio
class TestSlowWarmupNeverReachesTheTick:
    """The bug, stated directly."""

    async def test_a_warmup_slower_than_the_tick_timeout_is_paid_at_boot(self) -> None:
        tick_timeout_s = 0.1
        wm = _WarmableStubWorldModel(delay_s=tick_timeout_s * 3)
        orch = _build_orch(_settings(tick_timeout_s=tick_timeout_s), wm)

        await orch.start()
        # Exactly what ``run()`` does, for one tick.
        await asyncio.wait_for(orch.tick(), timeout=tick_timeout_s)

        assert wm.observed_while_cold == 0, (
            "the tick served an observation against a cold engine; lazily "
            "warming there is what blows tick_timeout_s and e-stops the rover"
        )
        assert wm.observe_calls == 1

    async def test_the_first_tick_is_not_slowed_by_the_build(self) -> None:
        wm = _WarmableStubWorldModel(delay_s=0.3)
        orch = _build_orch(_settings(), wm)
        await orch.start()

        started = time.perf_counter()
        await orch.tick()
        elapsed = time.perf_counter() - started

        assert elapsed < 0.3, (
            f"the first tick took {elapsed:.3f}s, which means it paid for the "
            "engine build instead of start() paying for it"
        )


@pytest.mark.asyncio
class TestWarmupFailure:
    """A build that cannot succeed must refuse to start, not start doomed."""

    async def test_failure_propagates_out_of_start(self) -> None:
        wm = _WarmableStubWorldModel(fail=True)
        orch = _build_orch(_settings(), wm)
        with pytest.raises(RuntimeError, match="engine build failed"):
            await orch.start()

    async def test_failure_is_logged_before_it_propagates(self) -> None:
        wm = _WarmableStubWorldModel(fail=True)
        orch = _build_orch(_settings(), wm)
        with structlog.testing.capture_logs() as logs, pytest.raises(RuntimeError):
            await orch.start()
        assert any(entry["event"] == "world_model_warmup_failed" for entry in logs)

    async def test_the_loop_is_not_armed_after_a_failed_warmup(self) -> None:
        """``_running`` is set after the warmup, so a failure leaves it false."""
        wm = _WarmableStubWorldModel(fail=True)
        orch = _build_orch(_settings(), wm)
        with pytest.raises(RuntimeError):
            await orch.start()
        assert orch._running is False
