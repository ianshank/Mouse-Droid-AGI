"""The safety interlock must see the whole tick, and Prometheus every tick.

Two pins, both red against the pre-fix orchestrator.

**Pin A — the interlock is armed.** ``tick()`` used to compute ``loop_time_ms``
immediately after ``sensor_manager.read_all()`` and never recompute it, then
hand that value to ``safety_monitor.evaluate``. World-model update, planning,
actuation and telemetry all run *after* the measurement, so a tick that spent
300 ms in the planner reported a healthy few milliseconds and the loop-overrun
emergency stop could not fire. The real duration was computed at
``_lifecycle_mixin.py`` and used only to size the sleep.

**Pin B — no sampling hole.** The loop-time gauge and histogram were written
only from the telemetry frame path, which is throttled to
``telemetry.publish_hz`` (10 Hz) while the loop runs at ``loop.control_hz``
(30 Hz). Two ticks in three produced no observation at all, so a tail-latency
regression affecting a minority of ticks was structurally invisible.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from mousedroid.common.time.protocol import MockClock
from mousedroid.config.schema import Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.telemetry.metrics.registry import MetricsRegistry

_SENSE_MS = 5.0
_ACT_MS = 300.0


def _observation(cfg: Settings) -> MouseDroidObservationBundle:
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(1024, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )


def _make_orchestrator(
    cfg: Settings, clock: MockClock, metrics: MetricsRegistry | None = None
) -> MouseDroidOrchestrator:
    """Orchestrator whose sense phase costs 5 ms and whose act phase costs 300 ms."""
    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )

    agent = MagicMock()
    agent.name = "timing_agent"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)

    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)

    async def _read_all() -> MouseDroidObservationBundle:
        clock.advance(_SENSE_MS / 1000.0)
        return _observation(cfg)

    sensor_manager = AsyncMock()
    sensor_manager.read_all = _read_all

    esp32 = AsyncMock()

    async def _send_velocity(*_args: object, **_kwargs: object) -> None:
        # Stands in for the planner + actuation cost that lands AFTER the old
        # measurement point.
        clock.advance(_ACT_MS / 1000.0)

    esp32.send_velocity = _send_velocity

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        clock=clock,
        metrics=metrics,
    )


@pytest.mark.asyncio
async def test_safety_monitor_receives_the_whole_tick_not_the_sensor_read() -> None:
    """Pin A. Red pre-fix: ``evaluate`` was handed 5.0 on every call."""
    cfg = Settings(mock_hardware=True)
    clock = MockClock(start=0.0)
    orch = _make_orchestrator(cfg, clock)

    await orch.tick()  # tick 0 — no previous duration to report
    await orch.tick()  # tick 1 — must see tick 0's TOTAL

    second_call = orch._safety_monitor.evaluate.call_args_list[1]
    observed_ms = second_call.args[1]
    assert observed_ms == pytest.approx(_SENSE_MS + _ACT_MS, abs=1.0), (
        f"safety monitor received {observed_ms} ms; it must receive the "
        f"PREVIOUS tick's full duration ({_SENSE_MS + _ACT_MS} ms = sense + "
        f"act), not the {_SENSE_MS} ms sensor-read segment alone. With the "
        "segment, a tick that blows its budget in the planner reports healthy "
        "and the loop-overrun emergency stop never fires."
    )


@pytest.mark.asyncio
async def test_first_tick_reports_zero_and_cannot_trip() -> None:
    """Tick 0 has no predecessor, so it must be structurally unable to trip."""
    cfg = Settings(mock_hardware=True)
    clock = MockClock(start=0.0)
    orch = _make_orchestrator(cfg, clock)

    await orch.tick()

    first_call = orch._safety_monitor.evaluate.call_args_list[0]
    assert first_call.args[1] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_every_tick_reaches_prometheus_despite_the_publish_throttle() -> None:
    """Pin B. Red pre-fix: the orchestrator never wrote the registry at all.

    ``publish_hz`` is a third of ``control_hz``, so a frame-driven metric
    writer samples two ticks in three away.
    """
    cfg = Settings(
        mock_hardware=True,
        loop={"control_hz": 30.0},
        telemetry={"publish_hz": 10.0},
    )
    clock = MockClock(start=0.0)
    registry = MetricsRegistry(cfg.metrics)
    orch = _make_orchestrator(cfg, clock, metrics=registry)

    for _ in range(3):
        await orch.tick()

    rendered = registry.render_prometheus()
    assert "mousedroid_loop_latency_ms_count 3" in rendered, (
        "expected one loop-latency observation per tick. Found:\n"
        + "\n".join(line for line in rendered.splitlines() if "loop_latency" in line)
    )


@pytest.mark.asyncio
async def test_per_phase_timings_are_recorded_every_tick() -> None:
    """The per-stage breakdown must exist, and attribute cost to the right phase."""
    cfg = Settings(mock_hardware=True)
    clock = MockClock(start=0.0)
    registry = MetricsRegistry(cfg.metrics)
    orch = _make_orchestrator(cfg, clock, metrics=registry)

    await orch.tick()
    await orch.tick()

    rendered = registry.render_prometheus()
    for phase in ("sense", "act", "plan", "post"):
        assert f'mousedroid_tick_phase_ms_count{{phase="{phase}"}} 2' in rendered, (
            f"phase {phase!r} must record one sample per tick"
        )
    # The 300 ms cost belongs to `act`, not to `sense`.
    assert f'mousedroid_tick_phase_ms_sum{{phase="sense"}} {_SENSE_MS * 2:g}' in rendered
    act_sum = next(
        line for line in rendered.splitlines() if 'tick_phase_ms_sum{phase="act"}' in line
    )
    assert float(act_sum.rsplit(" ", 1)[1]) == pytest.approx(_ACT_MS * 2, abs=2.0)


@pytest.mark.asyncio
async def test_a_tick_that_raises_still_latches_its_duration() -> None:
    """The `finally` must record even when the tick blows up.

    A tick that is chronically slow *because* it keeps failing must not stay
    invisible to the next tick's interlock.
    """
    cfg = Settings(mock_hardware=True)
    clock = MockClock(start=0.0)
    orch = _make_orchestrator(cfg, clock)
    orch._update_world_model = MagicMock(side_effect=RuntimeError("planner exploded"))

    with pytest.raises(RuntimeError, match="planner exploded"):
        await orch.tick()

    assert orch._last_tick_ms == pytest.approx(_SENSE_MS, abs=1.0), (
        "a raising tick must still latch its measured duration"
    )
