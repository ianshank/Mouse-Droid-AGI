"""C2.1: orchestrator wires + ticks MissionLifecycle once per POST_TICK."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from mousedroid.config.schema import MissionConfig, Settings
from mousedroid.orchestrator.mission_lifecycle import (
    MissionLifecycle,
    MissionLifecycleState,
)
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext


def _build_orch_with_lifecycle(
    cfg: Settings, lifecycle: MissionLifecycle | None
) -> MouseDroidOrchestrator:
    """Build a minimally-wired orchestrator for lifecycle wiring tests."""
    wm = MagicMock()
    combined = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
    wm.observe_step.return_value = (
        torch.zeros(1, combined),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, combined),
        0.1,
    )
    agent = MagicMock()
    agent.name = "mock"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)
    sm = MagicMock()
    sm.evaluate.return_value = SafetyContext(is_emergency=False)

    obs = MagicMock()
    obs.vision_features = np.zeros(8, dtype=np.float32)

    sensor_manager = MagicMock()
    sensor_manager.read_all = AsyncMock(return_value=obs)

    return MouseDroidOrchestrator(
        world_model=wm,
        agents=[agent],
        safety_monitor=sm,
        esp32=AsyncMock(),
        sensor_manager=sensor_manager,
        cfg=cfg,
        mission_lifecycle=lifecycle,
    )


@pytest.mark.asyncio
async def test_mission_lifecycle_tick_called_at_post_tick() -> None:
    """lifecycle.tick fires exactly once after prev_obs is cached (tick 2)."""
    cfg = Settings(mock_hardware=True)
    cfg.mission = MissionConfig(replan_enabled=True)
    lifecycle = MagicMock(spec=MissionLifecycle)
    lifecycle.tick = AsyncMock()
    lifecycle.current_state = MissionLifecycleState.RUNNING
    orch = _build_orch_with_lifecycle(cfg, lifecycle)

    # First tick: caches prev_obs, no lifecycle.tick yet (needs prev frame).
    await orch.tick()
    assert lifecycle.tick.await_count == 0

    # Second tick: prev_obs available, lifecycle should fire exactly once.
    await orch.tick()
    assert lifecycle.tick.await_count == 1


@pytest.mark.asyncio
async def test_mission_lifecycle_none_is_noop() -> None:
    """No lifecycle wired -> helper short-circuits cleanly across multiple ticks."""
    cfg = Settings(mock_hardware=True)
    orch = _build_orch_with_lifecycle(cfg, lifecycle=None)
    # Must not raise; guard branch holds.
    await orch.tick()
    await orch.tick()


@pytest.mark.asyncio
async def test_mission_lifecycle_prev_obs_owns_buffer_after_caching() -> None:
    """Cached _prev_obs_for_vlm must NOT alias the sensor's numpy buffer."""
    cfg = Settings(mock_hardware=True)
    cfg.mission = MissionConfig(replan_enabled=True)
    lifecycle = MagicMock(spec=MissionLifecycle)
    lifecycle.tick = AsyncMock()

    # Mutate the camera's vision_features buffer between ticks to simulate
    # ring-buffer recycling — the lifecycle's prev_t arg on tick 2 must
    # carry tick 1's data (zeros), NOT the mutated tick 2 data.
    shared_buffer = np.zeros(8, dtype=np.float32)
    obs = MagicMock()
    obs.vision_features = shared_buffer
    sensor_manager = MagicMock()
    sensor_manager.read_all = AsyncMock(return_value=obs)

    wm = MagicMock()
    combined = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
    wm.observe_step.return_value = (
        torch.zeros(1, combined),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, combined),
        0.1,
    )
    agent = MagicMock()
    agent.name = "mock"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)
    sm = MagicMock()
    sm.evaluate.return_value = SafetyContext(is_emergency=False)

    orch = MouseDroidOrchestrator(
        world_model=wm,
        agents=[agent],
        safety_monitor=sm,
        esp32=AsyncMock(),
        sensor_manager=sensor_manager,
        cfg=cfg,
        mission_lifecycle=lifecycle,
    )

    await orch.tick()  # cache zeros
    shared_buffer[:] = 7.0  # simulate ring-buffer recycling
    await orch.tick()  # should fire with prev=zeros, NOT prev=sevens

    assert lifecycle.tick.await_count == 1
    args, _kwargs = lifecycle.tick.await_args
    _obs_t, prev_t = args
    # prev_t was cached on tick 1 (zeros) and must NOT have been mutated
    # by the shared_buffer[:] = 7.0 between ticks.
    assert torch.all(prev_t == 0.0).item(), f"prev_t leaked the shared buffer mutation: {prev_t}"


@pytest.mark.asyncio
async def test_mission_lifecycle_skipped_for_empty_vision_features() -> None:
    """vision_features.size == 0 → helper early-returns before caching/ticking.

    Mock-hardware cold-start emits a zero-length fallback array before the
    camera warms up. The lifecycle must NOT receive a zero-d tensor (which
    would crash downstream VLM scoring) — the early-return at the top of
    ``_maybe_tick_mission_lifecycle`` guards this path.
    """
    cfg = Settings(mock_hardware=True)
    cfg.mission = MissionConfig(replan_enabled=True)
    lifecycle = MagicMock(spec=MissionLifecycle)
    lifecycle.tick = AsyncMock()

    obs = MagicMock()
    obs.vision_features = np.zeros(0, dtype=np.float32)  # degenerate fallback
    sensor_manager = MagicMock()
    sensor_manager.read_all = AsyncMock(return_value=obs)

    wm = MagicMock()
    combined = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
    wm.observe_step.return_value = (
        torch.zeros(1, combined),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, combined),
        0.1,
    )
    agent = MagicMock()
    agent.name = "mock"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)
    sm = MagicMock()
    sm.evaluate.return_value = SafetyContext(is_emergency=False)

    orch = MouseDroidOrchestrator(
        world_model=wm,
        agents=[agent],
        safety_monitor=sm,
        esp32=AsyncMock(),
        sensor_manager=sensor_manager,
        cfg=cfg,
        mission_lifecycle=lifecycle,
    )

    await orch.tick()
    await orch.tick()
    # Lifecycle must NOT have been ticked because vision_features was empty.
    assert lifecycle.tick.await_count == 0
    # And no prev_obs cache was populated either (size==0 returns BEFORE caching).
    assert orch._prev_obs_for_vlm is None
