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
