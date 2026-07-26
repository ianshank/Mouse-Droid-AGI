"""End-to-end: MissionLifecycle + dual WeightUpdatePoller + SafetyProjector
fire correctly within a single orchestrator tick without interaction defects.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from mousedroid.cloud.protocol import PendingWeightUpdate
from mousedroid.config.schema import MissionConfig, Settings
from mousedroid.orchestrator.mission_lifecycle import (
    MissionLifecycle,
    MissionLifecycleState,
)
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.safety.projector import GeometricSafetyProjector
from tests.unit.orchestrator.test_weight_update_swap import _StubPoller


@pytest.mark.asyncio
async def test_tier_c_features_coexist_on_single_tick() -> None:
    """Lifecycle + dual poller + projector coexist on a single tick."""
    cfg = Settings(mock_hardware=True)
    cfg.mission = MissionConfig(replan_enabled=True)
    cfg.safety.projector.enabled = True
    cfg.cloud.weight_update.poll_interval_s = 1.0
    cfg.cloud.weight_update.world_model_enabled = True

    policy_p = _StubPoller(
        [
            PendingWeightUpdate(
                repo_id="x",
                filename="p.bin",
                revision="r1",
                sha256="0" * 64,
                local_path=Path("/tmp/p.bin"),
                downloaded_at=time.time(),
                engine_type="policy",
            ),
        ]
    )
    wm_p = _StubPoller([])  # no pending world-model update this tick

    lifecycle = MagicMock(spec=MissionLifecycle)
    lifecycle.tick = AsyncMock()
    lifecycle.current_state = MissionLifecycleState.RUNNING

    projector = GeometricSafetyProjector(cfg.safety.projector)
    loader = MagicMock(return_value=MagicMock())

    wm = MagicMock()
    combined = cfg.model.hidden_dim + cfg.model.cfc_hidden_dim
    wm.observe_step.return_value = (
        torch.zeros(1, combined),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, combined),
        0.1,
    )
    agent = MagicMock()
    agent.name = "x"
    agent.act.return_value = torch.zeros(cfg.model.action_dim)
    sm = MagicMock()
    sm.evaluate.return_value = SafetyContext(is_emergency=False)

    obs = MagicMock()
    obs.vision_features = np.zeros(8, dtype=np.float32)
    sensor_manager = MagicMock()
    sensor_manager.read_all = AsyncMock(return_value=obs)

    orch = MouseDroidOrchestrator(
        world_model=wm,
        agents=[agent],
        safety_monitor=sm,
        esp32=AsyncMock(),
        sensor_manager=sensor_manager,
        cfg=cfg,
        weight_update_pollers={"policy": policy_p, "world_model": wm_p},
        weight_update_loader=loader,
        mission_lifecycle=lifecycle,
        safety_projector=projector,
    )

    # Two ticks so the lifecycle has both obs_t and obs_tminus1 cached.
    await orch.tick()
    await orch.tick()

    assert lifecycle.tick.await_count == 1, (
        "lifecycle.tick must fire once at POST_TICK on the second tick"
    )
    assert len(policy_p.ack_calls) == 1, "policy poller must swap + ack on tick 1"
    assert len(wm_p.ack_calls) == 0, "wm poller had no pending update — must not swap"


@pytest.mark.asyncio
async def test_tier_c_features_wired_via_build_orchestrator_defaults_off() -> None:
    """Defaults-off path (post-Tier-C2.3): lifecycle still None.

    With ``replan_enabled=True`` but ``vlm_progress_enabled=False`` and
    ``llm_replanner_enabled=False`` (the defaults), the factory still
    short-circuits to ``None`` — exactly as in the pre-Tier-C2.3 path.
    Safety projector + dual pollers remain non-None.
    """
    from mousedroid.factory import build_orchestrator

    cfg = Settings(mock_hardware=True)
    cfg.mission = MissionConfig(replan_enabled=True)
    cfg.safety.projector.enabled = True
    cfg.cloud.weight_update.poll_interval_s = 1.0
    cfg.cloud.weight_update.world_model_enabled = True

    orch = build_orchestrator(cfg)
    assert orch._mission_lifecycle is None
    assert orch._safety_projector is not None
    assert set(orch._weight_update_pollers.keys()) == {"policy", "world_model"}


@pytest.mark.asyncio
async def test_tier_c_features_wired_via_build_orchestrator_full_activation() -> None:
    """Tier C2.3: with all three flags on the lifecycle is fully wired.

    Inverts the PR #98 assertion — ``build_orchestrator`` now threads
    the VLM head + LLM replanner into :func:`build_mission_lifecycle`,
    so the lifecycle stops being a permanent ``None`` in production.
    """
    from mousedroid.factory import build_orchestrator

    cfg = Settings(mock_hardware=True)
    cfg.mission = MissionConfig(
        replan_enabled=True,
        vlm_progress_enabled=True,
        llm_replanner_enabled=True,
    )
    cfg.safety.projector.enabled = True
    cfg.cloud.weight_update.poll_interval_s = 1.0
    cfg.cloud.weight_update.world_model_enabled = True
    cfg.llm.enabled = True  # NOTE: field is cfg.llm, NOT cfg.llm_gateway.

    orch = build_orchestrator(cfg)
    assert orch._mission_lifecycle is not None
    assert orch._safety_projector is not None
    assert set(orch._weight_update_pollers.keys()) == {"policy", "world_model"}
