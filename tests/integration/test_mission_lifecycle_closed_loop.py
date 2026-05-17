"""Tier C2.3: full PENDING → RUNNING → SUCCEEDED + stall→replan via build_orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mousedroid.config.schema import MissionConfig, Settings
from mousedroid.llm_gateway.mission_parser import IntentType, MissionIntent
from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.orchestrator.mission_lifecycle import MissionLifecycleState


def _install_accepting_parser(orch: object, command: str) -> None:
    parser = MagicMock()
    parser.parse = MagicMock(
        return_value=MissionIntent(
            intent_type=IntentType.NAVIGATION,
            goal_vector=GoalVector(vx_target=0.4),
            confidence=0.99,
            raw_command=command,
        )
    )
    orch._mission_parser = parser  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_closed_loop_first_tick_after_mission_succeeds() -> None:
    """With mock VLM=0.95 > success_threshold=0.5, the scored tick succeeds."""
    from mousedroid.factory import build_orchestrator

    cfg = Settings(mock_hardware=True)
    cfg.mission = MissionConfig(
        replan_enabled=True,
        vlm_progress_enabled=True,
        llm_replanner_enabled=True,
        vlm_mock_progress_value=0.95,
        success_threshold=0.5,
    )
    cfg.llm.enabled = True

    orch = build_orchestrator(cfg)
    assert orch._mission_lifecycle is not None
    _install_accepting_parser(orch, "navigate to charger")


    await orch.process_mission("navigate to charger")
    assert orch._mission_lifecycle.current_state == MissionLifecycleState.RUNNING

    await orch.tick()  # caches prev_obs
    await orch.tick()  # scores via VLM -> SUCCEEDED
    assert orch._mission_lifecycle.current_state == MissionLifecycleState.SUCCEEDED


@pytest.mark.asyncio
async def test_closed_loop_stall_triggers_llm_replanner() -> None:
    """With mock VLM=0.0 and replanner returning a goal, lifecycle replans."""
    from mousedroid.factory import build_orchestrator

    cfg = Settings(mock_hardware=True)
    cfg.mission = MissionConfig(
        replan_enabled=True,
        vlm_progress_enabled=True,
        llm_replanner_enabled=True,
        vlm_mock_progress_value=0.0,
        success_threshold=0.5,
        stall_threshold=0.1,
        stall_window_ticks=2,
        max_replans_per_mission=3,
    )
    cfg.llm.enabled = True

    orch = build_orchestrator(cfg)
    assert orch._mission_lifecycle is not None

    # Override the replanner's gateway directly so we can script
    # ``translate_mission`` deterministically without an HTTP daemon.
    fake_gw = MagicMock()
    fake_gw.is_ready = True
    fake_gw.translate_mission = AsyncMock(return_value=GoalVector(vx_target=0.5))
    orch._mission_lifecycle._replanner._gateway = fake_gw

    _install_accepting_parser(orch, "walk")

    await orch.process_mission("walk")
    for _ in range(4):
        await orch.tick()

    assert orch._mission_lifecycle.current_state == MissionLifecycleState.RUNNING
    assert orch._mission_lifecycle.replan_count >= 1
    fake_gw.translate_mission.assert_awaited()
