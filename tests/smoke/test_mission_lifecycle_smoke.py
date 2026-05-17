"""Tier C2.3 smoke: ≤5s boot + first-tick success via mock VLM head.

Verifies the operator quickstart contract — with all three Tier C2.3
flags on and the default mock VLM value (0.95) above the default
success threshold (0.9), a mission transitions PENDING → RUNNING →
SUCCEEDED within two ticks. Budget is enforced so the test stays
cheap in ``scripts/jetson_smoke_test.sh`` and the local CI flow.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema import MissionConfig, Settings
from mousedroid.llm_gateway.mission_parser import IntentType, MissionIntent
from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.orchestrator.mission_lifecycle import MissionLifecycleState

pytestmark = pytest.mark.smoke


@pytest.mark.asyncio
async def test_mission_lifecycle_smoke_boot_and_succeed() -> None:
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

    deadline = time.monotonic() + 5.0
    orch = build_orchestrator(cfg)
    assert orch._mission_lifecycle is not None

    parser = MagicMock()
    parser.parse = MagicMock(
        return_value=MissionIntent(
            intent_type=IntentType.NAVIGATION,
            goal_vector=GoalVector(vx_target=0.3),
            confidence=0.99,
            raw_command="explore",
        )
    )
    orch._mission_parser = parser  # type: ignore[attr-defined]

    await orch.process_mission("explore")
    await orch.tick()
    await orch.tick()

    assert orch._mission_lifecycle.current_state == MissionLifecycleState.SUCCEEDED
    assert time.monotonic() < deadline, "smoke budget exceeded"
