"""Sanity and smoke tests for the Autonomous Architecture."""

from __future__ import annotations

import pytest

from mousedroid.config.schema.root import Settings
from mousedroid.factory import build_autonomous_orchestrator
from mousedroid.interfaces.protocols import GoalVector


def test_goal_vector_neutral_stop_sanity() -> None:
    """Sanity test: GoalVector.neutral_stop() produces zero velocity and safe e_stop."""
    stop_vec = GoalVector.neutral_stop()
    assert stop_vec.linear_velocity == 0.0
    assert stop_vec.angular_velocity == 0.0
    assert stop_vec.arm_action == "e_stop"
    assert stop_vec.is_safe is True


@pytest.mark.asyncio
async def test_smoke_orchestrator_quick_cycle() -> None:
    """Smoke test: Quick orchestrator build and single-step cycle."""
    cfg = Settings(mock_hardware=True)
    orch = build_autonomous_orchestrator(cfg)

    ok = await orch.execute_mission_step("scout area")
    assert ok is True
    await orch.stop()
