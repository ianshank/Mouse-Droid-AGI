"""C2.1 follow-through: process_mission() transitions the lifecycle PENDING -> RUNNING."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema import MissionConfig, Settings
from mousedroid.llm_gateway.mission_parser import IntentType, MissionIntent
from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.orchestrator.mission_lifecycle import (
    MissionLifecycle,
    MissionLifecycleState,
)

# Reuse the wiring fixture from test_mission_lifecycle_wiring.py
from tests.unit.orchestrator.test_mission_lifecycle_wiring import (
    _build_orch_with_lifecycle,
)


def _install_accepting_parser(orch: object, *, command: str) -> None:
    """Install a stub parser that accepts ``command`` with high confidence.

    Item #5 (Copilot MED): ``start_mission`` is now only called from
    inside an accepting Stage 1 / Stage 2 branch. Tests that exercise
    ``process_mission``'s lifecycle wiring need a parser that returns a
    confident, non-UNKNOWN intent so the orchestrator reaches the
    ``_start_mission_lifecycle_if_wired`` call site.
    """
    parser = MagicMock()
    parser.parse = MagicMock(
        return_value=MissionIntent(
            intent_type=IntentType.NAVIGATION,
            goal_vector=GoalVector(vx_target=0.5, vy_target=0.0, omega_target=0.0),
            confidence=0.99,
            raw_command=command,
        )
    )
    orch._mission_parser = parser  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_process_mission_starts_lifecycle() -> None:
    """process_mission must call start_mission() with the NL goal text."""
    cfg = Settings(mock_hardware=True)
    cfg.mission = MissionConfig(replan_enabled=True)
    lifecycle = MagicMock(spec=MissionLifecycle)
    lifecycle.start_mission = MagicMock()
    lifecycle.current_state = MissionLifecycleState.PENDING
    orch = _build_orch_with_lifecycle(cfg, lifecycle)
    _install_accepting_parser(orch, command="go to the kitchen")

    await orch.process_mission("go to the kitchen")

    lifecycle.start_mission.assert_called_once()
    args, kwargs = lifecycle.start_mission.call_args
    # Mission id should be a non-empty string, goal text should be the NL command.
    mission_id = kwargs.get("mission_id") or (args[0] if args else None)
    goal_text = kwargs.get("goal_text") or (args[1] if len(args) > 1 else None)
    assert mission_id, "start_mission requires a non-empty mission_id"
    assert goal_text == "go to the kitchen"


@pytest.mark.asyncio
async def test_process_mission_noop_when_no_lifecycle() -> None:
    """process_mission must not crash when no lifecycle is wired."""
    cfg = Settings(mock_hardware=True)
    orch = _build_orch_with_lifecycle(cfg, lifecycle=None)
    # Must not raise — no lifecycle wired.
    await orch.process_mission("explore")
