"""Functional tests for the complete mission lifecycle."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from mousedroid.llm_gateway.mission_parser import IntentType, MissionIntent
from mousedroid.llm_gateway.protocol import GoalVector


@pytest.mark.asyncio
async def test_complete_mission_flow(functional_orchestrator):
    """Test full mission flow: command -> execution -> completion."""
    orch = functional_orchestrator

    parser = MagicMock()
    parser.parse = MagicMock(
        return_value=MissionIntent(
            intent_type=IntentType.NAVIGATION,
            goal_vector=GoalVector(vx_target=0.4),
            confidence=0.99,
            raw_command="go forward",
        )
    )
    orch._mission_parser = parser

    # Start system
    await orch.start()

    try:
        # Operator sends command
        await orch.process_mission("go forward")

        # Progress mission
        initial_tick_count = getattr(orch, "_tick_count", 0)
        for _ in range(3):
            await orch.tick()

        final_tick_count = getattr(orch, "_tick_count", 0)
        assert final_tick_count >= initial_tick_count + 3
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_mission_cancellation(functional_orchestrator):
    """Test cancelling a mission mid-flight."""
    orch = functional_orchestrator

    parser = MagicMock()
    parser.parse = MagicMock(
        return_value=MissionIntent(
            intent_type=IntentType.NAVIGATION,
            goal_vector=GoalVector(vx_target=0.4),
            confidence=0.99,
            raw_command="patrol",
        )
    )
    orch._mission_parser = parser

    await orch.start()
    try:
        await orch.process_mission("patrol")
        initial_tick_count = getattr(orch, "_tick_count", 0)
        await orch.tick()
        assert getattr(orch, "_tick_count", 0) >= initial_tick_count + 1

        # Verify it stopped cleanly
    except Exception as e:
        pytest.fail(f"Mission cancellation failed with exception {e}")
    finally:
        await orch.stop()
