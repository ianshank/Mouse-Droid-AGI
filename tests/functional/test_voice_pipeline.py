"""Functional tests for voice feature."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mousedroid.llm_gateway.mission_parser import IntentType, MissionIntent
from mousedroid.llm_gateway.protocol import GoalVector


@pytest.mark.asyncio
async def test_mission_state_changes_trigger_tts(functional_orchestrator):
    """Test mission state changes trigger TTS announcements."""
    orch = functional_orchestrator

    # Mock voice engine
    mock_voice = AsyncMock()
    orch._voice_engine = mock_voice

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
        # Ensure voice was called for state change
        mock_voice.speak.assert_called()
    finally:
        await orch.stop()


@pytest.mark.asyncio
async def test_voice_system_degrades_gracefully(functional_orchestrator):
    """Test voice system degrades gracefully when TTS engine unavailable."""
    orch = functional_orchestrator

    # Simulate broken voice engine
    mock_voice = AsyncMock()
    mock_voice.speak.side_effect = Exception("TTS Offline")
    orch._voice_engine = mock_voice

    await orch.start()
    try:
        # Should not crash the orchestrator
        await orch.tick()
    finally:
        await orch.stop()
