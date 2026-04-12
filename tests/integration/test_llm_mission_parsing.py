"""Integration tests for LLM mission parsing pipeline.

Tests the full NL command -> structured intent -> goal vector flow
using diverse natural language inputs.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from mousedroid.llm_gateway.mission_parser import (
    IntentType,
    RuleBasedMissionParser,
)
from mousedroid.llm_gateway.protocol import GoalVector


@pytest.fixture
def parser() -> RuleBasedMissionParser:
    """Create a rule-based mission parser."""
    return RuleBasedMissionParser()


class TestDiverseNLCommands:
    """Integration tests for 10 diverse NL commands -> parsed intents."""

    def test_go_forward(self, parser: RuleBasedMissionParser) -> None:
        """'go forward' produces valid forward velocity."""
        result = parser.parse("go forward")
        assert result.intent_type == IntentType.VELOCITY
        assert result.goal_vector.vx_target > 0
        assert result.goal_vector.vy_target == 0.0
        assert result.goal_vector.omega_target == 0.0
        assert result.confidence > 0.5

    def test_turn_left_90_degrees(self, parser: RuleBasedMissionParser) -> None:
        """'turn left 90 degrees' produces rotation with angle param."""
        result = parser.parse("turn left 90 degrees")
        assert result.intent_type == IntentType.VELOCITY
        assert result.goal_vector.omega_target > 0
        assert result.goal_vector.vx_target == 0.0
        assert result.parameters.get("angle_degrees") == 90.0
        assert result.confidence > 0.5

    def test_stop(self, parser: RuleBasedMissionParser) -> None:
        """'stop' produces zero velocity with full confidence."""
        result = parser.parse("stop")
        assert result.intent_type == IntentType.STOP
        assert result.goal_vector == GoalVector(0.0, 0.0, 0.0)
        assert result.confidence == 1.0

    def test_patrol_the_hallway(self, parser: RuleBasedMissionParser) -> None:
        """'patrol the hallway' produces patrol intent."""
        result = parser.parse("patrol the hallway")
        assert result.intent_type == IntentType.PATROL
        assert result.goal_vector.vx_target > 0
        assert result.parameters.get("location") == "the hallway"
        assert result.confidence > 0.5

    def test_avoid_obstacles(self, parser: RuleBasedMissionParser) -> None:
        """'avoid obstacles' produces navigation intent."""
        result = parser.parse("avoid obstacles")
        assert result.intent_type == IntentType.NAVIGATION
        assert result.parameters.get("mode") == "obstacle_avoidance"
        assert result.goal_vector.vx_target > 0
        assert result.confidence > 0.5

    def test_move_backward(self, parser: RuleBasedMissionParser) -> None:
        """'move backward' produces negative vx."""
        result = parser.parse("move backward")
        assert result.intent_type == IntentType.VELOCITY
        assert result.goal_vector.vx_target < 0
        assert result.goal_vector.omega_target == 0.0

    def test_turn_right(self, parser: RuleBasedMissionParser) -> None:
        """'turn right' produces negative omega (clockwise)."""
        result = parser.parse("turn right")
        assert result.intent_type == IntentType.VELOCITY
        assert result.goal_vector.omega_target < 0

    def test_drive_forward_fast(self, parser: RuleBasedMissionParser) -> None:
        """'drive forward fast' applies speed modifier."""
        result = parser.parse("drive forward fast")
        assert result.intent_type == IntentType.VELOCITY
        assert result.goal_vector.vx_target >= 0.8

    def test_rotate_left(self, parser: RuleBasedMissionParser) -> None:
        """'rotate left' is an alternative turn command."""
        result = parser.parse("rotate left")
        assert result.intent_type == IntentType.VELOCITY
        assert result.goal_vector.omega_target > 0

    def test_emergency_stop(self, parser: RuleBasedMissionParser) -> None:
        """'emergency stop' is treated as stop intent."""
        result = parser.parse("emergency stop")
        assert result.intent_type == IntentType.STOP
        assert result.goal_vector == GoalVector(0.0, 0.0, 0.0)
        assert result.confidence == 1.0


class TestGoalVectorValidity:
    """Verify all parsed goal vectors have valid ranges."""

    COMMANDS: ClassVar[list[str]] = [
        "go forward",
        "turn left 90 degrees",
        "stop",
        "patrol the hallway",
        "avoid obstacles",
        "move backward",
        "turn right",
        "drive forward fast",
        "rotate left",
        "emergency stop",
    ]

    @pytest.mark.parametrize("command", COMMANDS)
    def test_goal_vector_in_range(self, parser: RuleBasedMissionParser, command: str) -> None:
        """All goal vector components are in [-1, 1]."""
        result = parser.parse(command)
        gv = result.goal_vector
        assert -1.0 <= gv.vx_target <= 1.0
        assert -1.0 <= gv.vy_target <= 1.0
        assert -1.0 <= gv.omega_target <= 1.0

    @pytest.mark.parametrize("command", COMMANDS)
    def test_confidence_in_range(self, parser: RuleBasedMissionParser, command: str) -> None:
        """Confidence is in [0, 1]."""
        result = parser.parse(command)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.parametrize("command", COMMANDS)
    def test_raw_command_preserved(self, parser: RuleBasedMissionParser, command: str) -> None:
        """Original command text is preserved."""
        result = parser.parse(command)
        assert result.raw_command == command


class TestGatewayEndToEnd:
    """Integration test for the full gateway pipeline."""

    @pytest.fixture
    def gateway(self) -> object:
        """Create gateway from config."""
        from mousedroid.llm_gateway.config import GatewayConfig
        from mousedroid.llm_gateway.gateway import LLMGateway

        cfg = GatewayConfig(model_path="/tmp/fake.gguf", enabled=False)
        return LLMGateway(cfg)

    async def test_disabled_gateway_start_stop(self, gateway: object) -> None:
        """Disabled gateway can start and stop cleanly."""
        from mousedroid.llm_gateway.gateway import LLMGateway

        assert isinstance(gateway, LLMGateway)
        await gateway.start()
        await gateway.stop()

    async def test_config_driven_gateway_from_settings(self) -> None:
        """Build gateway from root Settings using factory."""
        from mousedroid.config.schema import Settings
        from mousedroid.factory import build_llm_gateway
        from mousedroid.llm_gateway.protocol import LLMGatewayProtocol

        cfg = Settings(mock_hardware=True)
        gw = build_llm_gateway(cfg)
        assert isinstance(gw, LLMGatewayProtocol)
