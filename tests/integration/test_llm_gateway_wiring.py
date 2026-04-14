"""Integration tests for LLM gateway wiring and mission parser fallback chain.

Phase 5: Validates factory wiring, orchestrator process_mission(),
rule-based → LLM fallback chain, and lifecycle management.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from mousedroid.config.schema import MissionParserConfig, Settings
from mousedroid.llm_gateway.mission_parser import (
    IntentType,
    MissionIntent,
    RuleBasedMissionParser,
)
from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle


def _make_observation(cfg: Settings) -> MouseDroidObservationBundle:
    """Create a minimal observation bundle."""
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32),
        _audio_chunk=np.zeros(1024, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def _make_orchestrator(
    *,
    llm_gateway: AsyncMock | None = None,
    mission_parser: MagicMock | None = None,
) -> MouseDroidOrchestrator:
    """Create orchestrator with optional LLM gateway and mission parser."""
    cfg = Settings(mock_hardware=True)

    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )

    agent = MagicMock()
    agent.name = "test_agent"
    agent.act.return_value = torch.tensor([0.1, 0.0, 0.0])

    safety_ctx = SafetyContext(is_emergency=False)
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = safety_ctx

    esp32 = AsyncMock()

    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _make_observation(cfg)
    sensor_manager.recovery_attempt.return_value = 0

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
        llm_gateway=llm_gateway,
        mission_parser=mission_parser,
    )


# ---------------------------------------------------------------------------
# Config backward compatibility
# ---------------------------------------------------------------------------


def test_mission_parser_config_llm_fallback_default() -> None:
    """MissionParserConfig has backward-compatible llm_fallback_confidence."""
    cfg = MissionParserConfig()
    assert cfg.llm_fallback_confidence == pytest.approx(0.5)


def test_mission_parser_config_custom_threshold() -> None:
    """Custom llm_fallback_confidence is accepted."""
    cfg = MissionParserConfig(llm_fallback_confidence=0.8)
    assert cfg.llm_fallback_confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Rule-based parser resolves common commands (no LLM needed)
# ---------------------------------------------------------------------------


async def test_process_mission_rule_based_forward() -> None:
    """'go forward' resolves via rule-based parser without LLM."""
    parser = MagicMock()
    parser.parse.return_value = MissionIntent(
        intent_type=IntentType.VELOCITY,
        goal_vector=GoalVector(vx_target=0.5, vy_target=0.0, omega_target=0.0),
        confidence=0.9,
        raw_command="go forward",
    )

    llm = AsyncMock()
    orch = _make_orchestrator(llm_gateway=llm, mission_parser=parser)

    goal = await orch.process_mission("go forward")

    assert goal.vx_target == pytest.approx(0.5)
    assert goal.vy_target == pytest.approx(0.0)
    # Rule-based succeeded — LLM should NOT have been called
    llm.translate_mission.assert_not_awaited()


async def test_process_mission_rule_based_stop() -> None:
    """'stop' resolves via rule-based parser with full confidence."""
    parser = MagicMock()
    parser.parse.return_value = MissionIntent(
        intent_type=IntentType.STOP,
        goal_vector=GoalVector(0.0, 0.0, 0.0),
        confidence=1.0,
        raw_command="stop",
    )

    orch = _make_orchestrator(mission_parser=parser)
    goal = await orch.process_mission("stop")

    assert goal == GoalVector(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Unknown commands fall through to LLM
# ---------------------------------------------------------------------------


async def test_process_mission_llm_fallback_for_unknown() -> None:
    """Unknown commands fall through to LLM gateway."""
    parser = MagicMock()
    parser.parse.return_value = MissionIntent(
        intent_type=IntentType.UNKNOWN,
        confidence=0.0,
        raw_command="navigate to the Death Star plans",
    )

    llm = AsyncMock()
    llm.translate_mission.return_value = GoalVector(vx_target=0.7, vy_target=0.0, omega_target=0.2)

    orch = _make_orchestrator(llm_gateway=llm, mission_parser=parser)
    goal = await orch.process_mission("navigate to the Death Star plans")

    assert goal.vx_target == pytest.approx(0.7)
    assert goal.omega_target == pytest.approx(0.2)
    llm.translate_mission.assert_awaited_once_with("navigate to the Death Star plans")


async def test_process_mission_low_confidence_falls_to_llm() -> None:
    """Low-confidence parser results fall through to LLM."""
    parser = MagicMock()
    parser.parse.return_value = MissionIntent(
        intent_type=IntentType.VELOCITY,
        goal_vector=GoalVector(vx_target=0.3, vy_target=0.0, omega_target=0.0),
        confidence=0.2,  # Below default threshold of 0.5
        raw_command="kinda go that way",
    )

    llm = AsyncMock()
    llm.translate_mission.return_value = GoalVector(vx_target=0.5, vy_target=0.1, omega_target=0.0)

    orch = _make_orchestrator(llm_gateway=llm, mission_parser=parser)
    goal = await orch.process_mission("kinda go that way")

    # LLM result should be used since parser confidence < threshold
    assert goal.vx_target == pytest.approx(0.5)
    llm.translate_mission.assert_awaited_once()


# ---------------------------------------------------------------------------
# LLM failure → safe zero fallback
# ---------------------------------------------------------------------------


async def test_process_mission_llm_failure_returns_zero() -> None:
    """LLM failure returns safe zero GoalVector."""
    parser = MagicMock()
    parser.parse.return_value = MissionIntent(
        intent_type=IntentType.UNKNOWN,
        confidence=0.0,
        raw_command="ambiguous",
    )

    llm = AsyncMock()
    llm.translate_mission.side_effect = RuntimeError("model not loaded")

    orch = _make_orchestrator(llm_gateway=llm, mission_parser=parser)
    goal = await orch.process_mission("ambiguous")

    assert goal == GoalVector()  # Safe zero default


async def test_process_mission_no_llm_no_parser_returns_zero() -> None:
    """No parser and no LLM returns safe zero GoalVector."""
    orch = _make_orchestrator()
    goal = await orch.process_mission("anything")

    assert goal == GoalVector()


async def test_process_mission_empty_command_returns_zero() -> None:
    """Empty command returns safe zero GoalVector."""
    orch = _make_orchestrator()
    goal = await orch.process_mission("")

    assert goal == GoalVector()


async def test_process_mission_whitespace_command_returns_zero() -> None:
    """Whitespace-only command returns safe zero GoalVector."""
    orch = _make_orchestrator()
    goal = await orch.process_mission("   ")

    assert goal == GoalVector()


# ---------------------------------------------------------------------------
# No LLM but parser works → parser result used
# ---------------------------------------------------------------------------


async def test_process_mission_no_llm_parser_only() -> None:
    """Parser-only mode (no LLM) returns parser result for known commands."""
    parser = MagicMock()
    parser.parse.return_value = MissionIntent(
        intent_type=IntentType.VELOCITY,
        goal_vector=GoalVector(vx_target=0.5, vy_target=0.0, omega_target=0.0),
        confidence=0.9,
        raw_command="go forward",
    )

    orch = _make_orchestrator(mission_parser=parser)
    goal = await orch.process_mission("go forward")

    assert goal.vx_target == pytest.approx(0.5)


async def test_process_mission_no_llm_unknown_returns_zero() -> None:
    """Parser-only mode returns zero for unknown commands (no LLM fallback)."""
    parser = MagicMock()
    parser.parse.return_value = MissionIntent(
        intent_type=IntentType.UNKNOWN,
        confidence=0.0,
        raw_command="do a barrel roll",
    )

    orch = _make_orchestrator(mission_parser=parser)
    goal = await orch.process_mission("do a barrel roll")

    assert goal == GoalVector()


# ---------------------------------------------------------------------------
# Lifecycle — LLM gateway started/stopped with orchestrator
# ---------------------------------------------------------------------------


async def test_llm_gateway_started_on_start() -> None:
    """LLM gateway is started when orchestrator starts."""
    llm = AsyncMock()
    orch = _make_orchestrator(llm_gateway=llm)

    await orch.start()
    llm.start.assert_awaited_once()

    await orch.stop()


async def test_llm_gateway_stopped_on_stop() -> None:
    """LLM gateway is stopped when orchestrator stops."""
    llm = AsyncMock()
    orch = _make_orchestrator(llm_gateway=llm)

    await orch.start()
    await orch.stop()

    llm.stop.assert_awaited_once()


async def test_no_llm_gateway_no_crash() -> None:
    """Orchestrator without LLM gateway starts/stops without crash."""
    orch = _make_orchestrator()
    await orch.start()
    assert orch._running is True
    await orch.stop()


# ---------------------------------------------------------------------------
# Factory wiring integration
# ---------------------------------------------------------------------------


def test_factory_build_mission_parser() -> None:
    """build_mission_parser returns a MissionParserProtocol."""
    from mousedroid.factory import build_mission_parser
    from mousedroid.llm_gateway.mission_parser import MissionParserProtocol

    cfg = Settings(mock_hardware=True)
    parser = build_mission_parser(cfg)

    assert isinstance(parser, MissionParserProtocol)


def test_factory_build_llm_gateway() -> None:
    """build_llm_gateway returns an LLMGatewayProtocol."""
    from mousedroid.factory import build_llm_gateway

    cfg = Settings(mock_hardware=True)
    gateway = build_llm_gateway(cfg)

    assert hasattr(gateway, "translate_mission")
    assert hasattr(gateway, "start")
    assert hasattr(gateway, "stop")


# ---------------------------------------------------------------------------
# End-to-end: real parser + mock LLM
# ---------------------------------------------------------------------------


async def test_e2e_common_commands_resolve_via_parser() -> None:
    """10 common commands resolve via rule-based parser (no LLM call)."""
    parser = RuleBasedMissionParser()
    llm = AsyncMock()
    orch = _make_orchestrator(llm_gateway=llm, mission_parser=parser)

    commands = [
        "go forward",
        "go backward",
        "turn left",
        "turn right",
        "stop",
        "strafe left",
        "strafe right",
        "patrol the bridge",
        "move forward fast",
        "drive ahead slowly",
    ]

    for cmd in commands:
        goal = await orch.process_mission(cmd)
        # All commands should have resolved (at least one non-zero or it's stop)
        assert isinstance(goal, GoalVector), f"Failed for: {cmd}"

    # LLM should NOT have been called for any of these
    llm.translate_mission.assert_not_awaited()


async def test_e2e_unknown_command_triggers_llm() -> None:
    """Unknown command triggers LLM after parser returns UNKNOWN."""
    parser = RuleBasedMissionParser()
    llm = AsyncMock()
    llm.translate_mission.return_value = GoalVector(vx_target=0.3)

    orch = _make_orchestrator(llm_gateway=llm, mission_parser=parser)

    goal = await orch.process_mission("execute order 66")

    llm.translate_mission.assert_awaited_once()
    assert goal.vx_target == pytest.approx(0.3)
