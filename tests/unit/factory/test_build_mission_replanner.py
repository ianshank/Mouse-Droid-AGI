"""Tier C2.3: build_mission_replanner factory gate + LLMGateway threading."""

from __future__ import annotations

from unittest.mock import MagicMock

from mousedroid.config.schema import Settings
from mousedroid.factory import build_mission_replanner
from mousedroid.orchestrator.llm_replanner import LLMGatewayMissionReplanner


def test_returns_none_when_disabled() -> None:
    cfg = Settings(mock_hardware=True)
    assert cfg.mission.llm_replanner_enabled is False
    assert build_mission_replanner(cfg, llm_gateway=MagicMock()) is None


def test_returns_none_when_enabled_but_no_gateway() -> None:
    cfg = Settings(mock_hardware=True)
    cfg.mission.llm_replanner_enabled = True
    assert build_mission_replanner(cfg, llm_gateway=None) is None


def test_returns_adapter_when_enabled_with_gateway() -> None:
    cfg = Settings(mock_hardware=True)
    cfg.mission.llm_replanner_enabled = True
    gateway = MagicMock()
    replanner = build_mission_replanner(cfg, llm_gateway=gateway)
    assert isinstance(replanner, LLMGatewayMissionReplanner)
    assert replanner._gateway is gateway  # noqa: SLF001
