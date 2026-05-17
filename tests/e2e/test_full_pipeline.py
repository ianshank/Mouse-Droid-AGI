from __future__ import annotations

import pytest
import torch

from mousedroid.config.schema import Settings
from mousedroid.factory import build_orchestrator


@pytest.fixture(autouse=True)
def _set_mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")


def _mock_settings() -> Settings:
    return Settings(mock_hardware=True)


def _patch_imagine_step(cfg: Settings):
    def _fixed_imagine_step(action, h, z):
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return (
            torch.zeros(1, cfg.model.hidden_dim),
            torch.zeros(1, cfg.model.latent_dim),
            torch.tensor([[0.0]]),
        )

    return _fixed_imagine_step


async def test_full_lifecycle() -> None:
    cfg = _mock_settings()
    orch = build_orchestrator(cfg)
    orch._world_model.imagine_step = _patch_imagine_step(cfg)
    await orch.start()
    await orch.tick()
    await orch.stop()


async def test_health_check_ok() -> None:
    cfg = _mock_settings()
    orch = build_orchestrator(cfg)
    await orch.start()
    health = await orch.health_check()
    assert health["status"] == "ok"
    await orch.stop()


async def test_multiple_ticks_stable() -> None:
    cfg = _mock_settings()
    orch = build_orchestrator(cfg)
    orch._world_model.imagine_step = _patch_imagine_step(cfg)
    await orch.start()
    for _ in range(5):
        await orch.tick()
    await orch.stop()


async def test_orchestrator_agents_present() -> None:
    cfg = _mock_settings()
    orch = build_orchestrator(cfg)
    health = await orch.health_check()
    assert "mouse_droid_navigator" in health["agents"]


async def test_start_tick_stop_no_errors() -> None:
    cfg = _mock_settings()
    orch = build_orchestrator(cfg)
    orch._world_model.imagine_step = _patch_imagine_step(cfg)
    await orch.start()
    try:
        await orch.tick()
    finally:
        await orch.stop()


async def test_e2e_mission_lifecycle_full_succeed() -> None:
    """Tier C2.3 E2E: build_orchestrator → process_mission → tick loop → SUCCEEDED.

    Asserts the closed loop through the full DI graph (mock_hardware
    wiring): factory builds VLM head + LLM replanner + lifecycle, the
    rule-based parser accepts a NL command via a stub, and two ticks
    drive the lifecycle to SUCCEEDED via the default ``MockVLMProgress``
    backend (value 0.95 > success_threshold 0.5).
    """
    from unittest.mock import MagicMock

    from mousedroid.config.schema import MissionConfig
    from mousedroid.llm_gateway.mission_parser import IntentType, MissionIntent
    from mousedroid.llm_gateway.protocol import GoalVector
    from mousedroid.orchestrator.mission_lifecycle import MissionLifecycleState

    cfg = _mock_settings()
    cfg.mission = MissionConfig(
        replan_enabled=True,
        vlm_progress_enabled=True,
        llm_replanner_enabled=True,
        vlm_mock_progress_value=0.95,
        success_threshold=0.5,
    )
    cfg.llm.enabled = True
    orch = build_orchestrator(cfg)
    orch._world_model.imagine_step = _patch_imagine_step(cfg)

    parser = MagicMock()
    parser.parse = MagicMock(
        return_value=MissionIntent(
            intent_type=IntentType.NAVIGATION,
            goal_vector=GoalVector(vx_target=0.4),
            confidence=0.99,
            raw_command="proceed",
        )
    )
    orch._mission_parser = parser  # type: ignore[attr-defined]

    await orch.process_mission("proceed")
    for _ in range(3):
        await orch.tick()
    assert orch._mission_lifecycle is not None
    assert orch._mission_lifecycle.current_state == MissionLifecycleState.SUCCEEDED
