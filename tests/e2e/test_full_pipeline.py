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
    assert "mouse_droid_navigator" in health["agents"]  # default platform


async def test_start_tick_stop_no_errors() -> None:
    cfg = _mock_settings()
    orch = build_orchestrator(cfg)
    orch._world_model.imagine_step = _patch_imagine_step(cfg)
    await orch.start()
    try:
        await orch.tick()
    finally:
        await orch.stop()
