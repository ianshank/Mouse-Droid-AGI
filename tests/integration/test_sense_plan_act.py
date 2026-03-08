from __future__ import annotations

import torch

from mousedroid.config.schema import Settings
from mousedroid.factory import build_orchestrator


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


async def test_full_tick_cycle() -> None:
    cfg = _mock_settings()
    orch = build_orchestrator(cfg)
    orch._world_model.imagine_step = _patch_imagine_step(cfg)
    await orch.start()
    await orch.tick()
    await orch.stop()


async def test_orchestrator_start_stop() -> None:
    cfg = _mock_settings()
    orch = build_orchestrator(cfg)
    await orch.start()
    assert orch._running is True
    await orch.stop()
    assert orch._running is False


async def test_health_check_returns_ok() -> None:
    cfg = _mock_settings()
    orch = build_orchestrator(cfg)
    await orch.start()
    health = await orch.health_check()
    assert health["status"] == "ok"
    assert health["mock_hardware"] is True
    await orch.stop()


async def test_multiple_ticks() -> None:
    cfg = _mock_settings()
    orch = build_orchestrator(cfg)
    orch._world_model.imagine_step = _patch_imagine_step(cfg)
    await orch.start()
    for _ in range(3):
        await orch.tick()
    await orch.stop()


async def test_stop_without_start() -> None:
    cfg = _mock_settings()
    orch = build_orchestrator(cfg)
    await orch.stop()
