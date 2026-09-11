"""E2E: RSSM pretrain CLI-equivalent gate — mock skips; physics backends opt-in."""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.schema import (
    RoverConfig,
    RoverRewardConfig,
    RoverSimConfig,
    Settings,
    TrainingConfig,
    TrainingPipelineConfig,
)
from mousedroid.training.pipeline_orchestrator import PipelineOrchestrator


def _orch(settings: Settings, checkpoint_dir: Path) -> PipelineOrchestrator:
    from unittest.mock import AsyncMock, MagicMock

    gpu = AsyncMock()
    gpu.should_pause.return_value = False
    batch = MagicMock()
    batch.tune_batch_size.side_effect = lambda phase, base: base
    return PipelineOrchestrator(
        settings=settings,
        pipeline_config=TrainingPipelineConfig(checkpoint_dir=str(checkpoint_dir)),
        gpu_monitor=gpu,
        batch_tuner=batch,
    )


@pytest.mark.asyncio
async def test_e2e_mock_backend_still_skips_pretrain(tmp_path: Path) -> None:
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mock")),
        training=TrainingConfig(rssm_pretrain_enabled=True, weights_dir=str(tmp_path)),
    )
    orch = _orch(cfg, tmp_path)
    await orch._train_rssm(batch_size=2)
    assert not (tmp_path / cfg.training.rssm_checkpoint_name).exists()


@pytest.mark.asyncio
async def test_e2e_isaac_lab_gate_invokes_training_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = {"n": 0}

    async def _fake_run(self: PipelineOrchestrator, **kwargs: object) -> None:
        ran["n"] += 1
        env = kwargs["env"]
        close = getattr(env, "close", None)
        if callable(close):
            close()

    monkeypatch.setattr(PipelineOrchestrator, "_run_rssm_training", _fake_run)
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(
            sim=RoverSimConfig(backend="isaac_lab"),
            reward=RoverRewardConfig(),
        ),
        training=TrainingConfig(rssm_pretrain_enabled=True, weights_dir=str(tmp_path)),
    )
    orch = _orch(cfg, tmp_path)
    await orch._train_rssm(batch_size=2)
    assert ran["n"] == 1


@pytest.mark.asyncio
async def test_e2e_isaac_lab_skips_without_reward_block(tmp_path: Path) -> None:
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="isaac_lab")),
        training=TrainingConfig(rssm_pretrain_enabled=True, weights_dir=str(tmp_path)),
    )
    orch = _orch(cfg, tmp_path)
    await orch._train_rssm(batch_size=2)
    assert not (tmp_path / cfg.training.rssm_checkpoint_name).exists()


def test_e2e_physics_backends_include_mujoco() -> None:
    from mousedroid.sim.protocols import ROVER_RSSM_PHYSICS_BACKENDS

    assert "mujoco" in ROVER_RSSM_PHYSICS_BACKENDS
    assert "isaac_lab" in ROVER_RSSM_PHYSICS_BACKENDS
