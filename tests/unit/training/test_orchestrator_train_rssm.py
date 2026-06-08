"""_train_rssm is inert by default and runs the pretrainer when opted in."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mousedroid.config.schema import (
    RoverConfig,
    RoverSimConfig,
    Settings,
    TrainingConfig,
    TrainingPipelineConfig,
)
from mousedroid.training.pipeline_orchestrator import PipelineOrchestrator


def _orch(settings: Settings, checkpoint_dir: Path) -> PipelineOrchestrator:
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
async def test_train_rssm_inert_when_disabled(tmp_path: Path) -> None:
    cfg = Settings(mock_hardware=True)  # rssm_pretrain_enabled defaults False
    orch = _orch(cfg, tmp_path)
    await orch._train_rssm(batch_size=4)
    assert not (tmp_path / "rssm_pretrained.pt").exists()


@pytest.mark.asyncio
async def test_train_rssm_skipped_for_non_mujoco_backend(tmp_path: Path) -> None:
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mock")),
        training=TrainingConfig(rssm_pretrain_enabled=True, weights_dir=str(tmp_path)),
    )
    orch = _orch(cfg, tmp_path)
    await orch._train_rssm(batch_size=4)
    assert not (tmp_path / "rssm_pretrained.pt").exists()


@pytest.mark.asyncio
async def test_train_rssm_runs_when_enabled_and_mujoco(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")),
        training=TrainingConfig(
            rssm_pretrain_enabled=True,
            n_episodes=2,
            sequence_length=4,
            epochs=2,
            weights_dir=str(tmp_path),
        ),
    )
    orch = _orch(cfg, tmp_path)
    await orch._train_rssm(batch_size=2)
    assert (tmp_path / cfg.training.rssm_checkpoint_name).exists()
