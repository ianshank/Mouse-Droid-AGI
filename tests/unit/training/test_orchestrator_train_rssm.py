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
async def test_vision_finetune_skipped_without_checkpoint(tmp_path: Path) -> None:
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")),
        training=TrainingConfig(rssm_vision_finetune_enabled=True, weights_dir=str(tmp_path)),
    )
    orch = _orch(cfg, tmp_path)
    await orch._train_rssm(batch_size=2)
    assert not (tmp_path / cfg.training.rssm_vision_checkpoint_name).exists()


@pytest.mark.asyncio
async def test_vision_finetune_runs_when_enabled(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    import torch

    from mousedroid.config.schema import MujocoSimConfig
    from mousedroid.factory import build_rover_env, build_rssm_trainable

    # Skip if offscreen GL rendering is unavailable (headless CI).
    probe_cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(
            sim=RoverSimConfig(backend="mujoco", mujoco=MujocoSimConfig(render_vision=True))
        ),
    )
    probe = build_rover_env(probe_cfg)
    try:
        probe.reset(seed=0)
        probe.render_rgb()
    except Exception:
        pytest.skip("offscreen GL rendering unavailable")
    finally:
        probe.close()

    base_cfg = Settings(mock_hardware=True, rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")))
    pretrained = build_rssm_trainable(base_cfg)
    ckpt = tmp_path / "pre.pt"
    torch.save(pretrained.state_dict(), ckpt)

    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")),
        training=TrainingConfig(
            rssm_vision_finetune_enabled=True,
            rssm_finetune_checkpoint=str(ckpt),
            rssm_finetune_epochs=2,
            n_episodes=2,
            sequence_length=4,
            weights_dir=str(tmp_path),
        ),
    )
    orch = _orch(cfg, tmp_path)
    await orch._train_rssm(batch_size=2)
    assert (tmp_path / cfg.training.rssm_vision_checkpoint_name).exists()


@pytest.mark.asyncio
async def test_pretrain_and_finetune_both_run_when_both_enabled(tmp_path: Path) -> None:
    """Vision fine-tune must NOT short-circuit pretraining: both phases run + write."""
    pytest.importorskip("mujoco")
    from mousedroid.config.schema import MujocoSimConfig
    from mousedroid.factory import build_rover_env

    probe_cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(
            sim=RoverSimConfig(backend="mujoco", mujoco=MujocoSimConfig(render_vision=True))
        ),
    )
    probe = build_rover_env(probe_cfg)
    try:
        probe.reset(seed=0)
        probe.render_rgb()
    except Exception:
        pytest.skip("offscreen GL rendering unavailable")
    finally:
        probe.close()

    # Pretrain writes rssm_pretrained.pt FIRST; the fine-tune then consumes it.
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")),
        training=TrainingConfig(
            rssm_pretrain_enabled=True,
            rssm_vision_finetune_enabled=True,
            rssm_checkpoint_name="rssm_pretrained.pt",
            rssm_finetune_checkpoint=str(tmp_path / "rssm_pretrained.pt"),
            rssm_finetune_epochs=2,
            n_episodes=2,
            sequence_length=4,
            epochs=2,
            weights_dir=str(tmp_path),
        ),
    )
    orch = _orch(cfg, tmp_path)
    await orch._train_rssm(batch_size=2)
    assert (tmp_path / "rssm_pretrained.pt").exists()  # pretrain ran
    assert (tmp_path / cfg.training.rssm_vision_checkpoint_name).exists()  # fine-tune ran


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
