"""_train_rssm is inert by default and runs the pretrainer when opted in."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mousedroid.config.schema import (
    RoverConfig,
    RoverRewardConfig,
    RoverSimConfig,
    Settings,
    TrainingConfig,
    TrainingPipelineConfig,
)
from mousedroid.training.pipeline_orchestrator import (
    _ISAAC_REWARD_BLOCK_REASON,
    PipelineOrchestrator,
    _maybe_build_rover_env,
    _rssm_pretrain_skip_reason,
)


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


def test_maybe_build_rover_env_invokes_build_when_present() -> None:
    seen: list[str] = []

    class _Lazy:
        def build(self) -> None:
            seen.append("built")

    _maybe_build_rover_env(_Lazy())
    assert seen == ["built"]
    _maybe_build_rover_env(object())
    assert seen == ["built"]


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


def test_rssm_pretrain_skip_reason_for_isaac_without_reward() -> None:
    assert _rssm_pretrain_skip_reason(None) == "unsupported_rssm_backend"
    assert (
        _rssm_pretrain_skip_reason(RoverConfig(sim=RoverSimConfig(backend="isaac_lab")))
        == _ISAAC_REWARD_BLOCK_REASON
    )
    assert (
        _rssm_pretrain_skip_reason(
            RoverConfig(
                sim=RoverSimConfig(backend="isaac_lab"),
                reward=RoverRewardConfig(),
            )
        )
        is None
    )
    assert _rssm_pretrain_skip_reason(RoverConfig(sim=RoverSimConfig(backend="mujoco"))) is None


@pytest.mark.asyncio
async def test_train_rssm_skipped_when_isaac_lab_reward_missing(tmp_path: Path) -> None:
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="isaac_lab")),
        training=TrainingConfig(rssm_pretrain_enabled=True, weights_dir=str(tmp_path)),
    )
    orch = _orch(cfg, tmp_path)
    await orch._train_rssm(batch_size=4)
    assert not (tmp_path / cfg.training.rssm_checkpoint_name).exists()


@pytest.mark.asyncio
async def test_train_rssm_runs_when_enabled_and_isaac_lab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: dict[str, float] = {}

    async def _fake_run(self: PipelineOrchestrator, **kwargs: object) -> None:
        env = kwargs["env"]
        called["battery_v"] = float(kwargs["battery_v"])  # type: ignore[arg-type]
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
    await orch._train_rssm(batch_size=4)
    assert cfg.rover is not None
    assert called["battery_v"] == pytest.approx(cfg.rover.sim.battery_voltage_const_v)


@pytest.mark.asyncio
async def test_run_rssm_training_builds_env_inside_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    class _Lazy:
        action_dim = 2

        def build(self) -> None:
            seen.append("built")

        def close(self) -> None:
            seen.append("closed")

    class _Gen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def generate(self) -> object:
            seen.append("generate")
            return object()

    class _Trainer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def train(self, *args: object, **kwargs: object) -> list[float]:
            seen.append("train")
            return [1.0, 0.5]

    monkeypatch.setattr(
        "mousedroid.training.sim_episode_generator.SimEpisodeGenerator",
        _Gen,
    )
    monkeypatch.setattr(
        "mousedroid.training.rssm_pretrainer.RSSMPretrainer",
        _Trainer,
    )
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(
            sim=RoverSimConfig(backend="isaac_lab"),
            reward=RoverRewardConfig(),
        ),
        training=TrainingConfig(rssm_pretrain_enabled=True, weights_dir=str(tmp_path)),
    )
    orch = _orch(cfg, tmp_path)
    assert cfg.rover is not None
    await orch._run_rssm_training(
        model=MagicMock(),
        env=_Lazy(),
        battery_v=cfg.rover.sim.battery_voltage_const_v,
        checkpoint=tmp_path / "rssm.pt",
        epochs=1,
        event_prefix="rssm_training",
    )
    assert seen[0] == "built"
    assert "generate" in seen
    assert "train" in seen
    assert "closed" in seen


@pytest.mark.asyncio
async def test_vision_finetune_skipped_for_isaac_lab(tmp_path: Path) -> None:
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="isaac_lab")),
        training=TrainingConfig(
            rssm_vision_finetune_enabled=True,
            rssm_finetune_checkpoint=str(tmp_path / "missing.pt"),
            weights_dir=str(tmp_path),
        ),
    )
    orch = _orch(cfg, tmp_path)
    await orch._train_rssm(batch_size=2)
    assert not (tmp_path / cfg.training.rssm_vision_checkpoint_name).exists()


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
