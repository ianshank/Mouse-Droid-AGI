"""RSSMPretrainer runs an Adam loop and writes a checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from mousedroid.config.schema import ModelConfig
from mousedroid.training.rssm_pretrainer import RSSMPretrainer
from mousedroid.training.sim_episode_generator import EpisodeBatch
from mousedroid.world_model.rssm import RSSM


def _model() -> RSSM:
    return RSSM(ModelConfig(vision_dim=0, vision_proj_dim=0, lidar_dim=16, lidar_proj_dim=32))


def _batch(b: int = 3, t: int = 5) -> EpisodeBatch:
    return EpisodeBatch(
        motor=torch.randn(b, t, 4),
        ultrasonic=torch.rand(b, t, 1),
        lidar=torch.rand(b, t, 16),
        valid_mask=torch.ones(b, t, 5),
        action=torch.randn(b, t, 3),
        reward=torch.randn(b, t),
    )


def test_train_reduces_loss_and_writes_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _model()
    trainer = RSSMPretrainer(model, lr=1e-3, grad_clip=100.0, amp=False, device=torch.device("cpu"))
    history = trainer.train([_batch()], epochs=15, checkpoint_path=tmp_path / "rssm.pt")
    assert history[-1] < history[0]
    assert (tmp_path / "rssm.pt").exists()


def test_checkpoint_is_loadable(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _model()
    trainer = RSSMPretrainer(model, lr=1e-3, grad_clip=100.0, amp=False, device=torch.device("cpu"))
    trainer.train([_batch()], epochs=2, checkpoint_path=tmp_path / "rssm.pt")
    # weights_only=True: the checkpoint is a pure state_dict (tensors) — never
    # unpickle arbitrary objects from a model file.
    state = torch.load(tmp_path / "rssm.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # round-trips


def test_empty_batches_returns_empty_history(tmp_path: Path) -> None:
    model = _model()
    trainer = RSSMPretrainer(model, lr=1e-3, grad_clip=100.0, amp=False, device=torch.device("cpu"))
    assert trainer.train([], epochs=5, checkpoint_path=tmp_path / "none.pt") == []
    assert not (tmp_path / "none.pt").exists()  # no checkpoint written for empty input
