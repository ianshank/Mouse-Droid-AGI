"""End-to-end: vision-OFF checkpoint -> migrate -> render+extract -> fine-tune."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("mujoco")

from mousedroid.config.schema import MujocoSimConfig, RoverConfig, RoverSimConfig, Settings
from mousedroid.factory import (
    build_rover_env,
    build_rssm_trainable,
    build_rssm_vision_finetune,
    build_vision_feature_extractor,
)
from mousedroid.training.rover_obs_adapter import RoverObsAdapter
from mousedroid.training.rssm_pretrainer import RSSMPretrainer
from mousedroid.training.sim_episode_generator import SimEpisodeGenerator


def _render_cfg() -> Settings:
    return Settings(
        mock_hardware=True,
        rover=RoverConfig(
            sim=RoverSimConfig(backend="mujoco", mujoco=MujocoSimConfig(render_vision=True))
        ),
    )


def _gl_available(cfg: Settings) -> bool:
    env = build_rover_env(cfg)
    try:
        env.reset(seed=0)
        env.render_rgb()
    except Exception:
        return False
    else:
        return True
    finally:
        env.close()


def test_vision_finetune_round_trip(tmp_path: Path) -> None:
    cfg = _render_cfg()
    if not _gl_available(cfg):
        pytest.skip("offscreen GL rendering unavailable (headless CI)")
    torch.manual_seed(0)

    # 1) a vision-OFF pretrained checkpoint
    pretrained = build_rssm_trainable(cfg)
    ckpt = tmp_path / "pre.pt"
    torch.save(pretrained.state_dict(), ckpt)

    # 2) migrate to vision-ON + render-extract episodes + fine-tune
    model = build_rssm_vision_finetune(cfg, ckpt)
    assert model.encoder.vision_enabled is True
    env = build_rover_env(cfg)
    adapter = RoverObsAdapter(battery_v=cfg.rover.sim.mujoco.battery_voltage_const_v)
    batch = SimEpisodeGenerator(
        env,
        adapter,
        n_episodes=4,
        seq_len=6,
        seed=0,
        feature_extractor=build_vision_feature_extractor(cfg),
    ).generate()
    assert batch.vision.shape == (4, 6, cfg.camera.feature_dim)

    history = RSSMPretrainer(
        model, lr=1e-3, grad_clip=100.0, amp=False, device=torch.device("cpu")
    ).train([batch], epochs=40, checkpoint_path=tmp_path / "vis.pt")
    env.close()
    assert history[-1] < history[0]
    assert (tmp_path / "vis.pt").exists()
