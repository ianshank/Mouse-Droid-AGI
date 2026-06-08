"""End-to-end: MuJoCo env -> episodes -> RSSM pretrain -> checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("mujoco")

from mousedroid.config.schema import RoverConfig, RoverSimConfig, Settings
from mousedroid.factory import build_rover_env, build_rssm_trainable
from mousedroid.training.rover_obs_adapter import RoverObsAdapter
from mousedroid.training.rssm_pretrainer import RSSMPretrainer
from mousedroid.training.sim_episode_generator import SimEpisodeGenerator


def test_end_to_end_pretrain_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")),
    )
    env = build_rover_env(cfg)
    model = build_rssm_trainable(cfg)
    adapter = RoverObsAdapter(battery_v=cfg.rover.sim.mujoco.battery_voltage_const_v)
    batch = SimEpisodeGenerator(env, adapter, n_episodes=4, seq_len=6, seed=0).generate()
    trainer = RSSMPretrainer(model, lr=1e-3, grad_clip=100.0, amp=False, device=torch.device("cpu"))
    history = trainer.train([batch], epochs=40, checkpoint_path=tmp_path / "rssm.pt")
    env.close()
    assert history[-1] < history[0]
    assert (tmp_path / "rssm.pt").exists()
