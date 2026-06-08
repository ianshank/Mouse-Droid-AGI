"""Golden RSSM pretrain loss — tolerance-based decrease + ceiling (non-gating).

Deliberately NOT a point-wise ±1% curve: MuJoCo physics + GRU/CUDA nondeterminism
make exact reproduction across platforms/versions infeasible. We assert a robust
monotone-ish decrease + a sane absolute ceiling instead, CPU-deterministic.
"""

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


def test_loss_decreases_deterministically(tmp_path: Path) -> None:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")),
    )
    env = build_rover_env(cfg)
    model = build_rssm_trainable(cfg)
    adapter = RoverObsAdapter(battery_v=cfg.rover.sim.mujoco.battery_voltage_const_v)
    batch = SimEpisodeGenerator(env, adapter, n_episodes=4, seq_len=8, seed=0).generate()
    history = RSSMPretrainer(
        model, lr=1e-3, grad_clip=100.0, amp=False, device=torch.device("cpu")
    ).train([batch], epochs=40, checkpoint_path=tmp_path / "golden_rssm.pt")
    env.close()
    # Tolerance-based, NOT point-wise ±1% (cross-platform float / MuJoCo drift).
    assert history[-1] < history[0] * 0.95  # at least a 5% reduction
    assert history[-1] < 10.0  # sane absolute ceiling
