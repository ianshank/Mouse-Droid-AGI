"""SimEpisodeGenerator rolls deterministic episodes into batched RSSM tensors."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mousedroid.config.schema import RoverConfig, RoverSimConfig, Settings
from mousedroid.factory import build_rover_env
from mousedroid.training.rover_obs_adapter import RoverObsAdapter
from mousedroid.training.sim_episode_generator import SimEpisodeGenerator


def _gen(n: int, t: int) -> SimEpisodeGenerator:
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")),
    )
    env = build_rover_env(cfg)
    adapter = RoverObsAdapter(battery_v=cfg.rover.sim.mujoco.battery_voltage_const_v)
    return SimEpisodeGenerator(env, adapter, n_episodes=n, seq_len=t, seed=0)


def test_batch_tensor_shapes() -> None:
    batch = _gen(n=2, t=5).generate()
    assert batch.motor.shape == (2, 5, 4)
    assert batch.action.shape == (2, 5, 3)
    assert batch.valid_mask.shape == (2, 5, 5)
    assert batch.lidar.shape == (2, 5, 16)
    assert batch.reward.shape == (2, 5)


def test_deterministic_for_fixed_seed() -> None:
    b1 = _gen(2, 5).generate()
    b2 = _gen(2, 5).generate()
    assert np.allclose(b1.action.numpy(), b2.action.numpy())
