"""SimEpisodeGenerator + MockRoverEnv (always-on; no mujoco importorskip)."""

from __future__ import annotations

from mousedroid.config.schema import RoverConfig, RoverSimConfig, Settings
from mousedroid.factory import build_rover_env
from mousedroid.training.rover_obs_adapter import RoverObsAdapter
from mousedroid.training.sim_episode_generator import SimEpisodeGenerator


def test_mock_backend_generates_body_actions() -> None:
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mock")),
    )
    env = build_rover_env(cfg)
    assert cfg.rover is not None
    try:
        adapter = RoverObsAdapter(battery_v=cfg.rover.sim.battery_voltage_const_v)
        gen = SimEpisodeGenerator(env, adapter, n_episodes=2, seq_len=4, seed=0)
        batch = gen.generate()
        assert batch.action.shape[-1] == 3
        assert batch.motor.shape[0] == 2
    finally:
        env.close()
