"""Factory resolves backend='mujoco' to RoverMuJoCoEnv."""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from mousedroid.config.schema import RoverConfig, RoverSimConfig, Settings
from mousedroid.factory import build_rover_env
from mousedroid.sim.mujoco_rover_env import RoverMuJoCoEnv
from mousedroid.sim.protocols import RoverEnvProtocol


def test_mujoco_backend_builds_env() -> None:
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")),
    )
    env = build_rover_env(cfg)
    assert isinstance(env, RoverMuJoCoEnv)
    assert isinstance(env, RoverEnvProtocol)
