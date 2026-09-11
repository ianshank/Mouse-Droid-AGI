"""Factory-built Isaac env + injected fakes (always-on; no isaaclab extra)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mousedroid.config.schema import RoverConfig, RoverRewardConfig, RoverSimConfig, Settings
from mousedroid.factory import build_rover_env
from mousedroid.sim.isaaclab import rover_env as rover_env_module
from mousedroid.sim.isaaclab.constants import ROVER_CONTACT_SENSOR_NAME, ROVER_SENSOR_LINK_NAMES
from mousedroid.sim.isaaclab.rover_env import RoverIsaacLabEnv


def test_factory_isaac_lab_steps_with_injected_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """``build_rover_env`` returns Isaac; fakes let reset/step run without Sim."""
    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(
            sim=RoverSimConfig(backend="isaac_lab"),
            reward=RoverRewardConfig(),
        ),
    )
    env = build_rover_env(cfg)
    assert isinstance(env, RoverIsaacLabEnv)
    env._built = True
    env._sensors[ROVER_SENSOR_LINK_NAMES[0]] = SimpleNamespace(
        data=SimpleNamespace(
            lin_acc_b=np.zeros((1, 3), dtype=np.float32),
            ang_vel_b=np.zeros((1, 3), dtype=np.float32),
        )
    )
    env._sensors[ROVER_CONTACT_SENSOR_NAME] = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=np.zeros((1, 1, 3), dtype=np.float32))
    )
    try:
        obs, info = env.reset(seed=0)
        assert "chassis_pose" in obs
        assert info["step_idx"] == 0
        obs, _reward, _term, _trunc, info = env.step(np.zeros(2, dtype=np.float32))
        assert "vx_body_mps" in info
        assert obs["imu"].shape[0] == 6
    finally:
        env.close()
