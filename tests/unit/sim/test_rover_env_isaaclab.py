"""Tests for the Isaac Lab rover env stub.

These tests run unconditionally on the import-safe surface (action/obs
shape contract, error handling when Isaac Lab is absent) and skip the
``build`` smoke test when ``isaaclab`` is not importable. CI does not
install Isaac Lab; running these tests with the ``[isaac]`` extra on a
workstation exercises the full path.
"""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.config.schema import RoverConfig, RoverSimConfig, Settings
from mousedroid.factory import build_rover_env
from mousedroid.sim.isaaclab import rover_env as rover_env_module
from mousedroid.sim.isaaclab.rover_env import (
    IsaacLabUnavailableError,
    RoverEnvNotBuiltError,
    RoverIsaacLabEnv,
    _isaaclab_available,
)
from mousedroid.sim.protocols import RoverEnvProtocol


def _make_env():
    return RoverIsaacLabEnv(
        RoverConfig(sim=RoverSimConfig(backend="isaac_lab")),
        wheel_radius_m=0.042,
        track_width_m=0.20,
    )


def test_stub_constructs_without_isaaclab():
    env = _make_env()
    assert env.action_dim == 2
    assert "chassis_pose" in env.observation_keys


def test_stub_implements_protocol():
    env = _make_env()
    assert isinstance(env, RoverEnvProtocol)


def test_reset_raises_when_isaaclab_missing():
    if _isaaclab_available():
        pytest.skip("Isaac Lab installed; this path only triggers without it.")
    env = _make_env()
    with pytest.raises(IsaacLabUnavailableError):
        env.reset(seed=0)


def test_reset_raises_not_built_when_isaaclab_present_but_unbuilt(monkeypatch):
    """If isaaclab is installed but build() was never called, raise NotBuilt."""
    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env()
    with pytest.raises(RoverEnvNotBuiltError):
        env.reset(seed=0)


def test_step_raises_not_built_when_isaaclab_present_but_unbuilt(monkeypatch):
    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env()
    with pytest.raises(RoverEnvNotBuiltError):
        env.step(np.zeros(2, dtype=np.float32))


def test_step_validates_action_shape_before_isaaclab():
    """Shape validation runs before the Isaac Lab availability check."""
    env = _make_env()
    with pytest.raises(ValueError, match="action shape must be"):
        env.step(np.zeros(7, dtype=np.float32))


def test_build_via_factory_returns_isaaclab_env():
    cfg = Settings(rover=RoverConfig(sim=RoverSimConfig(backend="isaac_lab")))
    env = build_rover_env(cfg)
    assert isinstance(env, RoverIsaacLabEnv)


@pytest.mark.skipif(not _isaaclab_available(), reason="Isaac Lab not installed")
def test_build_succeeds_when_isaaclab_installed():
    env = _make_env()
    env.build()  # should not raise
    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict)
    assert info["step_idx"] == 0
    env.close()


def test_step_idx_increments_under_build_bypass(monkeypatch):
    """step_idx must increment monotonically to match MockRoverEnv parity."""
    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env()
    env._built = True  # bypass the real Isaac Lab build for the stub
    _obs, info = env.reset(seed=0)
    assert info["step_idx"] == 0
    for expected in (1, 2, 3):
        _o, _r, _t, _tr, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["step_idx"] == expected


def test_close_is_idempotent_without_build():
    env = _make_env()
    env.close()
    env.close()


def test_close_resets_built_flag(monkeypatch):
    """After close(), the env must require a fresh build() before reuse."""
    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env()
    env._built = True
    env.close()
    with pytest.raises(RoverEnvNotBuiltError):
        env.reset(seed=0)
