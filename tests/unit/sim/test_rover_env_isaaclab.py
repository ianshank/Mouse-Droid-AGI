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

from mousedroid.config.schema import (
    RoverConfig,
    RoverRewardConfig,
    RoverSimConfig,
    Settings,
)
from mousedroid.factory import build_rover_env
from mousedroid.sim.isaaclab import rover_env as rover_env_module
from mousedroid.sim.isaaclab.rover_env import (
    IsaacLabUnavailableError,
    RoverEnvNotBuiltError,
    RoverIsaacLabEnv,
    _isaaclab_available,
)
from mousedroid.sim.mock_rover_env import MockRoverEnv
from mousedroid.sim.protocols import RoverEnvProtocol


def _make_env(*, with_reward: bool = False):
    """Construct an Isaac Lab env stub.

    Args:
        with_reward: If True, attach the C4 ``RoverRewardConfig`` block
            so paths that bypass ``build()`` (the legacy stub tests)
            still reach ``step()`` without tripping the new reward
            presence check.
    """
    cfg = RoverConfig(
        sim=RoverSimConfig(backend="isaac_lab"),
        reward=RoverRewardConfig() if with_reward else None,
    )
    return RoverIsaacLabEnv(
        cfg,
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
    # Bypass build() but provide the reward block — step() requires it
    # to compose the forward-velocity / collision reward (C4 wiring).
    env = _make_env(with_reward=True)
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


def test_build_delegates_to_wire_isaaclab_scene_when_available(monkeypatch):
    """``build()`` reaches the Isaac-Lab-only wiring helper when the dep is mocked in.

    Pins the Tier C4 fixup refactor: the original single-method body
    was unreachable under CI coverage; the split into
    ``_wire_isaaclab_scene`` keeps the public guard (``_isaaclab_available``
    + reward presence) on the covered path while the live scene wiring
    sits under ``# pragma: no cover``.
    """
    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env(with_reward=True)
    called: list[bool] = []

    def _fake_wire(self) -> None:
        called.append(True)
        self._built = True

    monkeypatch.setattr(RoverIsaacLabEnv, "_wire_isaaclab_scene", _fake_wire)
    env.build()
    assert called == [True]
    assert env._built is True


def test_build_raises_value_error_when_reward_block_missing(monkeypatch):
    """``build()`` rejects ``cfg.rover.reward is None`` even with isaaclab present."""
    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env(with_reward=False)
    with pytest.raises(ValueError, match=r"cfg\.rover\.reward"):
        env.build()


def test_step_body_velocity_mode_fans_via_kinematic_mixer(monkeypatch):
    """body_velocity mode mixes vx + omega across the 4 wheels.

    Pins the second branch of ``_fan_out_action`` (vx clipped to
    ``cap * wheel_radius``; omega passed through unclipped). This
    runs unconditionally — the parametrized live-Isaac-Lab variant in
    ``tests/unit/sim/isaaclab/test_rover_env.py`` is skipped on CI.
    """
    from mousedroid.config.schema import RoverActionConfig

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    cfg = RoverConfig(
        sim=RoverSimConfig(backend="isaac_lab"),
        action=RoverActionConfig(mode="body_velocity"),
        reward=RoverRewardConfig(),
    )
    env = RoverIsaacLabEnv(cfg, wheel_radius_m=0.042, track_width_m=0.20)
    env._built = True
    env.reset(seed=0)
    # Saturating vx; non-trivial omega.
    action = np.array([1e6, 1.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    wheels = info["wheel_velocities"]
    assert len(wheels) == 4
    # vx_body clipped to cap * wheel_radius; resulting forward_velocity
    # magnitude must respect that bound.
    cap = cfg.action.max_wheel_rad_s
    assert abs(info["forward_velocity_mps"]) <= cap * 0.042 + 1e-6
    # Asymmetric omega -> left and right wheels differ.
    assert wheels[0] != pytest.approx(wheels[1])


def test_step_reads_collision_flag_from_contact_sensor(monkeypatch):
    """When a live contact sensor reports non-zero forces, ``info['is_colliding']`` is True.

    Pins the fixup's :meth:`_read_collision_flag` data-path: read
    ``contact.data.net_forces_w`` and treat any non-zero magnitude as
    a collision. The test wires a tiny stub sensor under the
    :data:`ROVER_CONTACT_SENSOR_NAME` key so the path runs without a
    live Isaac Sim handle.
    """
    from types import SimpleNamespace

    from mousedroid.sim.isaaclab.constants import ROVER_CONTACT_SENSOR_NAME

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env(with_reward=True)
    env._built = True
    fake_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32))
    )
    env._sensors = {ROVER_CONTACT_SENSOR_NAME: fake_sensor}
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["is_colliding"] is True
    # collision_weight * 1 is subtracted from the forward-velocity term.
    assert reward == pytest.approx(-RoverRewardConfig().collision_weight)


def test_step_collision_flag_zero_forces_means_no_collision(monkeypatch):
    """Zero-magnitude contact forces report as ``is_colliding=False``."""
    from types import SimpleNamespace

    from mousedroid.sim.isaaclab.constants import ROVER_CONTACT_SENSOR_NAME

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env(with_reward=True)
    env._built = True
    fake_sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=np.zeros((1, 1, 3), dtype=np.float32))
    )
    env._sensors = {ROVER_CONTACT_SENSOR_NAME: fake_sensor}
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["is_colliding"] is False


def test_reset_short_circuits_sim_context_reset_when_unbound(monkeypatch):
    """``reset()`` skips the sim-context reset when ``_sim_context`` is ``None``.

    Covers the fixup's new ``if self._sim_context is not None and hasattr``
    guard branch — exercised here under the build-bypass path that
    leaves ``_sim_context = None`` so the guard short-circuits without
    touching a live Isaac Lab handle.
    """
    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env(with_reward=True)
    env._built = True
    assert env._sim_context is None
    _, info = env.reset(seed=0)
    assert info["step_idx"] == 0


def test_chassis_pose_matches_mock_reset_identity(monkeypatch):
    """Stub reset() must agree with MockRoverEnv reset() on chassis_pose.

    Both encode heading as ``[x, y, cos(theta), sin(theta)]``. At reset
    (theta=0) the only valid identity value is ``[0, 0, 1, 0]``; the
    earlier ``[0, 0, 0, 0]`` violated cos^2 + sin^2 = 1 and disagreed
    with the mock backend.
    """
    cfg = RoverConfig(sim=RoverSimConfig(backend="isaac_lab"))
    mock_env = MockRoverEnv(cfg, wheel_radius_m=0.042, track_width_m=0.20)
    mock_obs, _ = mock_env.reset(seed=0)

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    isaac_env = _make_env()
    isaac_env._built = True
    isaac_obs, _ = isaac_env.reset(seed=0)

    np.testing.assert_array_equal(isaac_obs["chassis_pose"], mock_obs["chassis_pose"])
    # And the encoding must actually be a unit heading vector.
    cos_t, sin_t = float(isaac_obs["chassis_pose"][2]), float(isaac_obs["chassis_pose"][3])
    assert cos_t**2 + sin_t**2 == pytest.approx(1.0)
