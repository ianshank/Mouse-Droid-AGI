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


def test_observation_keys_match_mock():
    cfg = RoverConfig(sim=RoverSimConfig(backend="isaac_lab"))
    mock_env = MockRoverEnv(cfg, wheel_radius_m=0.042, track_width_m=0.20)
    env = _make_env()
    assert env.observation_keys == mock_env.observation_keys


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
    env = _make_env(with_reward=True)
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


def test_step_collision_flag_missing_net_forces_is_false(monkeypatch):
    """A contact handle without ``net_forces_w`` reports no collision."""
    from types import SimpleNamespace

    from mousedroid.sim.isaaclab.constants import ROVER_CONTACT_SENSOR_NAME

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env(with_reward=True)
    env._built = True
    env._sensors = {ROVER_CONTACT_SENSOR_NAME: SimpleNamespace(data=SimpleNamespace())}
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["is_colliding"] is False


def test_reset_samples_domain_when_dr_on_without_pending(monkeypatch):
    """DR-on reset without a pending sample still fills chassis params."""
    from mousedroid.config.schema import DomainRandomizationConfig

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    cfg = RoverConfig(
        sim=RoverSimConfig(backend="isaac_lab"),
        reward=RoverRewardConfig(),
    )
    env = RoverIsaacLabEnv(
        cfg,
        wheel_radius_m=0.042,
        track_width_m=0.20,
        domain_randomization=DomainRandomizationConfig(enabled=True),
    )
    env._built = True
    _, info = env.reset(seed=0)
    assert env._pending_domain is None
    assert info["dr_enabled"] is True


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


def test_step_info_includes_body_velocity_keys(monkeypatch):
    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env(with_reward=True)
    env._built = True
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert "vx_body_mps" in info
    assert "omega_rads" in info
    assert "forward_velocity_mps" in info
    assert info["vx_body_mps"] == pytest.approx(info["forward_velocity_mps"])


def test_to_body_action_differential(monkeypatch):
    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env(with_reward=True)
    body = env.to_body_action(np.array([2.0, 2.0], dtype=np.float32))
    assert body.shape == (3,)
    assert body[1] == pytest.approx(0.0)
    assert body[2] == pytest.approx(0.0)


def test_to_body_action_body_velocity_mode():
    from mousedroid.config.schema import RoverActionConfig

    cfg = RoverConfig(
        sim=RoverSimConfig(backend="isaac_lab"),
        action=RoverActionConfig(mode="body_velocity"),
        reward=RoverRewardConfig(),
    )
    env = RoverIsaacLabEnv(cfg, wheel_radius_m=0.042, track_width_m=0.20)
    body = env.to_body_action(np.array([0.4, -0.2], dtype=np.float32))
    assert body[0] == pytest.approx(0.4)
    assert body[1] == pytest.approx(0.0)
    assert body[2] == pytest.approx(-0.2)


def test_apply_domain_params_noop_when_dr_off(monkeypatch):
    from mousedroid.config.schema import DomainRandomizationConfig

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    cfg = RoverConfig(
        sim=RoverSimConfig(backend="isaac_lab"),
        reward=RoverRewardConfig(),
    )
    env = RoverIsaacLabEnv(
        cfg,
        wheel_radius_m=0.042,
        track_width_m=0.20,
        domain_randomization=DomainRandomizationConfig(enabled=False),
    )
    env._built = True
    env.apply_domain_params(friction=0.9, slip=0.1, mass_kg=2.5, motor_gain=1.1)
    assert env._pending_domain is None


def test_apply_domain_params_stores_when_dr_on(monkeypatch):
    from mousedroid.config.schema import DomainRandomizationConfig

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    cfg = RoverConfig(
        sim=RoverSimConfig(backend="isaac_lab"),
        reward=RoverRewardConfig(),
    )
    env = RoverIsaacLabEnv(
        cfg,
        wheel_radius_m=0.042,
        track_width_m=0.20,
        domain_randomization=DomainRandomizationConfig(enabled=True),
    )
    env._built = True
    env.apply_domain_params(friction=0.8, slip=0.02, mass_kg=2.6, motor_gain=0.9)
    assert env._pending_domain is not None
    assert env._pending_domain["friction"] == pytest.approx(0.8)


def test_pending_sample_is_consumed_on_reset_without_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mousedroid.config.schema import DomainRandomizationConfig
    from mousedroid.training.domain_randomization import DomainRandomizer

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    cfg = RoverConfig(
        sim=RoverSimConfig(backend="isaac_lab"),
        reward=RoverRewardConfig(),
    )
    env = RoverIsaacLabEnv(
        cfg,
        wheel_radius_m=0.042,
        track_width_m=0.20,
        domain_randomization=DomainRandomizationConfig(enabled=True),
    )
    env._built = True
    env.apply_domain_params(friction=0.8, slip=0.02, mass_kg=2.6, motor_gain=0.9)
    count = {"n": 0}
    original = DomainRandomizer.sample

    def _count(self: DomainRandomizer, rng: np.random.Generator) -> object:
        count["n"] += 1
        return original(self, rng)

    monkeypatch.setattr(DomainRandomizer, "sample", _count)
    env.reset(seed=0)
    assert env._pending_domain is None
    assert count["n"] == 0


def test_read_imu_from_fake_sensor_on_step(monkeypatch):
    from types import SimpleNamespace

    from mousedroid.sim.isaaclab.constants import ROVER_SENSOR_LINK_NAMES

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env(with_reward=True)
    env._built = True
    env.inject_sensor(
        ROVER_SENSOR_LINK_NAMES[0],
        SimpleNamespace(
            data=SimpleNamespace(
                lin_acc_b=np.array([[0.0, 0.0, 9.8]], dtype=np.float32),
                ang_vel_b=np.zeros((1, 3), dtype=np.float32),
            )
        ),
    )
    env.reset(seed=0)
    obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert obs["imu"].shape == (6,)
    assert obs["imu"][2] == pytest.approx(9.8)


def test_read_lidar_from_fake_sensor_on_step(monkeypatch):
    from types import SimpleNamespace

    from mousedroid.sim.isaaclab.constants import ROVER_SENSOR_LINK_NAMES

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env(with_reward=True)
    env._built = True
    env.inject_sensor(
        ROVER_SENSOR_LINK_NAMES[1],
        SimpleNamespace(data=SimpleNamespace(ray_distance=np.ones(8, dtype=np.float32) * 2.0)),
    )
    env.reset(seed=0)
    obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert obs["lidar"].shape == (env._cfg.observation.lidar_num_sectors,)
    assert float(obs["lidar"].max()) <= 1.0 + 1e-6


def test_inject_sensor_rejects_empty_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env(with_reward=True)
    with pytest.raises(ValueError, match="non-empty"):
        env.inject_sensor("", object())


def test_reset_clears_pending_so_next_reset_resamples(monkeypatch: pytest.MonkeyPatch) -> None:
    from mousedroid.config.schema import DomainRandomizationConfig
    from mousedroid.training.domain_randomization import DomainRandomizer

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    cfg = RoverConfig(
        sim=RoverSimConfig(backend="isaac_lab"),
        reward=RoverRewardConfig(),
    )
    env = RoverIsaacLabEnv(
        cfg,
        wheel_radius_m=0.042,
        track_width_m=0.20,
        domain_randomization=DomainRandomizationConfig(enabled=True),
    )
    env._built = True
    count = {"n": 0}
    original = DomainRandomizer.sample

    def _count(self: DomainRandomizer, rng: np.random.Generator) -> object:
        count["n"] += 1
        return original(self, rng)

    monkeypatch.setattr(DomainRandomizer, "sample", _count)
    env.reset(seed=0)
    assert env._pending_domain is None
    assert count["n"] == 1
    env.reset(seed=1)
    assert env._pending_domain is None
    assert count["n"] == 2


def test_step_truncates_when_episode_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    cfg = RoverConfig(
        sim=RoverSimConfig(
            backend="isaac_lab",
            sim_dt_s=0.1,
            decimation=1,
            episode_length_s=0.2,
        ),
        reward=RoverRewardConfig(),
    )
    env = RoverIsaacLabEnv(cfg, wheel_radius_m=0.042, track_width_m=0.20)
    env._built = True
    env.reset(seed=0)
    zeros = np.zeros(2, dtype=np.float32)
    _, _, terminated, truncated, _ = env.step(zeros)
    assert terminated is False
    assert truncated is False
    _, _, terminated, truncated, _ = env.step(zeros)
    assert terminated is False
    assert truncated is True


def test_step_terminates_when_pose_inside_goal(monkeypatch: pytest.MonkeyPatch) -> None:
    from mousedroid.config.schema import RoverTaskConfig

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    cfg = RoverConfig(
        sim=RoverSimConfig(backend="isaac_lab"),
        reward=RoverRewardConfig(),
        task=RoverTaskConfig(goal_xy_m=(0.0, 0.0), goal_reach_radius_m=0.5),
    )
    env = RoverIsaacLabEnv(cfg, wheel_radius_m=0.042, track_width_m=0.20)
    env._built = True
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is True
    assert truncated is False


def test_step_uses_measured_root_velocity_not_commanded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env(with_reward=True)
    env._built = True
    env._articulation = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_b=np.array([[0.4, 0.0, 0.0]], dtype=np.float32),
            root_ang_vel_b=np.array([[0.0, 0.0, -0.2]], dtype=np.float32),
        )
    )
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["vx_body_mps"] == pytest.approx(0.4)
    assert info["omega_rads"] == pytest.approx(-0.2)
    assert info["forward_velocity_mps"] == pytest.approx(0.4)
    assert reward == pytest.approx(RoverRewardConfig().forward_velocity_weight * 0.4)


def test_step_zero_measured_vx_does_not_fall_back_to_wheels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(rover_env_module, "_isaaclab_available", lambda: True)
    env = _make_env(with_reward=True)
    env._built = True
    env._articulation = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_b=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            root_ang_vel_b=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
        )
    )
    env.reset(seed=0)
    action = np.array([1.0, 1.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    assert info["vx_body_mps"] == pytest.approx(0.0)
    assert info["omega_rads"] == pytest.approx(0.5)
