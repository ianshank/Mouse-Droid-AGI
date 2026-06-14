"""SimEpisodeGenerator rolls deterministic episodes into batched RSSM tensors."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mousedroid.config.schema import MujocoSimConfig, RoverConfig, RoverSimConfig, Settings
from mousedroid.factory import build_rover_env
from mousedroid.sim.protocols import RoverEnvProtocol
from mousedroid.training.rover_obs_adapter import RoverObsAdapter
from mousedroid.training.sim_episode_generator import SimEpisodeGenerator

# Close every built env after each test so MuJoCo native handles never leak
# across the suite (close() is idempotent).
_OPEN_ENVS: list[RoverEnvProtocol] = []


@pytest.fixture(autouse=True)
def _close_tracked_envs() -> Iterator[None]:
    yield
    while _OPEN_ENVS:
        _OPEN_ENVS.pop().close()


def _track(env: RoverEnvProtocol) -> RoverEnvProtocol:
    """Register an env for guaranteed teardown after the current test."""
    _OPEN_ENVS.append(env)
    return env


def _gen(n: int, t: int) -> SimEpisodeGenerator:
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")),
    )
    env = _track(build_rover_env(cfg))
    adapter = RoverObsAdapter(battery_v=cfg.rover.sim.mujoco.battery_voltage_const_v)
    return SimEpisodeGenerator(env, adapter, n_episodes=n, seq_len=t, seed=0)


def test_batch_tensor_shapes() -> None:
    batch = _gen(n=2, t=5).generate()
    assert batch.motor.shape == (2, 5, 4)
    assert batch.action.shape == (2, 5, 3)
    assert batch.valid_mask.shape == (2, 5, 5)
    assert batch.lidar.shape == (2, 5, 16)
    assert batch.reward.shape == (2, 5)
    assert batch.vision.shape == (2, 5, 0)  # no extractor -> empty vision


def test_deterministic_for_fixed_seed() -> None:
    b1 = _gen(2, 5).generate()
    b2 = _gen(2, 5).generate()
    assert np.allclose(b1.action.numpy(), b2.action.numpy())


def _mj_env_and_adapter() -> tuple[object, RoverObsAdapter]:
    cfg = Settings(mock_hardware=True, rover=RoverConfig(sim=RoverSimConfig(backend="mujoco")))
    env = _track(build_rover_env(cfg))
    adapter = RoverObsAdapter(battery_v=cfg.rover.sim.mujoco.battery_voltage_const_v)
    return env, adapter


def test_domain_randomizer_applied_per_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    from mousedroid.config.schema import DomainRandomizationConfig
    from mousedroid.training.domain_randomization import DomainRandomizer

    env, adapter = _mj_env_and_adapter()
    dr = DomainRandomizer(DomainRandomizationConfig(enabled=True))
    calls: list[dict[str, float]] = []
    original = env.apply_domain_params  # type: ignore[attr-defined]

    def _spy(**kwargs: float) -> None:
        calls.append(kwargs)
        original(**kwargs)

    monkeypatch.setattr(env, "apply_domain_params", _spy)
    SimEpisodeGenerator(
        env, adapter, n_episodes=3, seq_len=3, seed=0, domain_randomizer=dr
    ).generate()
    assert len(calls) == 3  # one DR sample per episode
    assert all(0.7 <= c["friction"] <= 1.3 for c in calls)  # within configured range


def test_vision_features_populated_with_extractor() -> None:
    """With a feature extractor + render-capable env, EpisodeBatch.vision is filled."""
    from mousedroid.factory import build_vision_feature_extractor

    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(
            sim=RoverSimConfig(backend="mujoco", mujoco=MujocoSimConfig(render_vision=True))
        ),
    )
    env = _track(build_rover_env(cfg))
    # Skip if offscreen GL rendering is unavailable (headless CI).
    try:
        env.reset(seed=0)
        env.render_rgb()
    except Exception:
        pytest.skip("offscreen GL rendering unavailable")
    adapter = RoverObsAdapter(battery_v=cfg.rover.sim.mujoco.battery_voltage_const_v)
    extractor = build_vision_feature_extractor(cfg)
    batch = SimEpisodeGenerator(
        env, adapter, n_episodes=2, seq_len=3, seed=0, feature_extractor=extractor
    ).generate()
    assert batch.vision.shape == (2, 3, cfg.camera.feature_dim)


def test_domain_randomizer_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    from mousedroid.config.schema import DomainRandomizationConfig
    from mousedroid.training.domain_randomization import DomainRandomizer

    env, adapter = _mj_env_and_adapter()
    dr = DomainRandomizer(DomainRandomizationConfig(enabled=False))
    calls: list[dict[str, float]] = []
    monkeypatch.setattr(env, "apply_domain_params", lambda **kw: calls.append(kw))
    SimEpisodeGenerator(
        env, adapter, n_episodes=2, seq_len=3, seed=0, domain_randomizer=dr
    ).generate()
    assert calls == []  # disabled DR never touches the env
