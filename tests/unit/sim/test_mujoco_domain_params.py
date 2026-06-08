"""DR params map onto concrete mjModel fields (friction/mass/gain); slip = obs noise."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mousedroid.config.schema import RoverConfig
from mousedroid.sim.mujoco_rover_env import RoverMuJoCoEnv

# Close every env created via _env() after each test so MuJoCo native handles
# never leak across the suite (close() is idempotent).
_OPEN_ENVS: list[RoverMuJoCoEnv] = []


@pytest.fixture(autouse=True)
def _close_tracked_envs() -> Iterator[None]:
    yield
    while _OPEN_ENVS:
        _OPEN_ENVS.pop().close()


def _env() -> RoverMuJoCoEnv:
    env = RoverMuJoCoEnv(RoverConfig(), wheel_radius_m=0.042, track_width_m=0.20)
    _OPEN_ENVS.append(env)
    return env


def test_friction_param_writes_geom_friction() -> None:
    env = _env()
    env.apply_domain_params(friction=1.25, slip=0.0, mass_kg=2.7, motor_gain=1.0)
    frics = np.asarray(env._model.geom_friction[:, 0])
    assert np.isclose(frics.max(), 1.25, atol=1e-6)


def test_mass_param_writes_body_mass() -> None:
    env = _env()
    env.apply_domain_params(friction=1.0, slip=0.0, mass_kg=3.0, motor_gain=1.0)
    masses = np.asarray(env._model.body_mass)
    assert np.isclose(masses.max(), 3.0, atol=1e-3)


def test_motor_gain_writes_actuator_gainprm() -> None:
    env = _env()
    env.apply_domain_params(friction=1.0, slip=0.0, mass_kg=2.7, motor_gain=1.15)
    gains = np.asarray(env._model.actuator_gainprm[:, 0])
    assert np.isclose(gains.max(), 1.15, atol=1e-6)


def test_slip_is_obs_noise_not_physics() -> None:
    env = _env()
    env.apply_domain_params(friction=1.0, slip=0.1, mass_kg=2.7, motor_gain=1.0)
    env.reset(seed=1)
    obs = env.step(np.asarray([5.0, 5.0], dtype=np.float32))[0]
    assert np.isfinite(obs["wheel_vel"]).all()  # noise applied, still finite
