"""F-045 backwards-compat: DR off remains a no-op."""

from __future__ import annotations

from mousedroid.config.schema import DomainRandomizationConfig, RoverConfig, RoverRewardConfig
from mousedroid.sim.isaaclab.rover_env import RoverIsaacLabEnv


def test_dr_disabled_apply_is_noop() -> None:
    env = RoverIsaacLabEnv(
        RoverConfig(reward=RoverRewardConfig()),
        wheel_radius_m=0.042,
        track_width_m=0.20,
        domain_randomization=DomainRandomizationConfig(enabled=False),
    )
    env.apply_domain_params(friction=1.0, slip=0.0, mass_kg=2.7, motor_gain=1.0)
    assert env._pending_domain is None
