"""MujocoSimConfig is additive and defaults are sane."""

from __future__ import annotations

import yaml

from mousedroid.config.schema import MujocoSimConfig, RoverSimConfig, Settings


def test_mujoco_config_defaults() -> None:
    m = MujocoSimConfig()
    assert m.mjcf_path.endswith("mse6_4wd.xml")
    assert m.arena_half_extent_m > 0
    assert m.lidar_num_sectors == 16
    assert m.wheel_slip_default == 0.0  # obs-noise proxy off by default


def test_rover_sim_auto_populates_mujoco_subconfig() -> None:
    # A RoverSimConfig built without a mujoco block still resolves the defaults.
    assert RoverSimConfig().mujoco.lidar_num_sectors == 16


def test_opt_in_overrides_parse() -> None:
    raw = yaml.safe_load(
        """
        mock_hardware: true
        rover:
          sim:
            backend: mujoco
            mujoco:
              lidar_num_sectors: 12
              wheel_friction_default: 1.1
        """
    )
    cfg = Settings.model_validate(raw)
    assert cfg.rover.sim.backend == "mujoco"
    assert cfg.rover.sim.mujoco.lidar_num_sectors == 12
    assert cfg.rover.sim.mujoco.wheel_friction_default == 1.1


def test_pre_feature_yaml_without_mujoco_block_loads() -> None:
    raw = yaml.safe_load(
        """
        mock_hardware: true
        rover:
          sim:
            backend: mock
        """
    )
    cfg = Settings.model_validate(raw)
    assert cfg.rover.sim.backend == "mock"
    assert cfg.rover.sim.mujoco.wheel_slip_default == 0.0  # default proxy off
