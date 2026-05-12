"""Tests for the rover sim-to-real config models.

These tests pin the public surface of :class:`RoverConfig` and assert
the backwards-compatibility contract: existing YAML files without a
``rover:`` block must still load.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mousedroid.config.schema import (
    RoverActionConfig,
    RoverConfig,
    RoverInertialConfig,
    RoverObservationConfig,
    RoverSimConfig,
    RoverTaskConfig,
    Settings,
)


def test_rover_config_defaults():
    cfg = RoverConfig()
    assert cfg.sim.backend == "mock"
    assert cfg.sim.urdf_path == "assets/rover/mse6_4wd.urdf"
    assert cfg.sim.decimation == 4
    assert cfg.sim.num_envs == 1
    assert cfg.sim.headless is True

    assert cfg.action.mode == "differential"
    assert cfg.action.max_wheel_rad_s == 25.0
    assert cfg.action.slew_rad_s2 == 60.0

    assert cfg.observation.include_imu is True
    assert cfg.observation.include_wheel_encoders is True
    assert cfg.observation.include_chassis_pose is True
    assert cfg.observation.include_lidar_sectors is True
    assert cfg.observation.lidar_num_sectors == 16


def test_rover_inertial_defaults():
    inertial = RoverInertialConfig()
    assert inertial.shell_mass_kg == 0.85
    assert inertial.shell_thickness_m == 0.003
    assert 0.0 < inertial.shell_infill <= 1.0
    assert inertial.com_offset_xyz_m == (0.0, 0.0, 0.04)
    assert inertial.com_offset_xyz_m[2] > 0.0  # top-heavy is intentional
    assert inertial.wheel_mass_kg == 0.06


def test_rover_sim_config_dt_implies_30hz_control():
    """1/120 s physics step * decimation 4 must give 30 Hz control."""
    sim = RoverSimConfig()
    control_dt = sim.sim_dt_s * sim.decimation
    assert abs((1.0 / control_dt) - 30.0) < 1e-6


def test_rover_action_modes():
    diff = RoverActionConfig(mode="differential")
    body = RoverActionConfig(mode="body_velocity")
    assert diff.mode == "differential"
    assert body.mode == "body_velocity"


def test_rover_action_dim_derived_per_mode():
    """action_dim must come from RoverActionConfig, not a magic constant."""
    assert RoverActionConfig(mode="differential").action_dim == 2
    assert RoverActionConfig(mode="body_velocity").action_dim == 2


def test_rover_task_defaults():
    task = RoverTaskConfig()
    assert task.goal_xy_m == (2.0, 0.0)
    assert task.goal_reach_radius_m == 0.10


def test_rover_task_custom_goal_overrides_defaults():
    task = RoverTaskConfig(goal_xy_m=(5.0, -1.5), goal_reach_radius_m=0.25)
    assert task.goal_xy_m == (5.0, -1.5)
    assert task.goal_reach_radius_m == 0.25


def test_rover_observation_enabled_keys_default():
    """Default toggles enable all four observation modalities in order."""
    obs = RoverObservationConfig()
    assert obs.enabled_keys() == ("imu", "chassis_pose", "wheel_vel", "lidar")


def test_rover_observation_enabled_keys_drops_disabled():
    """Disabled toggles must drop their key while preserving the order."""
    obs = RoverObservationConfig(
        include_imu=True,
        include_chassis_pose=True,
        include_wheel_encoders=False,
        include_lidar_sectors=True,
        lidar_num_sectors=8,
    )
    assert obs.enabled_keys() == ("imu", "chassis_pose", "lidar")


def test_settings_loads_without_rover_block():
    """Backwards-compat invariant #9: existing YAML must still load."""
    cfg = Settings(rover=None)
    assert cfg.rover is None


def test_settings_loads_with_rover_block():
    cfg = Settings(rover=RoverConfig())
    assert cfg.rover is not None
    assert cfg.rover.sim.backend == "mock"


def test_default_yaml_has_rover_block():
    """``config/default.yaml`` should ship a rover block for discovery."""
    repo_root = Path(__file__).resolve().parents[3]
    yaml_path = repo_root / "config" / "default.yaml"
    data = yaml.safe_load(yaml_path.read_text())
    assert "rover" in data
    rover = data["rover"]
    assert rover["sim"]["backend"] == "mock"
    assert rover["action"]["mode"] == "differential"
    assert rover["task"]["goal_xy_m"] == [2.0, 0.0]
    assert rover["task"]["goal_reach_radius_m"] == 0.10
