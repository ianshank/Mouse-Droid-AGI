"""F-043 backwards-compat: omitting Isaac keys still loads; YAML unchanged."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mousedroid.config.schema import (
    RoverConfig,
    RoverIsaacSimConfig,
    RoverObservationConfig,
    RoverSimConfig,
    Settings,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_rover_isaac_sim_config_instance_defaults() -> None:
    cfg = RoverIsaacSimConfig()
    assert cfg.device_headless == "cuda:0"
    assert cfg.device_gui == "cpu"
    assert cfg.usd_path is None
    assert cfg.actuator_stiffness == 0.0
    assert cfg.contact_history_length == 0
    assert cfg.track_air_time is False


def test_omitting_isaac_block_still_loads() -> None:
    cfg = RoverSimConfig.model_validate({"backend": "mock"})
    assert cfg.isaac.device_headless == "cuda:0"
    assert cfg.battery_voltage_const_v == 12.0
    assert cfg.mujoco.battery_voltage_const_v == 12.0


def test_settings_without_isaac_key_loads() -> None:
    cfg = Settings(mock_hardware=True, rover=RoverConfig())
    assert cfg.rover is not None
    assert cfg.rover.sim.isaac.device_gui == "cpu"
    assert cfg.harness is None


def test_observation_lidar_max_range_default() -> None:
    assert RoverObservationConfig().lidar_max_range_m == 4.0


@pytest.mark.parametrize(
    "overlay",
    sorted(p.name for p in (_REPO_ROOT / "config").glob("*.yaml")),
)
def test_shipped_overlays_still_load_without_isaac_keys(overlay: str) -> None:
    path = _REPO_ROOT / "config" / overlay
    text = path.read_text(encoding="utf-8")
    if "# config-validator: skip" in text:
        pytest.skip(f"{overlay} carries skip marker (not a Settings overlay)")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        return
    cfg = Settings.model_validate({**raw, "mock_hardware": True})
    if cfg.rover is None:
        return
    assert cfg.rover.sim.isaac.device_headless == "cuda:0"
    rover = raw.get("rover")
    if isinstance(rover, dict) and isinstance(rover.get("sim"), dict):
        assert "isaac" not in rover["sim"]
