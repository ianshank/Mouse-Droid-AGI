"""F-044 backwards-compat: mock still default; IMU slot stays mask-zero."""

from __future__ import annotations

from mousedroid.config.schema import RoverConfig, RoverSimConfig, Settings
from mousedroid.training.rover_obs_adapter import RoverObsAdapter


def test_default_backend_is_still_mock() -> None:
    cfg = Settings(mock_hardware=True, rover=RoverConfig())
    assert cfg.rover is not None
    assert cfg.rover.sim.backend == "mock"


def test_adapter_still_zeros_rssm_imu_slot() -> None:
    adapter = RoverObsAdapter(battery_v=12.0)
    out = adapter.adapt(
        {"lidar": __import__("numpy").zeros(16, dtype="float32")},
        {"vx_body_mps": 0.1, "omega_rads": 0.0},
    )
    from mousedroid.constants import SENSOR_SLOT_MAP

    assert out["valid_mask"][SENSOR_SLOT_MAP["imu"]] == 0.0
    assert "imu" not in out


def test_isaac_backend_still_constructs_without_yaml_keys() -> None:
    cfg = Settings(
        mock_hardware=True,
        rover=RoverConfig(sim=RoverSimConfig(backend="isaac_lab")),
    )
    assert cfg.rover is not None
    assert cfg.rover.observation.lidar_max_range_m == 4.0
