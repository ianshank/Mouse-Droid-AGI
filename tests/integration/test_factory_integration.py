from __future__ import annotations

from mousedroid.agents.navigation import MouseDroidNavigationAgent
from mousedroid.comms.mock_driver import MockESP32Driver
from mousedroid.config.schema import Settings
from mousedroid.factory import (
    build_agent,
    build_camera,
    build_distance_sensor,
    build_esp32_driver,
    build_orchestrator,
    build_safety_monitor,
    build_world_model,
)
from mousedroid.hardware.camera.mock_camera import MockCamera
from mousedroid.hardware.sensors.mock_ultrasonic import MockUltrasonic
from mousedroid.safety.monitor import MouseDroidSafetyMonitor
from mousedroid.world_model.rssm import RSSM


def _mock_settings() -> Settings:
    return Settings(mock_hardware=True)


def test_build_orchestrator_creates_all_components() -> None:
    cfg = _mock_settings()
    orch = build_orchestrator(cfg)
    assert orch is not None


def test_build_esp32_driver_mock() -> None:
    cfg = _mock_settings()
    driver = build_esp32_driver(cfg)
    from mousedroid.resilience.resilient_driver import ResilientESP32Driver

    assert isinstance(driver, ResilientESP32Driver)
    assert isinstance(driver.inner, MockESP32Driver)


def test_build_camera_mock() -> None:
    cfg = _mock_settings()
    cam = build_camera(cfg)
    assert isinstance(cam, MockCamera)


def test_build_distance_sensor_mock() -> None:
    cfg = _mock_settings()
    sensor = build_distance_sensor(cfg)
    assert isinstance(sensor, MockUltrasonic)


def test_build_world_model_returns_rssm() -> None:
    cfg = _mock_settings()
    wm = build_world_model(cfg)
    assert isinstance(wm, RSSM)


def test_build_safety_monitor_returns_monitor() -> None:
    cfg = _mock_settings()
    monitor = build_safety_monitor(cfg)
    assert isinstance(monitor, MouseDroidSafetyMonitor)


def test_build_agent_returns_navigation_agent() -> None:
    cfg = _mock_settings()
    wm = build_world_model(cfg)
    agent = build_agent(cfg, wm)
    assert isinstance(agent, MouseDroidNavigationAgent)


def test_build_agent_has_name() -> None:
    cfg = _mock_settings()
    wm = build_world_model(cfg)
    agent = build_agent(cfg, wm)
    assert agent.name == "mouse_droid_navigator"


def test_build_distance_sensor_mock_has_range() -> None:
    cfg = _mock_settings()
    sensor = build_distance_sensor(cfg)
    assert sensor.max_range_m > 0
    assert sensor.min_range_m >= 0


def test_build_camera_mock_has_feature_dim() -> None:
    cfg = _mock_settings()
    cam = build_camera(cfg)
    assert cam.feature_dim > 0
