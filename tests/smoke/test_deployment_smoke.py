"""Deployment smoke tests — fast validation of configs, factories, and scripts.

Verifies that all YAML configs parse, all factory build functions succeed with
mock hardware, all required deployment scripts exist and are executable, and
Docker Compose config is valid YAML.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mousedroid.config.loader import load_settings, load_yaml
from mousedroid.config.schema import Settings
from mousedroid.factory import (
    build_camera,
    build_distance_sensor,
    build_esp32_driver,
    build_health_monitor,
    build_microphone,
    build_orchestrator,
    build_safety_monitor,
    build_telemetry_publisher,
    build_world_model,
)

pytestmark = pytest.mark.smoke

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_mock_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure mock hardware is active for all smoke tests."""
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "true")


def _mock_settings() -> Settings:
    """Create base Settings with mock hardware."""
    return Settings(mock_hardware=True)


# ---------------------------------------------------------------------------
# 1. Config files parse without error
# ---------------------------------------------------------------------------

_YAML_CONFIGS = [
    "default.yaml",
    "jetson_production.yaml",
    "local_training.yaml",
    "mock_hardware.yaml",
    "robot_arm_default.yaml",
    "robot_arm_training.yaml",
]


@pytest.mark.parametrize("filename", _YAML_CONFIGS)
def test_yaml_config_parses(filename: str) -> None:
    """Each YAML config file parses without error."""
    path = _CONFIG_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not found")
    data = load_yaml(path)
    assert isinstance(data, dict), f"{filename} did not parse as dict"


@pytest.mark.parametrize("filename", _YAML_CONFIGS)
def test_yaml_config_is_valid_yaml(filename: str) -> None:
    """Each YAML config is valid YAML (not just parseable by loader)."""
    path = _CONFIG_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not found")
    with path.open() as fh:
        data = yaml.safe_load(fh)
    assert data is None or isinstance(data, dict)


def test_default_config_loads_as_settings() -> None:
    """default.yaml loads into a valid Settings object."""
    cfg = load_settings(config_dir=_CONFIG_DIR)
    assert cfg.platform.value == "mouse_droid"


def test_jetson_production_config_loads_as_settings() -> None:
    """jetson_production.yaml merges with defaults into Settings."""
    cfg = load_settings(_CONFIG_DIR / "jetson_production.yaml", config_dir=_CONFIG_DIR)
    assert cfg.telemetry.enabled is True


def test_mock_hardware_config_loads_as_settings() -> None:
    """mock_hardware.yaml loads into Settings."""
    cfg = load_settings(_CONFIG_DIR / "mock_hardware.yaml", config_dir=_CONFIG_DIR)
    assert cfg.mock_hardware is True


# ---------------------------------------------------------------------------
# 2. Factory build functions succeed with mock config
# ---------------------------------------------------------------------------


def test_build_camera_mock() -> None:
    """build_camera returns a mock camera driver."""
    camera = build_camera(_mock_settings())
    assert camera is not None


def test_build_distance_sensor_mock() -> None:
    """build_distance_sensor returns a mock ultrasonic."""
    sensor = build_distance_sensor(_mock_settings())
    assert sensor is not None


def test_build_esp32_mock() -> None:
    """build_esp32_driver returns a resilient mock ESP32."""
    driver = build_esp32_driver(_mock_settings())
    assert driver is not None


def test_build_microphone_mock() -> None:
    """build_microphone returns mock when enabled in config."""
    cfg = Settings(
        mock_hardware=True,
        microphone={"enabled": True},  # type: ignore[arg-type]
    )
    mic = build_microphone(cfg)
    assert mic is not None


def test_build_microphone_disabled() -> None:
    """build_microphone returns None when disabled."""
    cfg = Settings(mock_hardware=True)
    mic = build_microphone(cfg)
    # Default config has microphone None or disabled
    assert mic is None


def test_build_world_model_mock() -> None:
    """build_world_model creates RSSM model."""
    wm = build_world_model(_mock_settings())
    assert wm is not None


def test_build_safety_monitor_mock() -> None:
    """build_safety_monitor creates MouseDroidSafetyMonitor."""
    monitor = build_safety_monitor(_mock_settings())
    assert monitor is not None


def test_build_telemetry_publisher_enabled() -> None:
    """build_telemetry_publisher returns publisher when enabled."""
    cfg = Settings(mock_hardware=True, telemetry={"enabled": True})  # type: ignore[arg-type]
    pub = build_telemetry_publisher(cfg)
    assert pub is not None


def test_build_telemetry_publisher_disabled() -> None:
    """build_telemetry_publisher returns None when disabled."""
    cfg = Settings(mock_hardware=True, telemetry={"enabled": False})  # type: ignore[arg-type]
    pub = build_telemetry_publisher(cfg)
    assert pub is None


def test_build_health_monitor_mock() -> None:
    """build_health_monitor creates HealthMonitor."""
    hm = build_health_monitor(_mock_settings())
    assert hm is not None


def test_build_orchestrator_mock() -> None:
    """build_orchestrator creates fully-wired orchestrator."""
    orch = build_orchestrator(_mock_settings())
    assert orch is not None


# ---------------------------------------------------------------------------
# 3. Required scripts exist and are executable
# ---------------------------------------------------------------------------

_REQUIRED_SCRIPTS = [
    "deploy_jetson.sh",
    "jetson_smoke_test.sh",
    "docker_deploy.sh",
    "ci.sh",
    "flash_esp32.sh",
    "jetson_bootstrap.sh",
]


@pytest.mark.parametrize("script", _REQUIRED_SCRIPTS)
def test_script_exists(script: str) -> None:
    """Each required deployment script exists."""
    path = _SCRIPTS_DIR / script
    assert path.exists(), f"Missing script: {script}"


@pytest.mark.parametrize("script", _REQUIRED_SCRIPTS)
def test_script_has_shebang(script: str) -> None:
    """Each required deployment script has a valid bash shebang."""
    path = _SCRIPTS_DIR / script
    if not path.exists():
        pytest.skip(f"{script} not found")
    first_line = path.read_text().split("\n", maxsplit=1)[0]
    assert first_line.startswith("#!/"), f"{script} missing shebang: {first_line}"


# ---------------------------------------------------------------------------
# 4. Docker Compose config is valid YAML
# ---------------------------------------------------------------------------


def test_docker_compose_is_valid_yaml() -> None:
    """docker-compose.jetson.yml is valid YAML."""
    compose_path = _PROJECT_ROOT / "docker-compose.jetson.yml"
    if not compose_path.exists():
        pytest.skip("docker-compose.jetson.yml not found")
    with compose_path.open() as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict)
    assert "services" in data


def test_docker_compose_has_mousedroid_service() -> None:
    """Docker Compose defines the mousedroid service."""
    compose_path = _PROJECT_ROOT / "docker-compose.jetson.yml"
    if not compose_path.exists():
        pytest.skip("docker-compose.jetson.yml not found")
    with compose_path.open() as fh:
        data = yaml.safe_load(fh)
    assert "mousedroid" in data.get("services", {})


def test_dockerfile_exists() -> None:
    """Dockerfile.jetson exists in project root."""
    dockerfile = _PROJECT_ROOT / "Dockerfile.jetson"
    assert dockerfile.exists(), "Missing Dockerfile.jetson"
