"""Backwards compatibility tests for GCP configuration.

Verifies that existing YAML configs and Settings instantiation work
unchanged after adding GCP config models.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def test_settings_default_gcp_is_none() -> None:
    """Settings() with mock_hardware=True defaults gcp to None."""
    from mousedroid.config.schema import Settings

    s = Settings(mock_hardware=True)
    assert s.gcp is None


def test_settings_with_gcp_project_id() -> None:
    """Settings with gcp.project_id loads all sub-config defaults."""
    from mousedroid.config.schema import Settings

    s = Settings(mock_hardware=True, gcp={"project_id": "test-project"})
    assert s.gcp is not None
    assert s.gcp.project_id == "test-project"
    assert s.gcp.pubsub.telemetry_topic == "mousedroid-telemetry"
    assert s.gcp.pubsub.experience_topic == "mousedroid-experience"
    assert s.gcp.storage.bucket == "mousedroid-experience"
    assert s.gcp.storage.compression == "gzip"
    assert s.gcp.logging.enabled is True
    assert s.gcp.monitoring.enabled is True
    assert s.gcp.firestore.enabled is False
    assert s.gcp.training is None
    assert s.gcp.simulation is None
    assert s.gcp.robot_id == "droid-001"
    assert s.gcp.credentials_path is None


def test_settings_gcp_circuit_breaker_defaults() -> None:
    """GCP circuit breaker has cloud-tuned defaults (higher timeouts)."""
    from mousedroid.config.schema import Settings

    s = Settings(mock_hardware=True, gcp={"project_id": "test"})
    assert s.gcp is not None
    assert s.gcp.circuit_breaker.failure_threshold == 3
    assert s.gcp.circuit_breaker.recovery_timeout_s == 60.0
    assert s.gcp.circuit_breaker.half_open_max_calls == 1


def test_settings_gcp_retry_defaults() -> None:
    """GCP retry config has cloud-tuned defaults."""
    from mousedroid.config.schema import Settings

    s = Settings(mock_hardware=True, gcp={"project_id": "test"})
    assert s.gcp is not None
    assert s.gcp.retry.max_attempts == 3
    assert s.gcp.retry.base_delay_s == 2.0
    assert s.gcp.retry.max_delay_s == 60.0


def test_settings_gcp_with_training() -> None:
    """GCP training config loads with defaults when provided."""
    from mousedroid.config.schema import Settings

    s = Settings(
        mock_hardware=True,
        gcp={
            "project_id": "test",
            "training": {"training_bucket": "my-bucket"},
        },
    )
    assert s.gcp is not None
    assert s.gcp.training is not None
    assert s.gcp.training.training_bucket == "my-bucket"
    assert s.gcp.training.machine_type == "a2-highgpu-1g"
    assert s.gcp.training.ewc_enabled is True


def test_settings_gcp_with_simulation() -> None:
    """GCP simulation config loads with defaults when provided."""
    from mousedroid.config.schema import Settings

    s = Settings(
        mock_hardware=True,
        gcp={
            "project_id": "test",
            "simulation": {"gke_cluster": "my-cluster"},
        },
    )
    assert s.gcp is not None
    assert s.gcp.simulation is not None
    assert s.gcp.simulation.gke_cluster == "my-cluster"
    assert s.gcp.simulation.max_parallel_pods == 50


def test_mock_hardware_yaml_still_loads() -> None:
    """config/mock_hardware.yaml still loads without changes."""
    config_path = Path(__file__).resolve().parents[3] / "config" / "mock_hardware.yaml"
    if not config_path.exists():
        pytest.skip("mock_hardware.yaml not found")

    with config_path.open() as f:
        data = yaml.safe_load(f)

    from mousedroid.config.schema import Settings

    s = Settings(**data)
    assert s.mock_hardware is True
    assert s.gcp is None


def test_gcp_digital_twin_yaml_loads() -> None:
    """config/gcp_digital_twin.yaml loads as a valid overlay."""
    config_path = Path(__file__).resolve().parents[3] / "config" / "gcp_digital_twin.yaml"
    if not config_path.exists():
        pytest.skip("gcp_digital_twin.yaml not found")

    with config_path.open() as f:
        data = yaml.safe_load(f)

    from mousedroid.config.schema import Settings

    s = Settings(mock_hardware=True, **data)
    assert s.gcp is not None
    assert s.gcp.project_id == "mousedroid-twin"
    assert s.gcp.pubsub.telemetry_topic == "mousedroid-telemetry"


def test_gcp_project_id_required() -> None:
    """GCP config requires project_id when gcp section is provided."""
    from pydantic import ValidationError

    from mousedroid.config.schema import Settings

    with pytest.raises(ValidationError, match="project_id"):
        Settings(mock_hardware=True, gcp={})


def test_existing_fields_unchanged() -> None:
    """Core Settings fields remain unchanged after GCP addition."""
    from mousedroid.config.schema import Settings

    s = Settings(mock_hardware=True)
    assert s.platform == "mouse_droid"
    assert s.debug is False
    assert s.experience.flush_every_n == 30
    assert s.circuit_breaker.failure_threshold == 5
    assert s.retry.max_attempts == 3
