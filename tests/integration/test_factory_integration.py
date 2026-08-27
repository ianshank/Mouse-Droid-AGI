from __future__ import annotations

import importlib.util

import pytest

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
from mousedroid.world_model.protocol import SafetyTraceProtocol, WorldModelProtocol
from mousedroid.world_model.rssm import RSSM

_ncps_available = importlib.util.find_spec("ncps") is not None
_skip_no_ncps = pytest.mark.skipif(not _ncps_available, reason="ncps not installed")


def _mock_settings() -> Settings:
    return Settings(mock_hardware=True)


def test_build_orchestrator_creates_all_components() -> None:
    cfg = _mock_settings()
    orch = build_orchestrator(cfg)
    assert orch is not None


def test_build_orchestrator_threads_shared_failure_recorder_to_voice_engine() -> None:
    """The orchestrator's shared failure_recorder must also reach its voice engine.

    Regression test: build_orchestrator built one shared failure_recorder and
    passed it to the orchestrator itself, but its build_voice_engine call
    omitted the kwarg entirely, so the voice engine always fell back to its
    own NullFailureRecorder in production regardless of what was configured.
    """
    cfg = Settings(mock_hardware=True, voice={"enabled": True}, speaker={"enabled": True})
    orch = build_orchestrator(cfg)
    assert orch._voice_engine is not None
    assert orch._voice_engine._failure_recorder is orch._failure_recorder


def test_build_orchestrator_threads_gcp_observability_collaborators() -> None:
    """F-032: the two new cloud collaborators reach the orchestrator via the
    full build_orchestrator() composition path, not just their own builder
    functions tested in isolation (tests/unit/factory/
    test_factory_cloud_observability.py) or the orchestrator constructor
    called directly with hand-built fakes (tests/unit/orchestrator/
    test_cloud_subsystems_wiring.py). Both existing precedent builders
    (build_cloud_telemetry_sink/build_cloud_experience_exporter) have this
    same integration-tier gap; this closes it for the two new ones per the
    plan's own explicit requirement.

    Requires both real SDKs: a post-merge hotfix made every cloud builder
    genuinely probe google.cloud.* availability up front (see the
    F-032 openspec bundle's Copilot-findings addendum) rather than only
    guarding the lightweight wrapper import, so this repo's own CI (which
    installs no [gcp] extra in any job) must skip rather than fail here.
    """
    pytest.importorskip("google.cloud.monitoring_v3")
    pytest.importorskip("google.cloud.firestore")
    from mousedroid.cloud.firestore_sync import CloudFirestoreSync
    from mousedroid.cloud.monitoring_exporter import CloudMetricsExporter

    cfg = Settings(
        mock_hardware=True,
        gcp={
            "project_id": "test-project",
            "monitoring": {"enabled": True},
            "firestore": {"enabled": True},
        },
        memory={"enabled": True},
    )
    orch = build_orchestrator(cfg)
    assert isinstance(orch._cloud_metrics_exporter, CloudMetricsExporter)
    assert isinstance(orch._cloud_firestore_sync, CloudFirestoreSync)


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


# ---------------------------------------------------------------------------
# Dual-Stream RSSM factory integration tests
# ---------------------------------------------------------------------------


@_skip_no_ncps
def test_build_world_model_returns_dual_stream_when_cfc_enabled() -> None:
    """Factory builds DualStreamRSSM when cfc_hidden_dim > 0."""
    from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

    cfg = Settings(mock_hardware=True)
    cfg.model.cfc_hidden_dim = 16
    wm = build_world_model(cfg)
    assert isinstance(wm, DualStreamRSSM)


def test_build_world_model_returns_classic_rssm_when_cfc_zero() -> None:
    """Factory builds classic RSSM when cfc_hidden_dim == 0 (default)."""
    cfg = _mock_settings()
    assert cfg.model.cfc_hidden_dim == 0
    wm = build_world_model(cfg)
    assert isinstance(wm, RSSM)


@_skip_no_ncps
def test_dual_stream_world_model_conforms_to_protocol() -> None:
    """DualStreamRSSM satisfies WorldModelProtocol."""
    cfg = Settings(mock_hardware=True)
    cfg.model.cfc_hidden_dim = 16
    wm = build_world_model(cfg)
    assert isinstance(wm, WorldModelProtocol)


@_skip_no_ncps
def test_dual_stream_world_model_has_safety_trace() -> None:
    """DualStreamRSSM satisfies SafetyTraceProtocol."""
    cfg = Settings(mock_hardware=True)
    cfg.model.cfc_hidden_dim = 16
    wm = build_world_model(cfg)
    assert isinstance(wm, SafetyTraceProtocol)


def test_cfc_config_defaults_backward_compatible() -> None:
    """Existing config (no CfC fields) still works."""
    cfg = Settings(mock_hardware=True)
    assert cfg.model.cfc_hidden_dim == 0
    assert cfg.model.cfc_backbone_units == 64
    assert cfg.model.cfc_backbone_layers == 1
    assert cfg.model.cfc_mode == "default"
    assert cfg.model.cfc_sparsity_level == 0.5


def test_dual_stream_training_config_defaults() -> None:
    """DualStreamTrainingConfig has expected defaults."""
    cfg = Settings(mock_hardware=True)
    assert cfg.dual_stream_training.gru_lr == 3e-4
    assert cfg.dual_stream_training.cfc_lr == 1e-4
    assert cfg.dual_stream_training.gru_grad_clip == 10.0
    assert cfg.dual_stream_training.cfc_grad_clip == 1.0
