"""Tests targeting specific uncovered lines identified in coverage report."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from mousedroid.config.schema import (
    HealthConfig,
    JetsonConfig,
    SafetyConfig,
    Settings,
)

# ---------------------------------------------------------------------------
# TensorRT optimizer — line 49-55 (enabled=True branch)
# ---------------------------------------------------------------------------


def test_tensorrt_optimizer_enabled():
    cfg = JetsonConfig(tensorrt_enabled=True, precision="fp16", workspace_gb=1.0, dla_enabled=False)
    from mousedroid.efficiency.tensorrt import TensorRTOptimizer

    opt = TensorRTOptimizer(cfg)
    model = torch.nn.Linear(10, 3)
    sample = torch.randn(1, 10)

    with patch.object(opt, "_compile", return_value=model) as mock_compile:
        result = opt.optimize(model, sample)
        mock_compile.assert_called_once_with(model, sample)
        assert result is model


# ---------------------------------------------------------------------------
# Health monitor — line 102 (_read_sysfs with real file)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_monitor_read_sysfs(tmp_path):
    temp_file = tmp_path / "temp"
    temp_file.write_text("45000\n")
    load_file = tmp_path / "load"
    load_file.write_text("750\n")

    cfg = HealthConfig()
    jetson_cfg = JetsonConfig(thermal_zone_path=temp_file, gpu_load_path=load_file)
    from mousedroid.health.monitor import HealthMonitor

    mon = HealthMonitor(cfg, jetson_cfg)
    temp = await mon.read_gpu_temp_c()
    assert temp == pytest.approx(45.0)

    load = await mon.read_gpu_load_pct()
    assert load == pytest.approx(75.0)


@pytest.mark.asyncio
async def test_health_monitor_check_health(tmp_path):
    temp_file = tmp_path / "temp"
    temp_file.write_text("45000\n")
    load_file = tmp_path / "load"
    load_file.write_text("500\n")

    cfg = HealthConfig()
    jetson_cfg = JetsonConfig(thermal_zone_path=temp_file, gpu_load_path=load_file)
    from mousedroid.health.monitor import HealthMonitor

    mon = HealthMonitor(cfg, jetson_cfg)
    result = await mon.check_health()
    assert result["status"] == "ok"
    assert result["gpu_temp_c"] == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# Safety monitor — line 121 (human detection emergency)
# ---------------------------------------------------------------------------


def test_safety_monitor_human_detection_emergency():
    from mousedroid.safety.monitor import MouseDroidSafetyMonitor

    cfg = SafetyConfig(min_forward_clearance_m=0.20)
    mon = MouseDroidSafetyMonitor(cfg)

    obs = MagicMock()
    obs.distance_m = 1.0  # clearance ok
    obs.motor_state = np.array([0.0, 0.0, 0.0, 12.0], dtype=np.float32)
    obs.valid_mask = np.ones(4, dtype=np.float32)
    obs.human_detected = True
    obs.human_dist_m = 0.1  # closer than min_forward_clearance_m

    ctx = mon.evaluate(obs, loop_time_ms=10.0)
    assert ctx.is_emergency is True
    assert ctx.human_detected is True


# ---------------------------------------------------------------------------
# MCTS _Node — line 33 (unvisited node mean_value)
# ---------------------------------------------------------------------------


def test_mcts_node_unvisited_mean_value():
    from mousedroid.world_model.mcts import _Node

    node = _Node(action=torch.zeros(3), h=torch.zeros(1, 64), z=torch.zeros(1, 32))
    assert node.mean_value == 0.0
    assert node.visit_count == 0


# ---------------------------------------------------------------------------
# RSSM imagine_step — line 198 (1-D action unsqueeze)
# ---------------------------------------------------------------------------


def test_rssm_imagine_step_1d_action():
    from mousedroid.config.schema import ModelConfig
    from mousedroid.world_model.rssm import RSSM

    cfg = ModelConfig()
    rssm = RSSM(cfg)
    rssm.eval()

    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)
    action_1d = torch.zeros(cfg.action_dim)  # 1-D, should be unsqueezed

    with torch.no_grad():
        new_h, new_z, _reward = rssm.imagine_step(action_1d, h, z)

    assert new_h.shape == (1, cfg.hidden_dim)
    assert new_z.shape == (1, cfg.latent_dim)


# ---------------------------------------------------------------------------
# Experience logger — line 99 (read deserialization)
# ---------------------------------------------------------------------------


def test_experience_logger_read_roundtrip(tmp_path):
    from mousedroid.config.schema import ExperienceConfig
    from mousedroid.experience.logger import ExperienceLogger
    from mousedroid.experience.record import MouseDroidExperienceRecord
    from tests import TEST_EXPERIENCE_MAP_SIZE_GB

    cfg = ExperienceConfig(path=str(tmp_path / "xp"), map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB)
    logger = ExperienceLogger(cfg)
    logger.open()

    record = MouseDroidExperienceRecord(
        distance_m=1.5,
        reward=0.8,
        surprise=0.3,
    )
    # Generate key the same way logger does internally, then log
    key = logger._make_key()
    # Write via LMDB directly using the known key
    with logger._env.begin(write=True) as txn:
        txn.put(key, record.serialize())

    loaded = logger.read(key)
    assert loaded is not None
    assert loaded.distance_m == pytest.approx(1.5)
    assert loaded.reward == pytest.approx(0.8)

    logger.close()


# ---------------------------------------------------------------------------
# Episodic memory — line 66 (all non-finite priorities)
# ---------------------------------------------------------------------------


def test_episodic_memory_non_finite_priorities():
    from mousedroid.config.schema import MemoryConfig
    from mousedroid.memory.episodic import EpisodicReplay

    cfg = MemoryConfig(episodic_capacity=100)
    mem = EpisodicReplay(cfg)

    # Push records normally then inject non-finite priorities directly
    for _i in range(5):
        mem.push(np.zeros(8, dtype=np.float32), priority=1.0)

    # Force all priorities to NaN to hit the non-finite fallback branch
    # Buffer entries are (experience, priority, seq) — preserve seq, which the
    # sampler relies on for stable ordering.
    for i in range(len(mem._buffer)):
        exp, _, seq = mem._buffer[i]
        mem._buffer[i] = (exp, float("nan"), seq)

    # Should not raise, uses uniform sampling fallback
    samples = mem.sample(batch_size=2)
    assert len(samples) == 2


# ---------------------------------------------------------------------------
# EWC — line 72 (gradient accumulation)
# ---------------------------------------------------------------------------


def test_ewc_consolidate_with_gradients(monkeypatch):
    from mousedroid.config.schema import LearningConfig
    from mousedroid.learning.ewc import EWCAgent

    model = torch.nn.Linear(4, 2)
    cfg = LearningConfig(ewc_lambda=1.0, ewc_fisher_samples=3)
    ewc = EWCAgent(cfg, model)

    # Generate some gradients
    x = torch.randn(1, 4)
    loss = model(x).sum()
    loss.backward()

    # Prevent consolidate()'s zero_grad() from clearing grads to None,
    # so the param.grad is not None branch is actually exercised.
    monkeypatch.setattr(model, "zero_grad", lambda **kw: None)

    ewc.consolidate()
    assert ewc._fisher is not None
    assert len(ewc._fisher) > 0
    assert any(v.abs().sum() > 0 for v in ewc._fisher.values())


# ---------------------------------------------------------------------------
# Factory — picamera2 branch (mock successful import)
# ---------------------------------------------------------------------------


def test_factory_build_camera_picamera2_backend():
    cfg = Settings(
        mock_hardware=False,
        ultrasonic={"trigger_pin": 17, "echo_pin": 27},
        camera={"backend": "picamera2"},
    )
    from mousedroid.factory import build_camera
    from mousedroid.hardware.camera.imx500 import IMX500Camera

    camera = build_camera(cfg)
    assert isinstance(camera, IMX500Camera)


# ---------------------------------------------------------------------------
# Shared comms constants are importable
# ---------------------------------------------------------------------------


def test_comms_constants_importable():
    from mousedroid.comms._utils import (
        ESP32_CMD_TYPE_BATTERY,
        ESP32_CMD_TYPE_STOP,
        ESP32_CMD_TYPE_VELOCITY,
        MAX_PWM,
    )

    assert MAX_PWM == 255
    assert ESP32_CMD_TYPE_VELOCITY == 1
    assert ESP32_CMD_TYPE_STOP == 0
    assert ESP32_CMD_TYPE_BATTERY == 2


# ---------------------------------------------------------------------------
# Bundle and record use named constants
# ---------------------------------------------------------------------------


def test_bundle_default_dimensions():
    from mousedroid.constants import (
        DEFAULT_MAX_DISTANCE_M,
        DEFAULT_MOTOR_STATE_DIM,
        DEFAULT_VISION_DIM,
    )
    from mousedroid.sensing.bundle import MouseDroidObservationBundle

    bundle = MouseDroidObservationBundle()
    assert bundle.vision_features.shape == (DEFAULT_VISION_DIM,)
    assert bundle.motor_state.shape == (DEFAULT_MOTOR_STATE_DIM,)
    assert bundle.distance_m == DEFAULT_MAX_DISTANCE_M


def test_record_default_dimensions():
    from mousedroid.constants import (
        DEFAULT_ACTION_DIM,
        DEFAULT_MAX_DISTANCE_M,
        DEFAULT_MOTOR_STATE_DIM,
        DEFAULT_VISION_DIM,
    )
    from mousedroid.experience.record import MouseDroidExperienceRecord

    record = MouseDroidExperienceRecord()
    assert record.vision_features.shape == (DEFAULT_VISION_DIM,)
    assert record.motor_state.shape == (DEFAULT_MOTOR_STATE_DIM,)
    assert record.action.shape == (DEFAULT_ACTION_DIM,)
    assert record.distance_m == DEFAULT_MAX_DISTANCE_M


# ---------------------------------------------------------------------------
# Config loader logging (smoke test)
# ---------------------------------------------------------------------------


def test_config_loader_logs(tmp_path):
    from mousedroid.config.loader import load_settings

    # No default.yaml exists in tmp_path — should still work with defaults
    settings = load_settings(config_dir=tmp_path)
    # Just verify we got a valid Settings instance
    assert settings is not None


def test_config_loader_drops_empty_nested_env_vars(monkeypatch, tmp_path):
    """Regression: empty MOUSEDROID_<SECTION>__<FIELD>= env vars must not
    materialise a partially-populated nested config.

    Before the fix, `MOUSEDROID_GCP__PROJECT_ID=""` (e.g. from a
    docker-compose image that baked in `ENV MOUSEDROID_GCP__PROJECT_ID=`)
    caused pydantic-settings v2 to instantiate
    ``GCPConfig(project_id="")`` which fails validation with
    ``GCPConfig requires non-empty values for: project_id``.

    The loader should treat empty/whitespace nested env vars as "unset"
    and leave optional nested configs ``None``.
    """
    import os

    from mousedroid.config.loader import load_settings

    # Simulate the docker image baking in an empty GCP project_id env var.
    monkeypatch.setenv("MOUSEDROID_GCP__PROJECT_ID", "")
    # Also test a whitespace-only value is dropped.
    monkeypatch.setenv("MOUSEDROID_GCP__ROBOT_ID", "   ")
    # And a legitimate non-empty nested env var should still pass through
    # untouched (the optional nested config stays None because project_id
    # would still be empty, but the env var itself must not be popped).
    monkeypatch.setenv("MOUSEDROID_LOGGING__LEVEL", "INFO")

    settings = load_settings(config_dir=tmp_path)

    # Optional gcp config must remain None (offline mode) rather than
    # erroring on empty project_id.
    assert settings.gcp is None
    # Empty nested vars are restored to os.environ after load_settings.
    assert os.environ.get("MOUSEDROID_GCP__PROJECT_ID") == ""
    assert os.environ.get("MOUSEDROID_GCP__ROBOT_ID") == "   "
    # Non-empty nested env vars are never dropped.
    assert os.environ.get("MOUSEDROID_LOGGING__LEVEL") == "INFO"


def test_load_jetson_runtime_settings_drops_empty_nested_gcp_env(monkeypatch, tmp_path):
    from tests._jetson_hardware import load_jetson_runtime_settings

    jetson_cfg = tmp_path / "jetson_production.yaml"
    jetson_cfg.write_text("mock_hardware: false\nlogging:\n  level: INFO\n")

    monkeypatch.delenv("MOUSEDROID_JETSON_CONFIG", raising=False)
    monkeypatch.setenv("MOUSEDROID_JETSON_CONFIGS", str(jetson_cfg))
    monkeypatch.setenv("MOUSEDROID_GCP__PROJECT_ID", "")
    monkeypatch.setenv("MOUSEDROID_GCP__ROBOT_ID", "   ")

    settings = load_jetson_runtime_settings()

    assert settings.gcp is None
    assert settings.mock_hardware is False


def test_hardware_e2e_helpers_drop_empty_nested_gcp_env(monkeypatch, tmp_path):
    from tests.hardware.test_e2e_edge_cases import _load_settings as load_edge_settings
    from tests.hardware.test_e2e_sense_plan_act import _load_settings as load_e2e_settings

    jetson_cfg = tmp_path / "jetson_production.yaml"
    jetson_cfg.write_text("mock_hardware: false\nlogging:\n  level: INFO\n")

    monkeypatch.delenv("MOUSEDROID_JETSON_CONFIG", raising=False)
    monkeypatch.setenv("MOUSEDROID_JETSON_CONFIGS", str(jetson_cfg))
    monkeypatch.setenv("MOUSEDROID_GCP__PROJECT_ID", "")
    monkeypatch.setenv("MOUSEDROID_GCP__ROBOT_ID", "   ")

    assert load_e2e_settings().gcp is None
    assert load_edge_settings().gcp is None


# ---------------------------------------------------------------------------
# Planning budget logging
# ---------------------------------------------------------------------------


def test_planning_budget_computation():
    from mousedroid.agents._planning import compute_mcts_budget

    budget = compute_mcts_budget(surprise=0.0, base=50, maximum=200)
    assert budget == 50

    budget = compute_mcts_budget(surprise=3.0, base=50, maximum=200)
    assert 50 <= budget <= 200


# ---------------------------------------------------------------------------
# LLM gateway config now has velocity norms and system prompt
# ---------------------------------------------------------------------------


def test_gateway_config_has_velocity_norms():
    from mousedroid.llm_gateway.config import GatewayConfig

    cfg = GatewayConfig(model_path="/tmp/model.gguf")
    assert cfg.max_vx_norm_mps == 0.5
    assert cfg.max_vy_norm_mps == 0.3
    assert cfg.max_omega_norm_rads == 2.0
    assert "MSE-6" in cfg.system_prompt
