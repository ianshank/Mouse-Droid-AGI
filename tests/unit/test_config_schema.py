from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import (
    CameraConfig,
    CircuitBreakerConfig,
    CognitiveConfig,
    CuriosityConfig,
    ESP32Config,
    ExperienceConfig,
    HailoConfig,
    HealthConfig,
    JetsonConfig,
    LearningConfig,
    LoggingConfig,
    LoopConfig,
    MCTSConfig,
    MemoryConfig,
    MetricsConfig,
    ModelConfig,
    PlatformType,
    RetryConfig,
    RewardConfig,
    RobotConfig,
    SafetyConfig,
    Settings,
    SurpriseConfig,
    TrainingConfig,
    UltrasonicConfig,
)

# -- PlatformType enum --------------------------------------------------------


def test_platform_type_mouse_droid():
    assert PlatformType.MOUSE_DROID == "mouse_droid"


def test_platform_type_is_str_enum():
    assert isinstance(PlatformType.MOUSE_DROID, str)


# -- Nested config defaults ---------------------------------------------------


def test_camera_config_defaults():
    c = CameraConfig()
    assert c.fps == 30
    assert c.feature_dim == 256


def test_circuit_breaker_config_defaults():
    c = CircuitBreakerConfig()
    assert c.failure_threshold == 5


def test_curiosity_config_defaults():
    c = CuriosityConfig()
    assert c.intrinsic_reward_scale == 0.1


def test_esp32_config_defaults():
    c = ESP32Config()
    assert c.protocol == "serial"
    assert c.serial_baud == 1_000_000


def test_experience_config_defaults():
    c = ExperienceConfig()
    assert c.map_size_gb == 20


def test_health_config_defaults():
    c = HealthConfig()
    assert c.check_interval_s == 5.0


def test_jetson_config_defaults():
    c = JetsonConfig()
    assert c.precision == "fp16"
    assert c.power_mode == "15W"


def test_learning_config_defaults():
    c = LearningConfig()
    assert c.ewc_lambda == 5000.0


def test_logging_config_defaults():
    c = LoggingConfig()
    assert c.level == "INFO"
    assert c.format == "json"


def test_loop_config_defaults():
    c = LoopConfig()
    assert c.perception_hz == 30.0


def test_mcts_config_defaults():
    c = MCTSConfig()
    assert c.n_simulations_base == 50


def test_memory_config_defaults():
    c = MemoryConfig()
    assert c.working_context_size == 8192


def test_metrics_config_defaults():
    c = MetricsConfig()
    assert c.enabled is True


def test_model_config_defaults():
    c = ModelConfig()
    assert c.vision_dim == 256
    assert c.action_dim == 3


def test_retry_config_defaults():
    c = RetryConfig()
    assert c.max_attempts == 3


def test_reward_config_defaults():
    c = RewardConfig()
    assert c.weight_truthfulness == 0.4


def test_robot_config_defaults():
    c = RobotConfig()
    assert c.wheel_type == "mecanum"


def test_safety_config_defaults():
    c = SafetyConfig()
    assert c.min_forward_clearance_m == 0.20


def test_surprise_config_defaults():
    c = SurpriseConfig()
    assert c.ema_alpha == 0.1


def test_training_config_defaults():
    c = TrainingConfig()
    assert c.batch_size == 32
    assert c.gpu.require_cuda is False
    assert c.generation.log_every_n_episodes == 100
    assert c.annotation.n_episodes == 500
    assert c.annotation.obstacle_clearance_m == 0.25
    assert c.warmstart.latent_stats_max_episodes == 100
    assert c.constitutional.validation_battery_v == 12.0


def test_cognitive_config_huggingface_defaults():
    c = CognitiveConfig()
    assert c.huggingface_repo == "ianshank/mousedroid-weights"
    assert c.huggingface_subfolder == "bdi"


def test_cognitive_config_rejects_repo_with_extra_path_segments():
    with pytest.raises(ValidationError):
        CognitiveConfig(huggingface_repo="owner/nested/repo")


@pytest.mark.parametrize("subfolder", ["/bdi", "bdi//nested", "../bdi", "bdi/../nested"])
def test_cognitive_config_rejects_invalid_huggingface_subfolder(subfolder: str):
    with pytest.raises(ValidationError):
        CognitiveConfig(huggingface_subfolder=subfolder)


@pytest.mark.parametrize("subfolder", ["", "bdi", "bdi/nested"])
def test_cognitive_config_accepts_relative_huggingface_subfolder(subfolder: str):
    c = CognitiveConfig(huggingface_subfolder=subfolder)
    assert c.huggingface_subfolder == subfolder


# -- UltrasonicConfig range_ordering -----------------------------------------


def test_ultrasonic_config_valid():
    c = UltrasonicConfig(trigger_pin=23, echo_pin=24, max_range_m=4.0, min_range_m=0.02)
    assert c.max_range_m == 4.0


def test_ultrasonic_config_invalid_range_ordering():
    with pytest.raises(ValidationError):
        UltrasonicConfig(trigger_pin=23, echo_pin=24, max_range_m=0.01, min_range_m=0.02)


def test_ultrasonic_config_equal_ranges_invalid():
    with pytest.raises(ValidationError):
        UltrasonicConfig(trigger_pin=23, echo_pin=24, max_range_m=1.0, min_range_m=1.0)


# -- Settings hardware_requires_pins -----------------------------------------


def test_settings_mock_true_without_ultrasonic_ok():
    s = Settings(mock_hardware=True)
    assert s.ultrasonic is None


def test_settings_mock_false_without_ultrasonic_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "false")
    with pytest.raises(ValidationError, match="at least one distance sensor"):
        Settings(mock_hardware=False, ultrasonic=None)


def test_settings_mock_false_with_ultrasonic_ok(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOUSEDROID_MOCK_HARDWARE", "false")
    s = Settings(
        mock_hardware=False,
        ultrasonic=UltrasonicConfig(trigger_pin=23, echo_pin=24),
    )
    assert s.ultrasonic is not None


# -- ESP32Config protocol literal values --------------------------------------


def test_esp32_protocol_serial():
    c = ESP32Config(protocol="serial")
    assert c.protocol == "serial"


def test_esp32_protocol_wifi():
    c = ESP32Config(protocol="wifi")
    assert c.protocol == "wifi"


def test_esp32_protocol_invalid():
    with pytest.raises(ValidationError):
        ESP32Config(protocol="bluetooth")


# -- JetsonConfig precision ---------------------------------------------------


@pytest.mark.parametrize("prec", ["fp32", "fp16", "int8"])
def test_jetson_precision_values(prec):
    c = JetsonConfig(precision=prec)
    assert c.precision == prec


def test_jetson_precision_invalid():
    with pytest.raises(ValidationError):
        JetsonConfig(precision="bf16")


# -- RobotConfig wheel_type --------------------------------------------------


@pytest.mark.parametrize("wt", ["mecanum", "standard"])
def test_robot_wheel_type_values(wt):
    c = RobotConfig(wheel_type=wt)
    assert c.wheel_type == wt


def test_robot_wheel_type_invalid():
    with pytest.raises(ValidationError):
        RobotConfig(wheel_type="omni")


# -- gt=0 field constraints ---------------------------------------------------


def test_camera_fps_rejects_zero():
    with pytest.raises(ValidationError):
        CameraConfig(fps=0)


def test_camera_fps_rejects_negative():
    with pytest.raises(ValidationError):
        CameraConfig(fps=-1)


def test_loop_perception_hz_rejects_zero():
    with pytest.raises(ValidationError):
        LoopConfig(perception_hz=0)


def test_model_vision_dim_rejects_zero():
    with pytest.raises(ValidationError):
        ModelConfig(vision_dim=0)


def test_esp32_serial_baud_rejects_zero():
    with pytest.raises(ValidationError):
        ESP32Config(serial_baud=0)


# -- RewardConfig weights valid range ----------------------------------------


def test_reward_weights_valid():
    c = RewardConfig(
        weight_truthfulness=0.0,
        weight_helpfulness=1.0,
        weight_safety=0.5,
        weight_engagement=0.5,
    )
    assert c.weight_truthfulness == 0.0
    assert c.weight_helpfulness == 1.0


def test_reward_weight_above_one_raises():
    with pytest.raises(ValidationError):
        RewardConfig(weight_truthfulness=1.1)


def test_reward_weight_below_zero_raises():
    with pytest.raises(ValidationError):
        RewardConfig(weight_truthfulness=-0.1)


# -- CameraConfig fps bounds -------------------------------------------------


def test_camera_fps_at_max():
    c = CameraConfig(fps=120)
    assert c.fps == 120


def test_camera_fps_above_max_raises():
    with pytest.raises(ValidationError):
        CameraConfig(fps=121)


# -- MOUSEDROID_ env prefix override -----------------------------------------


def test_env_prefix_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOUSEDROID_DEBUG", "true")
    s = Settings(mock_hardware=True)
    assert s.debug is True


# -- ModelConfig audio fields -------------------------------------------------


def test_model_config_audio_defaults():
    c = ModelConfig()
    assert c.audio_dim == 0
    assert c.audio_proj_dim == 32


def test_model_config_audio_custom():
    c = ModelConfig(audio_dim=1024, audio_proj_dim=64)
    assert c.audio_dim == 1024
    assert c.audio_proj_dim == 64


def test_model_config_audio_dim_negative_raises():
    with pytest.raises(ValidationError):
        ModelConfig(audio_dim=-1)


def test_model_config_audio_proj_dim_negative_raises():
    with pytest.raises(ValidationError):
        ModelConfig(audio_proj_dim=-1)


# -- CameraConfig feature extractor fields ------------------------------------


def test_camera_config_feature_extractor_default():
    c = CameraConfig()
    assert c.feature_extractor == "mean_pool"


def test_camera_config_feature_extractor_values():
    for val in ("mean_pool", "tensorrt", "hailo", "auto"):
        c = CameraConfig(feature_extractor=val)
        assert c.feature_extractor == val


def test_camera_config_feature_extractor_invalid():
    with pytest.raises(ValidationError):
        CameraConfig(feature_extractor="custom")


def test_camera_config_l2_normalize_default():
    c = CameraConfig()
    assert c.l2_normalize is True


def test_camera_config_l2_normalize_false():
    c = CameraConfig(l2_normalize=False)
    assert c.l2_normalize is False


# -- HailoConfig ---------------------------------------------------------------


def test_hailo_config_defaults():
    c = HailoConfig()
    assert c.enabled is False
    assert c.device_path == "/dev/hailo0"
    assert c.batch_size == 1
    assert c.power_mode == "performance"
    assert c.input_format == "uint8"
    assert c.timeout_ms == 100.0
    assert c.fallback_on_failure is True


def test_hailo_config_enabled():
    c = HailoConfig(enabled=True)
    assert c.enabled is True


def test_hailo_config_power_mode_values():
    for val in ("performance", "balanced", "power_save"):
        c = HailoConfig(power_mode=val)
        assert c.power_mode == val


def test_hailo_config_power_mode_invalid():
    with pytest.raises(ValidationError):
        HailoConfig(power_mode="turbo")


def test_hailo_config_batch_size_bounds():
    c = HailoConfig(batch_size=8)
    assert c.batch_size == 8
    with pytest.raises(ValidationError):
        HailoConfig(batch_size=0)
    with pytest.raises(ValidationError):
        HailoConfig(batch_size=9)


def test_hailo_config_none_on_settings():
    s = Settings(mock_hardware=True)
    assert s.hailo is None


def test_hailo_config_on_settings():
    s = Settings(mock_hardware=True, hailo=HailoConfig(enabled=True))
    assert s.hailo is not None
    assert s.hailo.enabled is True


# -- Settings with full valid config ------------------------------------------


def test_settings_full_valid():
    s = Settings(
        mock_hardware=True,
        platform=PlatformType.MOUSE_DROID,
        debug=False,
    )
    assert s.platform == PlatformType.MOUSE_DROID
    assert s.mock_hardware is True
    assert s.loop.perception_hz == 30.0


# -- Backwards compat: minimal args ------------------------------------------


def test_settings_minimal_args():
    s = Settings(mock_hardware=True)
    assert s.esp32.protocol == "serial"
    assert s.camera.fps == 30


# -- Harness config (agent-harness layer) ------------------------------------


def test_settings_harness_default_is_none():
    """Harness must default to None to keep existing YAML byte-identical."""
    s = Settings(mock_hardware=True)
    assert s.harness is None


def test_harness_config_defaults():
    from mousedroid.config.schema import HarnessConfig

    cfg = HarnessConfig()
    assert cfg.tracker.enabled is False
    assert cfg.tracker.history_size == 256
    assert cfg.tracker.default_timeout_s == 30.0
    assert cfg.tracker.max_active == 8
    assert cfg.hooks.enabled_hooks == []
    assert cfg.hooks.error_policy == "warn"
    assert cfg.hooks.journal_events is True
    assert cfg.hooks.fail_fast is False
    assert cfg.journal.backend == "null"
    assert cfg.journal.queue_max == 1024
    assert cfg.approval.gate == "auto"
    assert cfg.approval.on_timeout == "deny"
    assert cfg.skills.enabled is False
    assert cfg.skills.backend == "noop"


def test_harness_config_invalid_history_size_rejected():
    from mousedroid.config.schema import HarnessTrackerConfig

    with pytest.raises(ValidationError):
        HarnessTrackerConfig(history_size=0)


def test_harness_config_invalid_journal_backend_rejected():
    from mousedroid.config.schema import HarnessJournalConfig

    with pytest.raises(ValidationError):
        HarnessJournalConfig(backend="not_a_backend")  # type: ignore[arg-type]


def test_harness_config_invalid_approval_gate_rejected():
    from mousedroid.config.schema import HarnessApprovalConfig

    with pytest.raises(ValidationError):
        HarnessApprovalConfig(gate="bogus")  # type: ignore[arg-type]


def test_settings_with_harness_enabled():
    from mousedroid.config.schema import HarnessConfig

    s = Settings(mock_hardware=True, harness=HarnessConfig())
    assert s.harness is not None
    assert s.harness.tracker.enabled is False
    assert s.harness.journal.backend == "null"


def test_harness_approval_timeout_decision_configurable():
    from mousedroid.config.schema import HarnessApprovalConfig

    cfg = HarnessApprovalConfig(on_timeout="approve")
    assert cfg.on_timeout == "approve"


# -- LLM replanner config (arm) ----------------------------------------------


def test_arm_planning_config_llm_replanner_default_none():
    from mousedroid.config.schema import ArmPlanningConfig

    cfg = ArmPlanningConfig()
    assert cfg.llm_replanner is None


def test_llm_replanner_config_defaults():
    from mousedroid.config.schema import LLMReplannerConfig

    cfg = LLMReplannerConfig()
    assert cfg.enabled is False
    assert cfg.backend == "null"
    assert cfg.max_tokens == 1024
    assert cfg.temperature == 0.0
    assert cfg.api_key_env_var == "ANTHROPIC_API_KEY"
    assert cfg.request_timeout_s == 30.0
    assert cfg.max_retries == 3


def test_llm_replanner_config_temperature_bounds():
    from mousedroid.config.schema import LLMReplannerConfig

    with pytest.raises(ValidationError):
        LLMReplannerConfig(temperature=-0.1)
    with pytest.raises(ValidationError):
        LLMReplannerConfig(temperature=2.5)


def test_llm_replanner_config_max_retries_nonneg():
    from mousedroid.config.schema import LLMReplannerConfig

    LLMReplannerConfig(max_retries=0)  # 0 retries permitted
    with pytest.raises(ValidationError):
        LLMReplannerConfig(max_retries=-1)


def test_arm_planning_config_with_llm_replanner_enabled():
    from mousedroid.config.schema import ArmPlanningConfig, LLMReplannerConfig

    cfg = ArmPlanningConfig(
        llm_replanner=LLMReplannerConfig(enabled=True, backend="anthropic"),
    )
    assert cfg.llm_replanner is not None
    assert cfg.llm_replanner.enabled is True
    assert cfg.llm_replanner.backend == "anthropic"
