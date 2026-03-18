from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import (
    AnnotationConfig,
    AudioAIConfig,
    BDITrainingConfig,
    CameraConfig,
    CircuitBreakerConfig,
    CuriosityConfig,
    ESP32Config,
    ExperienceConfig,
    FusionConfig,
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
    VisionAIConfig,
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
    with pytest.raises(ValidationError, match="ultrasonic config required"):
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


# -- VisionAIConfig new fields ------------------------------------------------


def test_vision_ai_config_detector_defaults():
    c = VisionAIConfig()
    assert c.detector_imgsz == 640
    assert c.detector_half_precision is True


def test_vision_ai_config_person_class_names_default():
    c = VisionAIConfig()
    assert c.person_class_names == ["person"]


def test_vision_ai_config_law2_gesture_labels_default():
    c = VisionAIConfig()
    assert c.law2_gesture_labels == ["stop"]


def test_vision_ai_config_gesture_confidence_defaults():
    c = VisionAIConfig()
    assert c.gesture_min_detection_confidence == pytest.approx(0.5)
    assert c.gesture_min_tracking_confidence == pytest.approx(0.5)


def test_vision_ai_config_person_class_names_custom():
    c = VisionAIConfig(person_class_names=["person", "rider"])
    assert "rider" in c.person_class_names


def test_vision_ai_config_detector_imgsz_rejects_zero():
    with pytest.raises(ValidationError):
        VisionAIConfig(detector_imgsz=0)


# -- AudioAIConfig new fields -------------------------------------------------


def test_audio_ai_config_sample_rates_default():
    c = AudioAIConfig()
    assert c.asr_sample_rate_hz == 16000
    assert c.classifier_sample_rate_hz == 16000


def test_audio_ai_config_window_ms_default():
    c = AudioAIConfig()
    assert c.classifier_window_ms == pytest.approx(975.0)


def test_audio_ai_config_stop_keywords_default():
    c = AudioAIConfig()
    assert "stop" in c.stop_keywords
    assert "halt" in c.stop_keywords


def test_audio_ai_config_asr_beam_size_default():
    c = AudioAIConfig()
    assert c.asr_beam_size == 1


def test_audio_ai_config_asr_accumulate_s_default():
    c = AudioAIConfig()
    assert c.asr_accumulate_s == pytest.approx(3.0)


def test_audio_ai_config_sample_rate_rejects_zero():
    with pytest.raises(ValidationError):
        AudioAIConfig(asr_sample_rate_hz=0)


# -- FusionConfig new fields --------------------------------------------------


def test_fusion_config_midas_hub_repo_default():
    c = FusionConfig()
    assert c.midas_hub_repo == "intel-isl/MiDaS"


def test_fusion_config_midas_hub_repo_custom():
    c = FusionConfig(midas_hub_repo="custom-org/MiDaS-fork")
    assert c.midas_hub_repo == "custom-org/MiDaS-fork"


# -- MCTSConfig new fields ----------------------------------------------------


def test_mcts_config_ucb_candidates_default():
    c = MCTSConfig()
    assert 1.41 in c.ucb_candidates
    assert len(c.ucb_candidates) == 5


def test_mcts_config_warmstart_n_episodes_default():
    c = MCTSConfig()
    assert c.warmstart_n_episodes == 100


def test_mcts_config_ucb_candidates_custom():
    c = MCTSConfig(ucb_candidates=[0.5, 1.0])
    assert c.ucb_candidates == [0.5, 1.0]


def test_mcts_config_warmstart_n_episodes_rejects_zero():
    with pytest.raises(ValidationError):
        MCTSConfig(warmstart_n_episodes=0)


# -- AnnotationConfig new fields ----------------------------------------------


def test_annotation_config_nominal_battery_default():
    c = AnnotationConfig()
    assert c.nominal_battery_v == pytest.approx(12.0)


def test_annotation_config_nominal_obstacle_default():
    c = AnnotationConfig()
    assert c.nominal_obstacle_dist_m == pytest.approx(2.0)


def test_annotation_config_nominal_battery_rejects_zero():
    with pytest.raises(ValidationError):
        AnnotationConfig(nominal_battery_v=0)


def test_annotation_config_nominal_obstacle_rejects_zero():
    with pytest.raises(ValidationError):
        AnnotationConfig(nominal_obstacle_dist_m=0)


# -- BDITrainingConfig new fields ---------------------------------------------


def test_bdi_training_config_dims_default():
    c = BDITrainingConfig()
    assert c.obs_dim == 256
    assert c.belief_dim == 128
    assert c.desire_dim == 64
    assert c.affect_dim == 2


def test_bdi_training_config_obs_dim_rejects_zero():
    with pytest.raises(ValidationError):
        BDITrainingConfig(obs_dim=0)


def test_bdi_training_config_dims_custom():
    c = BDITrainingConfig(obs_dim=128, belief_dim=64, desire_dim=32, affect_dim=4)
    assert c.obs_dim == 128
    assert c.belief_dim == 64
    assert c.desire_dim == 32
    assert c.affect_dim == 4


# -- Settings wires AI sub-configs correctly ----------------------------------


def test_settings_vision_ai_defaults():
    s = Settings(mock_hardware=True)
    assert s.vision_ai.detector_imgsz == 640
    assert s.vision_ai.person_class_names == ["person"]


def test_settings_audio_ai_defaults():
    s = Settings(mock_hardware=True)
    assert s.audio_ai.asr_sample_rate_hz == 16000
    assert "stop" in s.audio_ai.stop_keywords


def test_settings_fusion_defaults():
    s = Settings(mock_hardware=True)
    assert s.fusion.midas_hub_repo == "intel-isl/MiDaS"


def test_settings_mcts_ucb_candidates():
    s = Settings(mock_hardware=True)
    assert len(s.mcts.ucb_candidates) > 0


def test_settings_annotation_nominal_values():
    s = Settings(mock_hardware=True)
    assert s.training.annotation.nominal_battery_v == pytest.approx(12.0)
    assert s.training.annotation.nominal_obstacle_dist_m == pytest.approx(2.0)
