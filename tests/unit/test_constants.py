"""Tests for shared constants — ensure single source of truth is importable."""

from __future__ import annotations

from mousedroid.constants import (
    AFFECT_ESTIMATOR_SEED,
    BELIEF_ENCODER_SEED,
    DEFAULT_ACTION_DIM,
    DEFAULT_AUDIO_CHUNK_SIZE,
    DEFAULT_LIDAR_FEATURE_DIM,
    DEFAULT_MAX_DISTANCE_M,
    DEFAULT_MOTOR_STATE_DIM,
    DEFAULT_VISION_DIM,
    DESIRE_ENCODER_SEED,
    HAILO_MOCK_FEATURE_EXTRACTOR_DIM,
    HAILO_MOCK_YOLO_OUTPUT_SHAPE,
    INTENTION_PREDICTOR_SEED,
    LIDAR_CRC8_POLYNOMIAL,
    LIDAR_DEFAULT_MOCK_CONFIDENCE,
    LIDAR_HEADER_BYTE,
    LIDAR_MM_PER_M,
    LIDAR_SCAN_TIMEOUT_MULTIPLIER,
    MILLISECONDS_PER_SECOND,
    MOCK_CAMERA_PROCEDURAL_HEIGHT,
    MOCK_CAMERA_PROCEDURAL_WIDTH,
    MOCK_ULTRASONIC_PIN_DEFAULT,
    N_SENSOR_MODALITIES,
    N_SENSOR_MODALITIES_WITH_LIDAR,
    POLICY_MLP_SEED,
    VALUE_MLP_SEED,
    WEIGHT_INIT_SCALE,
)


def test_dimension_constants_positive():
    assert DEFAULT_VISION_DIM > 0
    assert DEFAULT_MOTOR_STATE_DIM > 0
    assert DEFAULT_ACTION_DIM > 0
    assert DEFAULT_AUDIO_CHUNK_SIZE > 0
    assert N_SENSOR_MODALITIES > 0


def test_distance_constant_positive():
    assert DEFAULT_MAX_DISTANCE_M > 0.0


def test_milliseconds_conversion():
    assert MILLISECONDS_PER_SECOND == 1000.0


def test_weight_init_scale_small():
    assert 0 < WEIGHT_INIT_SCALE < 1.0


def test_seeds_are_unique():
    seeds = [
        BELIEF_ENCODER_SEED,
        DESIRE_ENCODER_SEED,
        INTENTION_PREDICTOR_SEED,
        AFFECT_ESTIMATOR_SEED,
        POLICY_MLP_SEED,
        VALUE_MLP_SEED,
    ]
    assert len(seeds) == len(set(seeds)), "All RNG seeds must be unique"


def test_seeds_are_non_negative():
    for seed in [
        BELIEF_ENCODER_SEED,
        DESIRE_ENCODER_SEED,
        INTENTION_PREDICTOR_SEED,
        AFFECT_ESTIMATOR_SEED,
        POLICY_MLP_SEED,
        VALUE_MLP_SEED,
    ]:
        assert seed >= 0


# ---------------------------------------------------------------------------
# LiDAR constants
# ---------------------------------------------------------------------------


def test_n_sensor_modalities_with_lidar():
    """N_SENSOR_MODALITIES_WITH_LIDAR should be 5."""
    assert N_SENSOR_MODALITIES_WITH_LIDAR == 5


def test_lidar_header_byte():
    """LIDAR_HEADER_BYTE should be 0x54."""
    assert LIDAR_HEADER_BYTE == 0x54


def test_lidar_default_feature_dim():
    """DEFAULT_LIDAR_FEATURE_DIM should be 36."""
    assert DEFAULT_LIDAR_FEATURE_DIM == 36


def test_lidar_crc8_polynomial():
    """LIDAR_CRC8_POLYNOMIAL should match the LD19 vendor CRC table."""
    assert LIDAR_CRC8_POLYNOMIAL == 0x4D


def test_lidar_mm_per_m():
    """LIDAR_MM_PER_M should be 1000.0."""
    assert LIDAR_MM_PER_M == 1000.0


def test_lidar_scan_timeout_multiplier():
    """LIDAR_SCAN_TIMEOUT_MULTIPLIER should be 2.0."""
    assert LIDAR_SCAN_TIMEOUT_MULTIPLIER == 2.0


def test_lidar_default_mock_confidence():
    """LIDAR_DEFAULT_MOCK_CONFIDENCE should be 200."""
    assert LIDAR_DEFAULT_MOCK_CONFIDENCE == 200


# ---------------------------------------------------------------------------
# Hailo mock output shape constants
# ---------------------------------------------------------------------------


def test_hailo_mock_yolo_output_shape_is_2d():
    """HAILO_MOCK_YOLO_OUTPUT_SHAPE must be a 2-tuple of positive ints."""
    assert len(HAILO_MOCK_YOLO_OUTPUT_SHAPE) == 2
    assert all(v > 0 for v in HAILO_MOCK_YOLO_OUTPUT_SHAPE)


def test_hailo_mock_yolo_output_shape_matches_runtime_default():
    """HAILO_MOCK_YOLO_OUTPUT_SHAPE must match MockHailoRuntime.DEFAULT_OUTPUT_SHAPES."""
    from mousedroid.hardware.accelerator.hailo_runtime import MockHailoRuntime

    assert MockHailoRuntime.DEFAULT_OUTPUT_SHAPES["yolo"] == HAILO_MOCK_YOLO_OUTPUT_SHAPE


def test_hailo_mock_feature_extractor_dim_positive():
    """HAILO_MOCK_FEATURE_EXTRACTOR_DIM must be a positive integer."""
    assert HAILO_MOCK_FEATURE_EXTRACTOR_DIM > 0


def test_hailo_mock_feature_extractor_dim_matches_runtime_default():
    """HAILO_MOCK_FEATURE_EXTRACTOR_DIM must match MockHailoRuntime.DEFAULT_OUTPUT_SHAPES."""
    from mousedroid.hardware.accelerator.hailo_runtime import MockHailoRuntime

    assert MockHailoRuntime.DEFAULT_OUTPUT_SHAPES["feature_extractor"] == (
        HAILO_MOCK_FEATURE_EXTRACTOR_DIM,
    )


def test_hailo_mock_feature_extractor_dim_matches_vision_dim():
    """Mock feature extractor output must match the default vision feature dimension."""
    assert HAILO_MOCK_FEATURE_EXTRACTOR_DIM == DEFAULT_VISION_DIM


# ---------------------------------------------------------------------------
# Mock camera procedural-frame dimension constants
# ---------------------------------------------------------------------------


def test_mock_camera_procedural_dimensions_positive():
    """Mock camera procedural frame dimensions must be positive."""
    assert MOCK_CAMERA_PROCEDURAL_WIDTH > 0
    assert MOCK_CAMERA_PROCEDURAL_HEIGHT > 0


def test_mock_camera_procedural_dimensions_match_class():
    """MockCamera must use the constants for its internal raw dimensions."""
    from mousedroid.config.schema import CameraConfig
    from mousedroid.hardware.camera.mock_camera import MockCamera

    cfg = CameraConfig()
    cam = MockCamera(cfg)
    assert cam._raw_width == MOCK_CAMERA_PROCEDURAL_WIDTH
    assert cam._raw_height == MOCK_CAMERA_PROCEDURAL_HEIGHT


# ---------------------------------------------------------------------------
# Mock sensor pin default constant
# ---------------------------------------------------------------------------


def test_mock_ultrasonic_pin_default_is_zero():
    """MOCK_ULTRASONIC_PIN_DEFAULT is the sentinel value 0 (no real GPIO)."""
    assert MOCK_ULTRASONIC_PIN_DEFAULT == 0


def test_mock_ultrasonic_pin_default_used_in_factory() -> None:
    """factory.build_distance_sensor uses MOCK_ULTRASONIC_PIN_DEFAULT for pins."""
    from mousedroid.config.schema import Settings

    cfg = Settings.model_validate({"mock_hardware": True})
    # In mock mode the factory must succeed even without an ultrasonic block.
    from mousedroid.factory import build_distance_sensor

    sensor = build_distance_sensor(cfg)
    assert sensor is not None
    # Verify the constant propagated all the way into the sensor's stored config.
    assert sensor._cfg.trigger_pin == MOCK_ULTRASONIC_PIN_DEFAULT
    assert sensor._cfg.echo_pin == MOCK_ULTRASONIC_PIN_DEFAULT
