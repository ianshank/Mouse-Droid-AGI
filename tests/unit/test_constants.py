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
    INTENTION_PREDICTOR_SEED,
    LIDAR_HEADER_BYTE,
    MILLISECONDS_PER_SECOND,
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
