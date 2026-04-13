from __future__ import annotations

import time

import numpy as np

from mousedroid.sensing.bundle import MouseDroidObservationBundle


def test_default_values():
    obs = MouseDroidObservationBundle()
    assert obs.distance_m == 4.0
    assert obs.n_modalities == 4


def test_timestamp_is_monotonic():
    before = time.monotonic()
    obs = MouseDroidObservationBundle()
    after = time.monotonic()
    assert before <= obs.timestamp <= after


def test_vision_features_shape():
    obs = MouseDroidObservationBundle()
    assert obs.vision_features.shape == (256,)


def test_vision_features_dtype():
    obs = MouseDroidObservationBundle()
    assert obs.vision_features.dtype == np.float32


def test_motor_state_shape():
    obs = MouseDroidObservationBundle()
    assert obs.motor_state.shape == (4,)


def test_valid_mask_shape():
    obs = MouseDroidObservationBundle()
    assert obs.valid_mask.shape == (4,)


def test_valid_mask_defaults_to_ones():
    obs = MouseDroidObservationBundle()
    np.testing.assert_array_equal(obs.valid_mask, np.ones(4, dtype=np.float32))


def test_custom_values():
    vision = np.ones(128, dtype=np.float32)
    motor = np.array([0.1, 0.2, 0.3, 12.0], dtype=np.float32)
    mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    obs = MouseDroidObservationBundle(
        _vision_features=vision,
        _distance_m=1.5,
        _motor_state=motor,
        _valid_mask=mask,
    )
    assert obs.distance_m == 1.5
    np.testing.assert_array_equal(obs.vision_features, vision)
    np.testing.assert_array_equal(obs.motor_state, motor)


def test_n_modalities_equals_4():
    obs = MouseDroidObservationBundle()
    assert obs.n_modalities == 4


def test_audio_chunk_shape():
    obs = MouseDroidObservationBundle()
    assert obs.audio_chunk.shape == (1024,)


def test_audio_chunk_dtype():
    obs = MouseDroidObservationBundle()
    assert obs.audio_chunk.dtype == np.float32


def test_audio_chunk_custom():
    audio = np.ones(512, dtype=np.float32) * 0.5
    obs = MouseDroidObservationBundle(_audio_chunk=audio)
    np.testing.assert_array_equal(obs.audio_chunk, audio)


# ---------------------------------------------------------------------------
# LiDAR fields
# ---------------------------------------------------------------------------


def test_lidar_features_default_none():
    """Default bundle has lidar_features is None."""
    obs = MouseDroidObservationBundle()
    assert obs.lidar_features is None


def test_lidar_features_custom():
    """Construct with explicit lidar_features, verify property returns it."""
    lidar = np.ones(36, dtype=np.float32)
    obs = MouseDroidObservationBundle(_lidar_features=lidar)
    assert obs.lidar_features is not None
    np.testing.assert_array_equal(obs.lidar_features, lidar)


def test_n_modalities_with_5_element_mask():
    """Construct with 5-element valid_mask, verify n_modalities == 5."""
    mask = np.ones(5, dtype=np.float32)
    obs = MouseDroidObservationBundle(_valid_mask=mask)
    assert obs.n_modalities == 5


def test_backwards_compat_4_modalities():
    """Default bundle still returns n_modalities == 4."""
    obs = MouseDroidObservationBundle()
    assert obs.n_modalities == 4
