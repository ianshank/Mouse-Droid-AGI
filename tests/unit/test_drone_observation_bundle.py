"""Tests for DroneObservationBundle — protocol conformance and properties."""

from __future__ import annotations

import numpy as np

from mousedroid.constants import N_DRONE_SENSOR_MODALITIES
from mousedroid.sensing.drone_bundle import DroneObservationBundle
from mousedroid.sensing.protocol import ObservationProtocol


def test_satisfies_observation_protocol():
    """DroneObservationBundle is recognised as ObservationProtocol."""
    bundle = DroneObservationBundle()
    assert isinstance(bundle, ObservationProtocol)


def test_n_modalities():
    bundle = DroneObservationBundle()
    assert bundle.n_modalities == N_DRONE_SENSOR_MODALITIES
    assert bundle.n_modalities == 7


def test_default_valid_mask_shape():
    bundle = DroneObservationBundle()
    assert bundle.valid_mask.shape == (N_DRONE_SENSOR_MODALITIES,)


def test_default_motor_state_shape():
    bundle = DroneObservationBundle()
    assert bundle.motor_state.shape == (7,)


def test_timestamp_is_positive():
    bundle = DroneObservationBundle()
    assert bundle.timestamp > 0


def test_default_vision_features():
    bundle = DroneObservationBundle()
    assert bundle.vision_features.dtype == np.float32
    assert bundle.vision_features.shape == (256,)


def test_default_distance():
    bundle = DroneObservationBundle()
    assert bundle.distance_m == 4.0


def test_default_audio_chunk():
    bundle = DroneObservationBundle()
    assert bundle.audio_chunk.dtype == np.float32


def test_altitude_property():
    bundle = DroneObservationBundle(_altitude_m=15.5)
    assert bundle.altitude_m == 15.5


def test_gps_position_property():
    pos = (37.7749, -122.4194, 50.0)
    bundle = DroneObservationBundle(_gps_position=pos)
    assert bundle.gps_position == pos


def test_imu_data_property():
    imu = np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03], dtype=np.float32)
    bundle = DroneObservationBundle(_imu_data=imu)
    np.testing.assert_array_equal(bundle.imu_data, imu)


def test_gps_fix_property():
    bundle = DroneObservationBundle(_gps_fix=True)
    assert bundle.gps_fix is True
    bundle_no_fix = DroneObservationBundle(_gps_fix=False)
    assert bundle_no_fix.gps_fix is False


def test_imu_healthy_property():
    bundle = DroneObservationBundle(_imu_healthy=True)
    assert bundle.imu_healthy is True
    bundle_bad = DroneObservationBundle(_imu_healthy=False)
    assert bundle_bad.imu_healthy is False


def test_armed_property():
    bundle = DroneObservationBundle(_armed=True)
    assert bundle.armed is True
    bundle_disarmed = DroneObservationBundle(_armed=False)
    assert bundle_disarmed.armed is False


def test_custom_valid_mask():
    mask = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    bundle = DroneObservationBundle(_valid_mask=mask)
    np.testing.assert_array_equal(bundle.valid_mask, mask)
