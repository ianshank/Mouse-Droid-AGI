"""Tests for DroneSensorManager — concurrent reads and failure handling."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np

from mousedroid.config.schema import Settings
from mousedroid.sensing.drone_bundle import DroneObservationBundle
from mousedroid.sensing.drone_manager import DroneSensorManager


def _make_manager() -> tuple[DroneSensorManager, dict[str, AsyncMock]]:
    """Build a DroneSensorManager with fully-mocked sensors."""
    cfg = Settings(mock_hardware=True, platform="drone")

    vision = AsyncMock()
    vision.capture_features = AsyncMock(
        return_value=np.ones(cfg.camera.feature_dim, dtype=np.float32),
    )
    vision.start = AsyncMock()
    vision.stop = AsyncMock()

    distance = AsyncMock()
    distance.read_distance_m = AsyncMock(return_value=2.5)
    distance.max_range_m = 4.0

    motor_controller = AsyncMock()
    motor_controller.read_state = AsyncMock(
        return_value=np.array([0.1, 0.2, -0.1, 0.05, 10.0, 16.0, 1.0], dtype=np.float32),
    )
    motor_controller.platform_type = "drone"

    fc = AsyncMock()
    fc.armed = True
    fc.get_imu_data = AsyncMock(
        return_value=np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03], dtype=np.float32),
    )
    fc.get_gps_position = AsyncMock(return_value=(37.7749, -122.4194, 50.0))
    fc.get_altitude_m = AsyncMock(return_value=15.0)

    mgr = DroneSensorManager(vision, distance, motor_controller, fc, cfg)

    mocks = {
        "vision": vision,
        "distance": distance,
        "motor_controller": motor_controller,
        "fc": fc,
    }
    return mgr, mocks


def _make_manager_with_mic() -> tuple[DroneSensorManager, dict[str, AsyncMock]]:
    """Build a DroneSensorManager with a microphone mock."""
    cfg = Settings(mock_hardware=True, platform="drone")

    vision = AsyncMock()
    vision.capture_features = AsyncMock(
        return_value=np.ones(cfg.camera.feature_dim, dtype=np.float32),
    )
    vision.start = AsyncMock()
    vision.stop = AsyncMock()

    distance = AsyncMock()
    distance.read_distance_m = AsyncMock(return_value=2.5)
    distance.max_range_m = 4.0

    motor_controller = AsyncMock()
    motor_controller.read_state = AsyncMock(
        return_value=np.array([0.0] * 7, dtype=np.float32),
    )
    motor_controller.platform_type = "drone"

    fc = AsyncMock()
    fc.armed = False
    fc.get_imu_data = AsyncMock(
        return_value=np.zeros(6, dtype=np.float32),
    )
    fc.get_gps_position = AsyncMock(return_value=(0.0, 0.0, 0.0))
    fc.get_altitude_m = AsyncMock(return_value=0.0)

    mic = AsyncMock()
    mic.chunk_size = 1024
    mic.channels = 1
    mic.read_chunk = AsyncMock(
        return_value=np.ones(1024, dtype=np.float32),
    )
    mic.start = AsyncMock()
    mic.stop = AsyncMock()

    mgr = DroneSensorManager(vision, distance, motor_controller, fc, cfg, microphone=mic)

    mocks = {
        "vision": vision,
        "distance": distance,
        "motor_controller": motor_controller,
        "fc": fc,
        "mic": mic,
    }
    return mgr, mocks


class TestConstructor:
    def test_manager_init(self):
        mgr, _ = _make_manager()
        assert mgr._vision is not None
        assert mgr._fc is not None
        assert mgr._motor_controller is not None


class TestReadAll:
    async def test_returns_drone_bundle(self):
        mgr, _ = _make_manager()
        bundle = await mgr.read_all()
        assert isinstance(bundle, DroneObservationBundle)

    async def test_timestamp_positive(self):
        mgr, _ = _make_manager()
        bundle = await mgr.read_all()
        assert bundle.timestamp > 0

    async def test_distance_value(self):
        mgr, _ = _make_manager()
        bundle = await mgr.read_all()
        assert bundle.distance_m == 2.5

    async def test_altitude_value(self):
        mgr, _ = _make_manager()
        bundle = await mgr.read_all()
        assert bundle.altitude_m == 15.0

    async def test_gps_position(self):
        mgr, _ = _make_manager()
        bundle = await mgr.read_all()
        assert bundle.gps_position == (37.7749, -122.4194, 50.0)

    async def test_imu_data(self):
        mgr, _ = _make_manager()
        bundle = await mgr.read_all()
        expected = np.array([0.1, 0.2, 9.8, 0.01, 0.02, 0.03], dtype=np.float32)
        np.testing.assert_array_almost_equal(bundle.imu_data, expected)

    async def test_armed_state_propagated(self):
        mgr, _ = _make_manager()
        bundle = await mgr.read_all()
        assert bundle.armed is True

    async def test_all_valid_mask_entries(self):
        """6 of 7 sensor reads succeed (no mic) → audio slot is 0."""
        mgr, _ = _make_manager()
        bundle = await mgr.read_all()
        expected = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(bundle.valid_mask, expected)

    async def test_all_valid_with_mic(self):
        """All 7 sensors valid when microphone is present."""
        mgr, _ = _make_manager_with_mic()
        bundle = await mgr.read_all()
        expected = np.ones(7, dtype=np.float32)
        np.testing.assert_array_equal(bundle.valid_mask, expected)

    async def test_n_modalities(self):
        mgr, _ = _make_manager()
        bundle = await mgr.read_all()
        assert bundle.n_modalities == 7

    async def test_motor_state_shape(self):
        mgr, _ = _make_manager()
        bundle = await mgr.read_all()
        assert bundle.motor_state.shape == (7,)


class TestSensorFailures:
    async def test_vision_failure_zeros_mask(self):
        mgr, mocks = _make_manager()
        mocks["vision"].capture_features.side_effect = RuntimeError("camera fail")
        bundle = await mgr.read_all()
        assert bundle.valid_mask[0] == 0.0
        # Other sensors should still be valid
        assert bundle.valid_mask[1] == 1.0

    async def test_distance_failure_zeros_mask(self):
        mgr, mocks = _make_manager()
        mocks["distance"].read_distance_m.side_effect = RuntimeError("sensor fail")
        bundle = await mgr.read_all()
        assert bundle.valid_mask[1] == 0.0

    async def test_motor_failure_zeros_mask(self):
        mgr, mocks = _make_manager()
        mocks["motor_controller"].read_state.side_effect = RuntimeError("motor fail")
        bundle = await mgr.read_all()
        assert bundle.valid_mask[2] == 0.0
        # Motor state should be zero-filled
        assert bundle.motor_state.sum() == 0.0

    async def test_imu_failure_zeros_mask(self):
        mgr, mocks = _make_manager()
        mocks["fc"].get_imu_data.side_effect = RuntimeError("imu fail")
        bundle = await mgr.read_all()
        assert bundle.valid_mask[4] == 0.0
        assert bundle.imu_healthy is False

    async def test_gps_failure_zeros_mask(self):
        mgr, mocks = _make_manager()
        mocks["fc"].get_gps_position.side_effect = RuntimeError("gps fail")
        bundle = await mgr.read_all()
        assert bundle.valid_mask[5] == 0.0
        assert bundle.gps_fix is False
        assert bundle.gps_position == (0.0, 0.0, 0.0)

    async def test_altitude_failure_zeros_mask(self):
        mgr, mocks = _make_manager()
        mocks["fc"].get_altitude_m.side_effect = RuntimeError("alt fail")
        bundle = await mgr.read_all()
        assert bundle.valid_mask[6] == 0.0
        assert bundle.altitude_m == 0.0

    async def test_multiple_failures(self):
        """Multiple sensor failures reflected independently in valid_mask."""
        mgr, mocks = _make_manager()
        mocks["vision"].capture_features.side_effect = RuntimeError("fail")
        mocks["fc"].get_gps_position.side_effect = RuntimeError("fail")
        mocks["fc"].get_altitude_m.side_effect = RuntimeError("fail")
        bundle = await mgr.read_all()
        assert bundle.valid_mask[0] == 0.0  # vision
        assert bundle.valid_mask[1] == 1.0  # distance OK
        assert bundle.valid_mask[5] == 0.0  # gps
        assert bundle.valid_mask[6] == 0.0  # altitude


class TestAudioWithMicrophone:
    async def test_audio_valid_with_mic(self):
        mgr, mocks = _make_manager_with_mic()
        bundle = await mgr.read_all()
        assert bundle.valid_mask[3] == 1.0
        mocks["mic"].read_chunk.assert_awaited_once()

    async def test_audio_failure_zeros_mask(self):
        mgr, mocks = _make_manager_with_mic()
        mocks["mic"].read_chunk.side_effect = RuntimeError("mic fail")
        bundle = await mgr.read_all()
        assert bundle.valid_mask[3] == 0.0

    async def test_no_mic_audio_invalid(self):
        """Without microphone, audio slot in valid_mask is 0.0."""
        mgr, _ = _make_manager()
        bundle = await mgr.read_all()
        assert bundle.valid_mask[3] == 0.0


class TestLifecycle:
    async def test_start_calls_vision_start(self):
        mgr, mocks = _make_manager()
        await mgr.start()
        mocks["vision"].start.assert_awaited_once()

    async def test_stop_calls_vision_stop(self):
        mgr, mocks = _make_manager()
        await mgr.stop()
        mocks["vision"].stop.assert_awaited_once()

    async def test_start_calls_mic_start(self):
        mgr, mocks = _make_manager_with_mic()
        await mgr.start()
        mocks["mic"].start.assert_awaited_once()

    async def test_stop_calls_mic_stop(self):
        mgr, mocks = _make_manager_with_mic()
        await mgr.stop()
        mocks["mic"].stop.assert_awaited_once()
