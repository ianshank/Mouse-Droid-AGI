"""Tests for SensorManager degraded motor polling."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np

from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.schema import Settings
from mousedroid.sensing.manager import SensorManager


def _make_manager(
    *,
    max_consecutive_timeouts: int = 5,
    degraded_poll_interval_s: float = 1.0,
) -> tuple[SensorManager, AsyncMock]:
    """Create a SensorManager with configurable ESP32 degraded settings."""
    cfg = Settings(
        mock_hardware=True,
        esp32={
            "max_consecutive_timeouts": max_consecutive_timeouts,
            "degraded_poll_interval_s": degraded_poll_interval_s,
        },
    )

    vision = AsyncMock()
    vision.capture_features = AsyncMock(
        return_value=np.ones(cfg.camera.feature_dim, dtype=np.float32),
    )
    vision.start = AsyncMock()
    vision.stop = AsyncMock()

    distance = AsyncMock()
    distance.read_distance_m = AsyncMock(return_value=1.5)
    distance.max_range_m = 4.0

    esp32 = AsyncMock()
    esp32.read_encoders = AsyncMock(return_value=EncoderReading())
    esp32.get_battery_voltage = AsyncMock(return_value=12.0)

    mgr = SensorManager(vision, distance, esp32, cfg)
    return mgr, esp32


# -- Initial state --


def test_motor_degraded_state_initially_off():
    mgr, _ = _make_manager()
    assert mgr._motor_degraded is False
    assert mgr._motor_consecutive_failures == 0


def test_motor_max_failures_from_config():
    mgr, _ = _make_manager(max_consecutive_timeouts=7)
    assert mgr._motor_max_failures == 7


def test_motor_degraded_interval_from_config():
    mgr, _ = _make_manager(degraded_poll_interval_s=2.5)
    assert mgr._motor_degraded_interval == 2.5


async def test_distance_none_uses_configured_fallback():
    cfg = Settings(
        mock_hardware=True,
        safety={"distance_fallback_m": 123.0},
    )

    vision = AsyncMock()
    vision.capture_features = AsyncMock(
        return_value=np.ones(cfg.camera.feature_dim, dtype=np.float32),
    )
    vision.start = AsyncMock()
    vision.stop = AsyncMock()

    esp32 = AsyncMock()
    esp32.read_encoders = AsyncMock(return_value=EncoderReading())
    esp32.get_battery_voltage = AsyncMock(return_value=12.0)

    mgr = SensorManager(vision, None, esp32, cfg)

    bundle = await mgr.read_all()

    assert bundle.distance_m == 123.0
    assert bundle.valid_mask[1] == 0.0


# -- Failure counting + degraded entry --


async def test_motor_failures_increment_on_read_failure():
    mgr, esp32 = _make_manager(max_consecutive_timeouts=10)
    esp32.read_encoders.side_effect = RuntimeError("serial timeout")

    await mgr.read_all()
    assert mgr._motor_consecutive_failures == 1
    assert mgr._motor_degraded is False


async def test_motor_enters_degraded_at_threshold():
    mgr, esp32 = _make_manager(max_consecutive_timeouts=3)
    esp32.read_encoders.side_effect = RuntimeError("serial timeout")

    for _ in range(3):
        await mgr.read_all()

    assert mgr._motor_degraded is True
    assert mgr._motor_consecutive_failures == 3


# -- Cached motor reads in degraded mode --


async def test_degraded_returns_cached_motor_state():
    mgr, esp32 = _make_manager(
        max_consecutive_timeouts=1,
        degraded_poll_interval_s=60.0,  # very long so we never probe
    )

    # First read succeeds — caches motor state
    esp32.read_encoders.return_value = EncoderReading(
        left_velocity_mps=0.1,
        right_velocity_mps=0.2,
        heading_rad=0.3,
    )
    esp32.get_battery_voltage.return_value = 11.5
    bundle1 = await mgr.read_all()
    assert bundle1.valid_mask[2] == 1.0  # motor valid

    # Second read fails — enters degraded
    esp32.read_encoders.side_effect = RuntimeError("timeout")
    await mgr.read_all()
    assert mgr._motor_degraded is True

    # Third read — returns cached value, doesn't call esp32
    esp32.read_encoders.reset_mock()
    esp32.get_battery_voltage.reset_mock()
    bundle3 = await mgr.read_all()
    # Motor slot is False (stale data) but has cached values
    assert bundle3.valid_mask[2] == 0.0
    esp32.read_encoders.assert_not_called()
    esp32.get_battery_voltage.assert_not_called()

    # Cached motor state should match last successful read
    np.testing.assert_allclose(
        mgr._cached_motor_state,
        [0.1, 0.2, 0.3, 11.5],
        atol=1e-6,
    )


# -- Recovery from degraded --


async def test_motor_recovers_from_degraded():
    mgr, esp32 = _make_manager(
        max_consecutive_timeouts=1,
        degraded_poll_interval_s=0.001,  # nearly instant probe
    )

    # Enter degraded
    esp32.read_encoders.side_effect = RuntimeError("timeout")
    await mgr.read_all()
    assert mgr._motor_degraded is True

    # Force probe interval to have elapsed by backdating the last probe time
    mgr._motor_last_probe = 0.0

    # Recover on next successful read
    esp32.read_encoders.side_effect = None
    esp32.read_encoders.return_value = EncoderReading(
        left_velocity_mps=0.5,
        right_velocity_mps=0.5,
    )
    esp32.get_battery_voltage.return_value = 12.0
    bundle = await mgr.read_all()

    assert mgr._motor_degraded is False
    assert mgr._motor_consecutive_failures == 0
    assert bundle.valid_mask[2] == 1.0


# -- Probe interval --


async def test_degraded_probes_after_interval():
    mgr, esp32 = _make_manager(
        max_consecutive_timeouts=1,
        degraded_poll_interval_s=0.01,
    )

    # Enter degraded
    esp32.read_encoders.side_effect = RuntimeError("timeout")
    await mgr.read_all()
    assert mgr._motor_degraded is True

    # Force probe interval to have elapsed.
    mgr._motor_last_probe = 0.0

    # Next read should attempt a real probe
    esp32.read_encoders.reset_mock()
    esp32.get_battery_voltage.reset_mock()
    esp32.read_encoders.side_effect = RuntimeError("still dead")
    await mgr.read_all()

    # Should have attempted the read (probe interval elapsed)
    esp32.read_encoders.assert_called_once()


# -- Success resets counter --


async def test_success_resets_failure_counter():
    mgr, esp32 = _make_manager(max_consecutive_timeouts=10)
    esp32.read_encoders.side_effect = RuntimeError("timeout")

    # Accumulate 5 failures
    for _ in range(5):
        await mgr.read_all()
    assert mgr._motor_consecutive_failures == 5

    # One success resets
    esp32.read_encoders.side_effect = None
    esp32.read_encoders.return_value = EncoderReading()
    esp32.get_battery_voltage.return_value = 12.0
    await mgr.read_all()

    assert mgr._motor_consecutive_failures == 0
    assert mgr._motor_degraded is False


# -- Backwards compatibility --


async def test_normal_operation_unaffected():
    """When ESP32 is healthy, behavior is identical to pre-adaptive code."""
    mgr, esp32 = _make_manager()
    esp32.read_encoders.return_value = EncoderReading(
        left_velocity_mps=0.3,
        right_velocity_mps=0.4,
        heading_rad=1.0,
    )
    esp32.get_battery_voltage.return_value = 11.8

    bundle = await mgr.read_all()

    assert bundle.valid_mask[2] == 1.0
    assert mgr._motor_degraded is False
    assert mgr._motor_consecutive_failures == 0
    np.testing.assert_allclose(bundle.motor_state, [0.3, 0.4, 1.0, 11.8], atol=1e-6)
