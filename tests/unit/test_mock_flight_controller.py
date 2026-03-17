"""Tests for MockFlightController."""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.comms.flight_protocol import FlightControllerProtocol
from mousedroid.comms.mock_flight_controller import MockFlightController
from mousedroid.config.schema import FlightControllerConfig


@pytest.fixture
def fc() -> MockFlightController:
    """Create a MockFlightController with default config."""
    return MockFlightController(FlightControllerConfig())


def test_satisfies_protocol(fc: MockFlightController):
    """MockFlightController is recognised as FlightControllerProtocol."""
    assert isinstance(fc, FlightControllerProtocol)


async def test_connect_disconnect(fc: MockFlightController):
    await fc.connect()
    assert fc._connected is True
    await fc.disconnect()
    assert fc._connected is False
    assert fc._armed is False


async def test_arm_disarm(fc: MockFlightController):
    await fc.arm()
    assert fc.armed is True
    await fc.disarm()
    assert fc.armed is False


async def test_takeoff_sets_altitude(fc: MockFlightController):
    await fc.arm()
    await fc.takeoff(15.0)
    alt = await fc.get_altitude_m()
    assert alt == pytest.approx(15.0)
    assert fc.flight_mode == "GUIDED"


async def test_land_resets_altitude(fc: MockFlightController):
    await fc.arm()
    await fc.takeoff(10.0)
    await fc.land()
    alt = await fc.get_altitude_m()
    assert alt == pytest.approx(0.0)
    assert fc.flight_mode == "LAND"


async def test_send_velocity_ned(fc: MockFlightController):
    await fc.send_velocity_ned(1.0, 0.5, -0.3, 0.2)
    assert fc._last_velocity == (1.0, 0.5, -0.3, 0.2)


async def test_get_gps_position(fc: MockFlightController):
    pos = await fc.get_gps_position()
    assert len(pos) == 3
    assert all(v == pytest.approx(0.0) for v in pos)


async def test_get_imu_data(fc: MockFlightController):
    imu = await fc.get_imu_data()
    assert imu.shape == (6,)
    assert imu.dtype == np.float32
    np.testing.assert_array_equal(imu, np.zeros(6))


async def test_get_battery_voltage(fc: MockFlightController):
    voltage = await fc.get_battery_voltage()
    assert voltage == pytest.approx(16.8)


async def test_set_flight_mode(fc: MockFlightController):
    await fc.set_flight_mode("LOITER")
    assert fc.flight_mode == "LOITER"


async def test_return_to_launch(fc: MockFlightController):
    await fc.return_to_launch()
    assert fc.flight_mode == "RTL"


async def test_emergency_stop(fc: MockFlightController):
    await fc.arm()
    assert fc.armed is True
    await fc.emergency_stop()
    assert fc.armed is False
    assert fc._last_velocity == (0.0, 0.0, 0.0, 0.0)
