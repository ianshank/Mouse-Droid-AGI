from __future__ import annotations

import pytest

from mousedroid.comms.mock_driver import MockESP32Driver
from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.schema import ESP32Config


@pytest.fixture
def driver() -> MockESP32Driver:
    return MockESP32Driver(ESP32Config())


def test_construct(driver: MockESP32Driver):
    assert driver._connected is False


async def test_connect(driver: MockESP32Driver):
    await driver.connect()
    assert driver._connected is True


async def test_disconnect(driver: MockESP32Driver):
    await driver.connect()
    await driver.disconnect()
    assert driver._connected is False


async def test_send_velocity_stores_values(driver: MockESP32Driver):
    await driver.send_velocity(0.1, 0.2, 0.3)
    assert driver._last_velocity == (0.1, 0.2, 0.3)


async def test_send_velocity_overwrites(driver: MockESP32Driver):
    await driver.send_velocity(0.1, 0.2, 0.3)
    await driver.send_velocity(0.4, 0.5, 0.6)
    assert driver._last_velocity == (0.4, 0.5, 0.6)


async def test_read_encoders_returns_encoder_reading(driver: MockESP32Driver):
    reading = await driver.read_encoders()
    assert isinstance(reading, EncoderReading)
    assert reading.left_velocity_mps == 0.0


async def test_get_battery_voltage_default(driver: MockESP32Driver):
    v = await driver.get_battery_voltage()
    assert v == 12.0


async def test_get_battery_voltage_custom(driver: MockESP32Driver):
    driver._battery_voltage = 10.0
    v = await driver.get_battery_voltage()
    assert v == 10.0


async def test_emergency_stop_zeros_velocity(driver: MockESP32Driver):
    await driver.send_velocity(1.0, 1.0, 1.0)
    await driver.emergency_stop()
    assert driver._last_velocity == (0.0, 0.0, 0.0)


def test_initial_velocity_is_zero(driver: MockESP32Driver):
    assert driver._last_velocity == (0.0, 0.0, 0.0)
