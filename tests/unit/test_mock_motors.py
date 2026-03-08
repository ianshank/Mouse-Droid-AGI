"""Tests for MockMotorDriver."""

from __future__ import annotations

from mousedroid.hardware.motors.mock_motors import MockMotorDriver


def test_constructor():
    driver = MockMotorDriver()
    assert driver._last_velocity == (0.0, 0.0, 0.0)


async def test_send_velocity():
    driver = MockMotorDriver()
    await driver.send_velocity(0.1, 0.2, 0.3)
    assert driver._last_velocity == (0.1, 0.2, 0.3)


async def test_send_velocity_overwrites():
    driver = MockMotorDriver()
    await driver.send_velocity(0.1, 0.2, 0.3)
    await driver.send_velocity(0.4, 0.5, 0.6)
    assert driver._last_velocity == (0.4, 0.5, 0.6)


async def test_emergency_stop():
    driver = MockMotorDriver()
    await driver.send_velocity(1.0, 1.0, 1.0)
    await driver.emergency_stop()
    assert driver._last_velocity == (0.0, 0.0, 0.0)
