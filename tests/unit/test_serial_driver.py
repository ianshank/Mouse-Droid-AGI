"""Tests for SerialESP32Driver — full coverage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mousedroid.comms._utils import clamp as _clamp
from mousedroid.comms.serial_driver import SerialESP32Driver
from mousedroid.config.schema import ESP32Config


def _make_driver() -> SerialESP32Driver:
    cfg = ESP32Config()
    return SerialESP32Driver(cfg)


def test_constructor():
    driver = _make_driver()
    assert driver._connected is False
    assert driver._port == "/dev/ttyUSB0"
    assert driver._baud == 1_000_000


def test_constructor_custom_config():
    cfg = ESP32Config(serial_port="/dev/ttyS1", serial_baud=115200)
    driver = SerialESP32Driver(cfg)
    assert driver._port == "/dev/ttyS1"
    assert driver._baud == 115200


def test_initial_velocity_zero():
    driver = _make_driver()
    assert driver._last_velocity == (0.0, 0.0, 0.0)


def test_clamp_within_bounds():
    assert _clamp(0.5, 0.0, 1.0) == 0.5


def test_clamp_below():
    assert _clamp(-1.0, 0.0, 1.0) == 0.0


def test_clamp_above():
    assert _clamp(2.0, 0.0, 1.0) == 1.0


# -- Async method tests with mocked serial --


async def test_connect_sets_connected():
    driver = _make_driver()
    mock_serial = MagicMock()
    with patch("mousedroid.comms.serial_driver._serial_mod", mock_serial):
        await driver.connect()
    assert driver._connected is True


async def test_connect_raises_without_pyserial():
    driver = _make_driver()
    with (
        patch("mousedroid.comms.serial_driver._serial_mod", None),
        pytest.raises(RuntimeError, match="pyserial is not installed"),
    ):
        await driver.connect()


async def test_disconnect_clears_connected():
    driver = _make_driver()
    driver._connected = True
    driver._serial = MagicMock()
    await driver.disconnect()
    assert driver._connected is False


async def test_disconnect_when_no_serial():
    driver = _make_driver()
    driver._serial = None
    await driver.disconnect()
    assert driver._connected is False


async def test_send_velocity():
    driver = _make_driver()
    driver._connected = True
    with patch.object(driver, "_send_json", new_callable=AsyncMock) as mock_send:
        await driver.send_velocity(0.2, 0.0, 0.0)
    mock_send.assert_awaited_once()
    assert driver._last_velocity == (0.2, 0.0, 0.0)


async def test_read_encoders():
    driver = _make_driver()
    mock_data = {"lv": 0.5, "rv": 0.3, "ox": 1.0, "oy": 2.0, "h": 0.1, "ts": 100.0}
    with patch.object(driver, "_read_json", new_callable=AsyncMock, return_value=mock_data):
        reading = await driver.read_encoders()
    assert reading.left_velocity_mps == 0.5
    assert reading.right_velocity_mps == 0.3
    assert reading.odometry_x_m == 1.0


async def test_read_encoders_empty_response():
    driver = _make_driver()
    with patch.object(driver, "_read_json", new_callable=AsyncMock, return_value={}):
        reading = await driver.read_encoders()
    assert reading.left_velocity_mps == 0.0


async def test_get_battery_voltage():
    driver = _make_driver()
    with (
        patch.object(driver, "_send_json", new_callable=AsyncMock),
        patch.object(driver, "_read_json", new_callable=AsyncMock, return_value={"v": 12.3}),
    ):
        voltage = await driver.get_battery_voltage()
    assert voltage == 12.3


async def test_emergency_stop():
    driver = _make_driver()
    with patch.object(driver, "_send_json", new_callable=AsyncMock) as mock_send:
        await driver.emergency_stop()
    mock_send.assert_awaited_once()
    assert driver._last_velocity == (0.0, 0.0, 0.0)


async def test_read_json_empty_line():
    driver = _make_driver()
    with patch.object(driver, "_read_line", return_value=""):
        result = await driver._read_json()
    assert result == {}


async def test_read_json_returns_empty_on_non_json_line():
    """Stock-firmware boot banners must not crash the driver — return {}."""
    driver = _make_driver()
    with patch.object(driver, "_read_line", return_value="ESP32-WROOM boot banner OK"):
        result = await driver._read_json()
    assert result == {}


async def test_read_json_returns_empty_on_json_array():
    """Driver expects a dict; arrays/strings are tolerated as empty."""
    driver = _make_driver()
    with patch.object(driver, "_read_line", return_value="[1, 2, 3]"):
        result = await driver._read_json()
    assert result == {}


async def test_read_json_returns_parsed_dict_on_valid_response():
    driver = _make_driver()
    with patch.object(driver, "_read_line", return_value='{"lv": 0.1, "rv": 0.1}'):
        result = await driver._read_json()
    assert result == {"lv": 0.1, "rv": 0.1}
