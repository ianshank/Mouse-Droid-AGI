"""Tests for WiFiESP32Driver — full coverage."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from mousedroid.comms._utils import clamp as _clamp
from mousedroid.comms.wifi_driver import WiFiESP32Driver
from mousedroid.config.schema import ESP32Config


def _make_driver() -> WiFiESP32Driver:
    cfg = ESP32Config(protocol="wifi")
    return WiFiESP32Driver(cfg)


def test_constructor():
    driver = _make_driver()
    assert driver._connected is False
    assert driver._base_url == "http://192.168.4.1:80"


def test_constructor_custom_host_port():
    cfg = ESP32Config(protocol="wifi", wifi_host="10.0.0.1", wifi_port=8080)
    driver = WiFiESP32Driver(cfg)
    assert driver._base_url == "http://10.0.0.1:8080"


def test_initial_velocity_zero():
    driver = _make_driver()
    assert driver._last_velocity == (0.0, 0.0, 0.0)


async def test_connect_sets_connected():
    driver = _make_driver()
    await driver.connect()
    assert driver._connected is True


async def test_disconnect_clears_connected():
    driver = _make_driver()
    await driver.connect()
    await driver.disconnect()
    assert driver._connected is False


def test_clamp():
    assert _clamp(0.5, -1.0, 1.0) == 0.5
    assert _clamp(-2.0, -1.0, 1.0) == -1.0
    assert _clamp(3.0, -1.0, 1.0) == 1.0


# -- Async method tests with mocked HTTP --


async def test_send_velocity():
    driver = _make_driver()
    with patch.object(driver, "_post_json", new_callable=AsyncMock, return_value={}):
        await driver.send_velocity(0.3, -0.1, 0.5)
    assert driver._last_velocity == (0.3, -0.1, 0.5)


async def test_read_encoders():
    driver = _make_driver()
    mock_data = {"lv": 0.5, "rv": 0.3, "ox": 1.0, "oy": 2.0, "h": 0.1, "ts": 100.0}
    with patch.object(driver, "_get_json", new_callable=AsyncMock, return_value=mock_data):
        reading = await driver.read_encoders()
    assert reading.left_velocity_mps == 0.5
    assert reading.right_velocity_mps == 0.3
    assert reading.odometry_x_m == 1.0


async def test_read_encoders_empty_response():
    driver = _make_driver()
    with patch.object(driver, "_get_json", new_callable=AsyncMock, return_value={}):
        reading = await driver.read_encoders()
    assert reading.left_velocity_mps == 0.0


async def test_get_battery_voltage():
    driver = _make_driver()
    with patch.object(driver, "_get_json", new_callable=AsyncMock, return_value={"v": 11.8}):
        voltage = await driver.get_battery_voltage()
    assert voltage == 11.8


async def test_get_battery_voltage_empty():
    driver = _make_driver()
    with patch.object(driver, "_get_json", new_callable=AsyncMock, return_value={}):
        voltage = await driver.get_battery_voltage()
    assert voltage == 0.0


async def test_emergency_stop():
    driver = _make_driver()
    with patch.object(driver, "_post_json", new_callable=AsyncMock, return_value={}):
        await driver.emergency_stop()
    assert driver._last_velocity == (0.0, 0.0, 0.0)


# -- JSON-shape guard tests --


def test_decode_json_object_returns_dict():
    driver = _make_driver()
    assert driver._decode_json_object('{"v": 11.8}', path="/bat") == {"v": 11.8}


def test_decode_json_object_empty_body_returns_empty_dict():
    driver = _make_driver()
    assert driver._decode_json_object("   ", path="/bat") == {}


def test_decode_json_object_non_object_returns_empty_dict():
    # A bare list / number / string is not a mapping; the guard must degrade to
    # ``{}`` instead of returning a non-dict to mapping-expecting callers.
    driver = _make_driver()
    assert driver._decode_json_object("[1, 2, 3]", path="/enc") == {}
    assert driver._decode_json_object("42", path="/enc") == {}
    assert driver._decode_json_object('"oops"', path="/enc") == {}


def test_decode_json_object_malformed_json_returns_empty_dict():
    # Truncated frame / HTTP error page / UART noise: a JSONDecodeError must NOT
    # escape the to_thread wrapper — it degrades to ``{}`` with a structured warn.
    driver = _make_driver()
    assert driver._decode_json_object("{not json", path="/enc") == {}
    assert driver._decode_json_object("<html>500</html>", path="/bat") == {}
    assert driver._decode_json_object('{"v": 1', path="/bat") == {}  # truncated


def test_decode_json_object_malformed_json_logs_warning():
    # The degraded path emits a structured wifi_esp32_non_json_response event so
    # the operator can grep for it (mirrors the serial driver contract).
    driver = _make_driver()
    with patch("mousedroid.comms.wifi_driver._log") as mock_log:
        driver._decode_json_object("garbage", path="/enc")
    mock_log.warning.assert_called_once()
    assert mock_log.warning.call_args.args[0] == "wifi_esp32_non_json_response"


def test_decode_json_object_non_dict_logs_warning():
    driver = _make_driver()
    with patch("mousedroid.comms.wifi_driver._log") as mock_log:
        driver._decode_json_object("[1, 2]", path="/enc")
    mock_log.warning.assert_called_once()
    assert mock_log.warning.call_args.args[0] == "wifi_esp32_unexpected_json_shape"
