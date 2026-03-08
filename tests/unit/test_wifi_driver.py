"""Tests for WiFiESP32Driver."""

from __future__ import annotations

from mousedroid.comms.wifi_driver import WiFiESP32Driver, _clamp
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
