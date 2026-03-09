"""Tests for BaseESP32Driver — shared high-level driver logic.

Uses a minimal in-memory ``StubESP32Driver`` to exercise the inherited
``send_velocity``, ``read_encoders``, ``get_battery_voltage``, and
``emergency_stop`` methods without real hardware or mock patches.
"""

from __future__ import annotations

from typing import Any

import pytest

from mousedroid.comms._utils import (
    ESP32_CMD_TYPE_BATTERY,
    ESP32_CMD_TYPE_STOP,
    ESP32_CMD_TYPE_VELOCITY,
    MAX_PWM,
)
from mousedroid.comms.base_driver import BaseESP32Driver
from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.schema import ESP32Config


# ---------------------------------------------------------------------------
# Minimal stub subclass — records calls, returns canned responses
# ---------------------------------------------------------------------------


class StubESP32Driver(BaseESP32Driver):
    """Concrete stub for testing ``BaseESP32Driver`` shared logic."""

    def __init__(self, cfg: ESP32Config) -> None:
        super().__init__(cfg)
        self.commands_sent: list[dict[str, int]] = []
        self.query_responses: dict[str, dict[str, Any]] = {}
        self.connect_called = False
        self.disconnect_called = False

    async def connect(self) -> None:
        self._connected = True
        self.connect_called = True

    async def disconnect(self) -> None:
        self._connected = False
        self.disconnect_called = True

    async def _send_command(self, cmd: dict[str, int]) -> None:
        self.commands_sent.append(cmd)

    async def _query_data(
        self,
        resource: str,
        cmd: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        return self.query_responses.get(resource, {})


def _make_stub(
    max_vel: float = 1.0,
    max_omega: float = 1.0,
) -> StubESP32Driver:
    cfg = ESP32Config(max_velocity_mps=max_vel, max_omega_rads=max_omega)
    return StubESP32Driver(cfg)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_initial_state():
    stub = _make_stub()
    assert stub._connected is False
    assert stub._last_velocity == (0.0, 0.0, 0.0)


async def test_connect_sets_connected():
    stub = _make_stub()
    await stub.connect()
    assert stub._connected is True


# ---------------------------------------------------------------------------
# send_velocity
# ---------------------------------------------------------------------------


async def test_send_velocity_dispatches_correct_command():
    stub = _make_stub(max_vel=1.0, max_omega=1.0)
    await stub.send_velocity(0.5, -0.5, 1.0)

    assert len(stub.commands_sent) == 1
    cmd = stub.commands_sent[0]
    assert cmd["T"] == ESP32_CMD_TYPE_VELOCITY
    # 0.5 / 1.0 = 0.5 → int(0.5 * 255) = 127 (truncation in build_velocity_cmd)
    assert cmd["vx"] == 127
    assert cmd["vy"] == -127
    assert cmd["omega"] == MAX_PWM  # 1.0 / 1.0 = 1.0 → int(1.0 * 255) = 255


async def test_send_velocity_tracks_last_velocity():
    stub = _make_stub()
    await stub.send_velocity(0.3, -0.1, 0.5)
    assert stub._last_velocity == (0.3, -0.1, 0.5)


async def test_send_velocity_clamps_to_max_pwm():
    stub = _make_stub(max_vel=1.0, max_omega=1.0)
    await stub.send_velocity(10.0, 10.0, 10.0)  # Way over max
    cmd = stub.commands_sent[0]
    assert cmd["vx"] == MAX_PWM
    assert cmd["vy"] == MAX_PWM
    assert cmd["omega"] == MAX_PWM


async def test_send_velocity_clamps_to_negative_max_pwm():
    stub = _make_stub(max_vel=1.0, max_omega=1.0)
    await stub.send_velocity(-10.0, -10.0, -10.0)
    cmd = stub.commands_sent[0]
    assert cmd["vx"] == -MAX_PWM
    assert cmd["vy"] == -MAX_PWM
    assert cmd["omega"] == -MAX_PWM


# ---------------------------------------------------------------------------
# read_encoders
# ---------------------------------------------------------------------------


async def test_read_encoders_parses_full_response():
    stub = _make_stub()
    stub.query_responses["encoders"] = {
        "lv": 0.5,
        "rv": 0.3,
        "ox": 1.0,
        "oy": 2.0,
        "h": 0.1,
        "ts": 100.0,
    }
    reading = await stub.read_encoders()
    assert isinstance(reading, EncoderReading)
    assert reading.left_velocity_mps == 0.5
    assert reading.right_velocity_mps == 0.3
    assert reading.odometry_x_m == 1.0
    assert reading.odometry_y_m == 2.0
    assert reading.heading_rad == 0.1
    assert reading.timestamp == 100.0


async def test_read_encoders_defaults_missing_keys_to_zero():
    stub = _make_stub()
    stub.query_responses["encoders"] = {}
    reading = await stub.read_encoders()
    assert reading.left_velocity_mps == 0.0
    assert reading.right_velocity_mps == 0.0
    assert reading.odometry_x_m == 0.0


# ---------------------------------------------------------------------------
# get_battery_voltage
# ---------------------------------------------------------------------------


async def test_get_battery_voltage_returns_parsed_voltage():
    stub = _make_stub()
    stub.query_responses["battery"] = {"v": 11.7}
    voltage = await stub.get_battery_voltage()
    assert voltage == pytest.approx(11.7)


async def test_get_battery_voltage_defaults_to_zero_on_empty():
    stub = _make_stub()
    voltage = await stub.get_battery_voltage()
    assert voltage == 0.0


async def test_get_battery_voltage_sends_battery_cmd(monkeypatch: pytest.MonkeyPatch):
    """Verify the ESP32_CMD_TYPE_BATTERY command is passed to _query_data."""
    received: list[dict[str, int] | None] = []

    original_query = stub_ref = None

    stub = _make_stub()

    async def capturing_query(
        resource: str,
        cmd: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        received.append(cmd)
        return {"v": 9.9}

    stub._query_data = capturing_query  # type: ignore[method-assign]
    await stub.get_battery_voltage()

    assert len(received) == 1
    assert received[0] == {"T": ESP32_CMD_TYPE_BATTERY}


# ---------------------------------------------------------------------------
# emergency_stop
# ---------------------------------------------------------------------------


async def test_emergency_stop_sends_stop_command():
    stub = _make_stub()
    stub._last_velocity = (1.0, 0.0, 0.5)
    await stub.emergency_stop()
    assert len(stub.commands_sent) == 1
    assert stub.commands_sent[0] == {"T": ESP32_CMD_TYPE_STOP}


async def test_emergency_stop_zeros_last_velocity():
    stub = _make_stub()
    stub._last_velocity = (1.0, 0.5, 0.2)
    await stub.emergency_stop()
    assert stub._last_velocity == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# isinstance / protocol checks
# ---------------------------------------------------------------------------


def test_serial_driver_is_base_esp32_driver():
    from mousedroid.comms.serial_driver import SerialESP32Driver

    cfg = ESP32Config()
    driver = SerialESP32Driver(cfg)
    assert isinstance(driver, BaseESP32Driver)


def test_wifi_driver_is_base_esp32_driver():
    from mousedroid.comms.wifi_driver import WiFiESP32Driver

    cfg = ESP32Config(protocol="wifi")
    driver = WiFiESP32Driver(cfg)
    assert isinstance(driver, BaseESP32Driver)
