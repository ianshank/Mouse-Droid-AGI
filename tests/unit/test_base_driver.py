"""Tests for BaseESP32Driver — shared high-level driver logic.

Uses a minimal in-memory ``StubESP32Driver`` to exercise the inherited
``send_velocity``, ``read_encoders``, ``get_battery_voltage``, and
``emergency_stop`` methods without real hardware or mock patches.
"""

from __future__ import annotations

from collections.abc import Mapping
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
        self.commands_sent: list[dict[str, float]] = []
        self.query_responses: dict[str, dict[str, Any]] = {}
        self.connect_called = False
        self.disconnect_called = False

    async def connect(self) -> None:
        self._connected = True
        self.connect_called = True

    async def disconnect(self) -> None:
        self._connected = False
        self.disconnect_called = True

    async def _send_command(self, cmd: Mapping[str, float]) -> None:
        self.commands_sent.append(dict(cmd))

    async def _query_data(
        self,
        resource: str,
        cmd: Mapping[str, float] | None = None,
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


# ---------------------------------------------------------------------------
# F-025 — command-set delegation (codec seam through the shared base)
#
# The stock pins live HERE because features.yaml F-025's validation_command
# runs this file: heartbeat armed at connect (T=136), battery step sends
# CMD_BASE_FEEDBACK (T=130) and NEVER the legacy {"T":2} (= stock
# CMD_SET_MOTOR_PID, a motor-controller write), and the legacy path stays
# byte-identical on the wire.
# ---------------------------------------------------------------------------


class RecordingStubDriver(StubESP32Driver):
    """Stub that also records query commands and mirrors transport connect.

    Real transports call ``_arm_command_set()`` at the end of ``connect()``;
    the base stub predates that hook, so this subclass reproduces the
    transport contract for connect-sequence pins.
    """

    def __init__(self, cfg: ESP32Config) -> None:
        super().__init__(cfg)
        # Mapping[str, float], not dict[str, int]: the stock codec builds
        # float-valued payloads ({"X": 0.25}), and the base class's payload
        # type is deliberately covariant so both codecs satisfy it.
        self.query_commands: list[dict[str, float] | None] = []

    async def connect(self) -> None:
        await super().connect()
        await self._arm_command_set()

    async def _query_data(
        self,
        resource: str,
        cmd: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        self.query_commands.append(dict(cmd) if cmd is not None else None)
        return self.query_responses.get(resource, {})


def _make_stock_stub(**overrides: Any) -> RecordingStubDriver:
    cfg = ESP32Config.model_validate(
        {
            "command_set": "waveshare_stock",
            "max_velocity_mps": 1.0,
            "max_omega_rads": 2.0,
            **overrides,
        }
    )
    return RecordingStubDriver(cfg)


def _make_legacy_recording_stub(**overrides: Any) -> RecordingStubDriver:
    cfg = ESP32Config.model_validate({"max_velocity_mps": 1.0, "max_omega_rads": 1.0, **overrides})
    return RecordingStubDriver(cfg)


class TestStockCommandSetDelegation:
    """waveshare_stock: every wire payload comes from the stock codec."""

    async def test_send_velocity_dispatches_ros_ctrl(self) -> None:
        driver = _make_stock_stub()
        await driver.send_velocity(0.25, 0.0, 1.5)
        assert driver.commands_sent == [{"T": 13, "X": 0.25, "Z": 1.5}]
        assert driver._last_velocity == (0.25, 0.0, 1.5)

    async def test_emergency_stop_sends_zero_velocity_ros_ctrl(self) -> None:
        """Stock firmware defines no e-stop command — the stop IS T=13 zeros."""
        driver = _make_stock_stub()
        await driver.emergency_stop()
        assert driver.commands_sent == [{"T": 13, "X": 0.0, "Z": 0.0}]
        assert driver._last_velocity == (0.0, 0.0, 0.0)

    async def test_battery_sends_base_feedback_never_pid_write(self) -> None:
        """The power_chain battery step must not poke the motor controller.

        Legacy ``{"T":2}`` is stock ``CMD_SET_MOTOR_PID`` — a WRITE that
        ``assert_power_chain`` fired immediately before commanding motion
        (audit R1/R5). Under the stock codec the battery step polls
        ``CMD_BASE_FEEDBACK`` (a read) and parses the 1001 frame.
        """
        driver = _make_stock_stub()
        driver.query_responses["battery"] = {"T": 1001, "L": 0.0, "R": 0.0, "v": 11.6}
        voltage = await driver.get_battery_voltage()
        assert voltage == pytest.approx(11.6)
        assert driver.query_commands == [{"T": 130}]
        assert {"T": ESP32_CMD_TYPE_BATTERY} not in driver.query_commands

    async def test_read_encoders_polls_base_feedback(self) -> None:
        """Stock frames are polled — an un-polled serial read yields nothing."""
        driver = _make_stock_stub()
        driver.query_responses["encoders"] = {"T": 1001, "L": 0.2, "R": -0.2, "v": 12.0}
        reading = await driver.read_encoders()
        assert driver.query_commands == [{"T": 130}]
        assert reading.left_velocity_mps == pytest.approx(0.2)
        assert reading.right_velocity_mps == pytest.approx(-0.2)

    async def test_stale_frame_parses_to_silent_zero(self) -> None:
        driver = _make_stock_stub()
        driver.query_responses["battery"] = {"T": 1003, "v": 11.6}
        assert await driver.get_battery_voltage() == 0.0

    async def test_connect_arms_heartbeat(self) -> None:
        """CMD_HEART_BEAT_SET goes out once per connect.

        The window is derived from the driver's worst-case command gap —
        with the shipped defaults that is ``degraded_poll_interval_s``
        (1.0 s) x multiple 3.0 = 3000 ms.
        """
        driver = _make_stock_stub()
        await driver.connect()
        assert driver.commands_sent == [{"T": 136, "cmd": 3000}]

    async def test_connect_heartbeat_disabled_sends_nothing(self) -> None:
        driver = _make_stock_stub(heartbeat_enabled=False)
        await driver.connect()
        assert driver.commands_sent == []


class TestLegacyCommandSetByteIdentity:
    """Default selector: the wire bytes are the pre-F-025 protocol exactly."""

    async def test_connect_sends_zero_extra_writes(self) -> None:
        """The _arm_command_set hook is a no-op under legacy — the connect
        sequence is byte-identical to the pre-selector driver."""
        driver = _make_legacy_recording_stub()
        await driver.connect()
        assert driver.commands_sent == []

    async def test_wire_json_byte_identical_to_historical_protocol(self) -> None:
        """Serialize every captured payload exactly as the serial transport
        does and compare against pre-F-025 golden strings."""
        import json

        driver = _make_legacy_recording_stub()
        await driver.send_velocity(0.5, -0.5, 1.0)
        await driver.emergency_stop()
        await driver.get_battery_voltage()
        wire = [json.dumps(dict(c)) for c in driver.commands_sent]
        wire += [json.dumps(dict(c)) for c in driver.query_commands if c is not None]
        assert wire == [
            '{"T": 1, "vx": 127, "vy": -127, "omega": 255}',
            '{"T": 0}',
            '{"T": 2}',
        ]

    async def test_encoder_read_still_sends_nothing(self) -> None:
        driver = _make_legacy_recording_stub()
        driver.query_responses["encoders"] = {"lv": 0.1, "rv": 0.1}
        await driver.read_encoders()
        assert driver.query_commands == [None]


class TestLateralVelocityWarnLatch:
    """vy on a codec without a lateral axis: WARN once, reset on reconnect."""

    async def test_latch_sets_once_under_stock(self) -> None:
        driver = _make_stock_stub()
        assert driver._lateral_warn_emitted is False
        await driver.send_velocity(0.1, 0.4, 0.0)
        assert driver._lateral_warn_emitted is True
        # Second send keeps the latch (DEBUG path) — no state flap.
        await driver.send_velocity(0.1, 0.4, 0.0)
        assert driver._lateral_warn_emitted is True

    async def test_zero_vy_never_sets_latch(self) -> None:
        driver = _make_stock_stub()
        await driver.send_velocity(0.1, 0.0, 0.2)
        assert driver._lateral_warn_emitted is False

    async def test_legacy_codec_never_sets_latch(self) -> None:
        """Legacy supports vy — the latch must stay untouched."""
        driver = _make_legacy_recording_stub()
        await driver.send_velocity(0.1, 0.4, 0.0)
        assert driver._lateral_warn_emitted is False

    async def test_reconnect_resets_latch(self) -> None:
        """Resilience-wrapper reconnects re-arm the operator warning."""
        driver = _make_stock_stub()
        await driver.send_velocity(0.1, 0.4, 0.0)
        assert driver._lateral_warn_emitted is True
        await driver.connect()
        assert driver._lateral_warn_emitted is False
