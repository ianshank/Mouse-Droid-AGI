"""F-059: the ESP32 liveness probe over the REAL driver stack and a fake serial port.

The unit tier scripts a fake driver behind the factory seam. This tier lets
``build_esp32_driver`` assemble the real ``ResilientESP32Driver`` ->
``SerialESP32Driver`` pair and answers from the F-025 ``FakeSerialPort``, so the
stock codec, the T-gate that sets ``EncoderReading.imu_valid``, the adaptive
timeout counters and the exact bytes written to the wire all participate.

The wire constants below are restated on purpose rather than imported from
``mousedroid.comms.command_set``: the assertions pin what actually goes down the
UART independently of the production constants that generate it.
"""

from __future__ import annotations

import json

import pytest

from mousedroid.config.schema.hardware import WAVESHARE_STOCK_BAUD
from mousedroid.validation.runtime import collect_esp32_probe
from tests.integration.test_f025_integration import FakeSerialPort, _settings

_CMD_ROS_CTRL = 13
_CMD_BASE_FEEDBACK = 130
_CMD_HEART_BEAT_SET = 136
_FEEDBACK_BASE_INFO = 1001


def _patch_port(monkeypatch: pytest.MonkeyPatch, responses: list[str]) -> FakeSerialPort:
    """Put a ``FakeSerialPort`` behind ``SerialESP32Driver`` (the F-025 idiom).

    ``_serial_mod`` is replaced by ``setattr`` on the specific symbol, never via
    ``patch.dict(sys.modules)`` + ``reload`` — the reload form evicts real
    modules from the import cache and poisons later tests in the same process.
    The probe builds its own driver, so unlike F-025 there is no driver to hand
    back; the port is what the tests inspect.
    """
    from mousedroid.comms import serial_driver as serial_driver_mod
    from mousedroid.comms.serial_driver import SerialESP32Driver

    port = FakeSerialPort(responses)
    # The driver refuses to connect when pyserial is absent; the fake port
    # supplies the behaviour under test, so only the presence check matters.
    monkeypatch.setattr(serial_driver_mod, "_serial_mod", object(), raising=False)
    monkeypatch.setattr(SerialESP32Driver, "_open_serial", lambda _self: port)
    return port


async def test_probe_polls_cmd_base_feedback_and_reads_the_bus_voltage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = json.dumps(
        {"T": _FEEDBACK_BASE_INFO, "L": 0.0, "R": 0.0, "r": 0.0, "p": 0.0, "y": 0.1, "v": 11.8}
    )
    port = _patch_port(monkeypatch, responses=[frame, frame])

    result = await collect_esp32_probe(_settings(command_set="waveshare_stock"))

    assert result.responded
    assert result.frame_type == _FEEDBACK_BASE_INFO
    assert result.battery_v == pytest.approx(11.8)
    commands = port.written_commands
    # Heartbeat armed at connect, then one encoder poll, then one battery poll.
    assert [c["T"] for c in commands] == [
        _CMD_HEART_BEAT_SET,
        _CMD_BASE_FEEDBACK,
        _CMD_BASE_FEEDBACK,
    ]
    # A liveness probe must never move the wheels: no CMD_ROS_CTRL, ever.
    assert all(c["T"] != _CMD_ROS_CTRL for c in commands)
    assert port.closed


async def test_wrong_typed_frames_never_count_as_a_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stock shares one stream across frame types; only T:1001 is the witness."""
    port = _patch_port(monkeypatch, responses=[json.dumps({"T": 1003})] * 2)

    result = await collect_esp32_probe(
        _settings(command_set="waveshare_stock", max_consecutive_timeouts=2)
    )

    assert not result.responded
    assert result.frame_type is None
    assert [c["T"] for c in port.written_commands] == [
        _CMD_HEART_BEAT_SET,
        _CMD_BASE_FEEDBACK,
        _CMD_BASE_FEEDBACK,
    ]
    assert port.closed


async def test_silent_port_spends_the_poll_budget_then_names_port_and_stock_baud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead or mis-baud board: the case the probe exists to diagnose."""
    poll_budget = 3
    port = _patch_port(monkeypatch, responses=[])

    result = await collect_esp32_probe(
        _settings(command_set="waveshare_stock", max_consecutive_timeouts=poll_budget)
    )

    assert not result.responded
    assert result.frame_type is None
    assert result.battery_v is None
    assert [c["T"] for c in port.written_commands] == [
        _CMD_HEART_BEAT_SET,
        *([_CMD_BASE_FEEDBACK] * poll_budget),
    ]
    # The operator-facing detail must carry what a baud-mismatch triage needs.
    assert "no FEEDBACK_BASE_INFO" in result.detail
    assert f"{WAVESHARE_STOCK_BAUD} baud" in result.detail
    assert port.closed
