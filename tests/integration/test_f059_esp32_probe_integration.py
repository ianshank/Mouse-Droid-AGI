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
from pathlib import Path
from typing import Any

import pytest

import mousedroid.factory as factory
from mousedroid.config.schema import Settings
from mousedroid.config.schema.hardware import WAVESHARE_STOCK_BAUD
from mousedroid.resilience.retry import RetryExhaustedError
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


def _record_built_drivers(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Wrap the factory seam so a test can reach the driver the probe built for itself."""
    built: list[Any] = []
    real_build = factory.build_esp32_driver

    def recording(cfg: Settings) -> Any:
        driver = real_build(cfg)
        built.append(driver)
        return driver

    monkeypatch.setattr(factory, "build_esp32_driver", recording)
    return built


async def test_a_port_that_cannot_be_opened_does_not_strand_the_serial_io_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copilot review on #256: a failed ``connect()`` must still reach ``disconnect()``.

    ``SerialESP32Driver._run_io`` starts its single-worker executor before the open call
    runs, so an unopenable port (node missing, permission denied) left a live worker
    thread behind unless ``disconnect()`` released it -- and the probe only called that
    after a *successful* connect. Only the open is faked; the resilient wrapper and the
    serial driver are the real pair.
    """
    from mousedroid.comms import serial_driver as serial_driver_mod
    from mousedroid.comms.serial_driver import SerialESP32Driver

    monkeypatch.setattr(serial_driver_mod, "_serial_mod", object(), raising=False)

    def refuse(_self: SerialESP32Driver) -> None:
        msg = "could not open port: [Errno 2] No such file or directory"
        raise OSError(msg)

    monkeypatch.setattr(SerialESP32Driver, "_open_serial", refuse)
    built = _record_built_drivers(monkeypatch)
    cfg = _settings(command_set="waveshare_stock")
    # One attempt: the default retry policy would sleep through its real backoff delays.
    cfg.retry = cfg.retry.model_copy(update={"max_attempts": 1})

    with pytest.raises(RetryExhaustedError, match="No such file or directory"):
        await collect_esp32_probe(cfg)

    (resilient,) = built
    assert resilient.inner._io_executor is None


async def test_discovery_opens_a_by_id_node_but_the_summary_names_the_configured_literal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Why the summary says ``configured`` (Copilot review on #256).

    With USB-C discovery on and a stale literal ``serial_port``, the factory hands the
    serial driver a copy of the config pointing at the live by-id node. The probe only
    holds the original ``cfg``, so the configured literal is all it can name; it must
    not be presented as the port that answered. Discovery is real, over a temp by-id
    directory; only the serial port is fake.
    """
    from mousedroid.comms import serial_driver as serial_driver_mod
    from mousedroid.comms.serial_driver import SerialESP32Driver

    by_id = tmp_path / "by-id"
    by_id.mkdir()
    live = by_id / "usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_AAA-if00-port0"
    live.touch()
    stale = "/dev/serial/by-id/stale-literal-that-does-not-exist"
    frame = json.dumps(
        {"T": _FEEDBACK_BASE_INFO, "L": 0.0, "R": 0.0, "r": 0.0, "p": 0.0, "y": 0.1, "v": 11.8}
    )
    port = FakeSerialPort([frame, frame])
    opened: list[str] = []

    def open_recording(self: SerialESP32Driver) -> FakeSerialPort:
        opened.append(self._port)
        return port

    monkeypatch.setattr(serial_driver_mod, "_serial_mod", object(), raising=False)
    monkeypatch.setattr(SerialESP32Driver, "_open_serial", open_recording)
    cfg = Settings.model_validate(
        {
            "mock_hardware": False,
            "ultrasonic": {"trigger_pin": 17, "echo_pin": 27},
            "esp32": {
                "protocol": "serial",
                "enabled": True,
                "command_set": "waveshare_stock",
                "serial_port": stale,
            },
            "usbc_discovery": {
                "enabled": True,
                "by_id_root": str(by_id),
                "required_endpoints": [{"name": "rover_esp32", "by_id_glob": "*CP2102N*"}],
            },
        }
    )

    result = await collect_esp32_probe(cfg)

    assert result.responded
    assert opened == [str(live)]  # the driver opened the substituted node, not the literal
    assert f"configured serial_port={stale}" in result.detail
    assert str(live) not in result.detail  # the probe cannot see it; it must not imply it
