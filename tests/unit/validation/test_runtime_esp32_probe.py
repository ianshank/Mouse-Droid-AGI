"""Unit tests for ``collect_esp32_probe`` — a scripted driver behind the factory seam.

The probe is the witness behind the preflight ``esp32`` check (F-059, D-3): it
connects, polls for one stock ``FEEDBACK_BASE_INFO`` frame through the PUBLIC
``read_encoders()`` path, reads the bus voltage, and always disconnects. These
tests script the driver's answers so every branch is pinned without a wire;
``tests/integration/test_f059_esp32_probe_integration.py`` drives the real
driver stack over a fake serial port.
"""

from __future__ import annotations

import dataclasses

import pytest

import mousedroid.factory as factory
from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.schema import Settings
from mousedroid.validation import runtime
from mousedroid.validation.runtime import Esp32ProbeResult, collect_esp32_probe

#: ``Settings`` refuses ``mock_hardware=false`` without a distance sensor. The
#: probe never reads it; it only has to make the config valid.
_ULTRASONIC_CFG = {"trigger_pin": 17, "echo_pin": 27}

#: Poll budget used by the boundary tests (``esp32.max_consecutive_timeouts``).
_POLL_BUDGET = 3


def _cfg(**esp32: object) -> Settings:
    """Real-hardware settings on the stock command set; ``esp32`` overrides win."""
    return Settings.model_validate(
        {
            "mock_hardware": False,
            "ultrasonic": _ULTRASONIC_CFG,
            "esp32": {"command_set": "waveshare_stock", **esp32},
        }
    )


class _FakeDriver:
    """Scripted ``ESP32CommProtocol`` stand-in that records every call it gets."""

    def __init__(self, readings: list[EncoderReading], voltage: float = 11.9) -> None:
        self.readings = list(readings)
        self.voltage = voltage
        self.calls: list[str] = []

    async def connect(self) -> None:
        self.calls.append("connect")

    async def disconnect(self) -> None:
        self.calls.append("disconnect")

    async def read_encoders(self) -> EncoderReading:
        self.calls.append("read")
        return self.readings.pop(0) if self.readings else EncoderReading()

    async def get_battery_voltage(self) -> float:
        self.calls.append("battery")
        return self.voltage

    async def send_velocity(self, vx: float, vy: float, omega: float) -> None:
        self.calls.append("velocity")

    async def emergency_stop(self) -> None:
        self.calls.append("stop")


class _ReadRaises(_FakeDriver):
    """Driver whose encoder read blows up mid-probe (USB yanked, port vanished)."""

    async def read_encoders(self) -> EncoderReading:
        self.calls.append("read")
        raise OSError("port vanished")


class _ConnectRaises(_FakeDriver):
    """Driver that cannot open its port at all."""

    async def connect(self) -> None:
        self.calls.append("connect")
        raise OSError("no such port")


def _install(monkeypatch: pytest.MonkeyPatch, driver: _FakeDriver) -> None:
    """Make ``build_esp32_driver`` hand the probe ``driver``."""
    monkeypatch.setattr(factory, "build_esp32_driver", lambda cfg: driver)


async def test_first_base_info_frame_is_a_response(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDriver([EncoderReading(imu_valid=True)])
    _install(monkeypatch, fake)
    result = await collect_esp32_probe(_cfg())
    assert isinstance(result, Esp32ProbeResult)
    assert result.responded
    assert (result.frame_type, result.battery_v) == (1001, pytest.approx(11.9))
    assert result.elapsed_s >= 0.0  # monotonic clock granularity on Windows can read 0.0
    assert "after 1 poll(s)" in result.detail
    assert "battery_v=11.90" in result.detail
    assert fake.calls == ["connect", "read", "battery", "disconnect"]  # no motion command, ever


async def test_silent_board_is_bounded_by_max_consecutive_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDriver([])
    _install(monkeypatch, fake)
    result = await collect_esp32_probe(_cfg(max_consecutive_timeouts=_POLL_BUDGET))
    assert not result.responded
    assert result.frame_type is None
    assert result.battery_v is None
    assert fake.calls.count("read") == _POLL_BUDGET
    assert "battery" not in fake.calls  # nothing to read the bus voltage from
    assert fake.calls[-1] == "disconnect"
    assert "no FEEDBACK_BASE_INFO" in result.detail


@pytest.mark.parametrize(
    ("silent_polls", "expect_response"),
    [(_POLL_BUDGET - 1, True), (_POLL_BUDGET, False)],
    ids=["answers-on-the-last-allowed-poll", "answers-one-poll-past-the-budget"],
)
async def test_poll_budget_is_inclusive_of_max_consecutive_timeouts(
    monkeypatch: pytest.MonkeyPatch, silent_polls: int, expect_response: bool
) -> None:
    """Off-by-one pin: poll N counts, poll N+1 is never waited for."""
    readings = [EncoderReading()] * silent_polls + [EncoderReading(imu_valid=True)]
    fake = _FakeDriver(readings)
    _install(monkeypatch, fake)
    result = await collect_esp32_probe(_cfg(max_consecutive_timeouts=_POLL_BUDGET))
    assert result.responded is expect_response
    assert fake.calls.count("read") == min(silent_polls + 1, _POLL_BUDGET)
    assert fake.calls[-1] == "disconnect"
    if expect_response:
        assert f"after {_POLL_BUDGET} poll(s)" in result.detail


async def test_disconnects_even_when_the_read_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _ReadRaises([])
    _install(monkeypatch, fake)
    with pytest.raises(OSError, match="port vanished"):
        await collect_esp32_probe(_cfg())
    assert fake.calls == ["connect", "read", "disconnect"]


async def test_connect_errors_propagate_without_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _ConnectRaises([])
    _install(monkeypatch, fake)
    with pytest.raises(OSError, match="no such port"):
        await collect_esp32_probe(_cfg())
    assert "read" not in fake.calls
    assert "battery" not in fake.calls


async def test_zero_volt_sentinel_is_reported_as_no_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDriver([EncoderReading(imu_valid=True)], voltage=0.0)
    _install(monkeypatch, fake)
    result = await collect_esp32_probe(_cfg())
    assert result.responded  # the board answered; only its voltage field was unreadable
    assert result.battery_v is None
    # A missing voltage must never be rendered as a number an operator reads as a flat pack.
    assert "battery_v=unavailable" in result.detail
    assert "0.00" not in result.detail


async def test_legacy_command_set_refuses_to_fabricate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        factory, "build_esp32_driver", lambda cfg: pytest.fail("factory must not be reached")
    )
    with pytest.raises(ValueError, match="waveshare_stock"):
        await collect_esp32_probe(_cfg(command_set="legacy"))


def test_result_fields_are_the_public_contract() -> None:
    """Preflight and the MCP ``esp32_diagnostics`` tool both consume this shape."""
    fields = dataclasses.fields(Esp32ProbeResult)
    assert [f.name for f in fields] == [
        "responded",
        "frame_type",
        "battery_v",
        "elapsed_s",
        "detail",
    ]
    result = Esp32ProbeResult(False, None, None, 0.0, "no response")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, fields[0].name, True)


def test_probe_is_exported_from_the_runtime_facade() -> None:
    """The preflight suite monkeypatches ``runtime.<name>``, so the facade must carry it."""
    assert "Esp32ProbeResult" in runtime.__all__
    assert "collect_esp32_probe" in runtime.__all__
