"""Tests for SerialESP32Driver adaptive timeout state machine."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.comms.serial_driver import SerialESP32Driver
from mousedroid.config.schema import ESP32Config


def _make_driver(
    *,
    command_timeout_s: float = 0.5,
    degraded_timeout_s: float = 0.05,
    max_consecutive_timeouts: int = 5,
    degraded_poll_interval_s: float = 1.0,
) -> SerialESP32Driver:
    """Create a driver with configurable adaptive timeout settings."""
    cfg = ESP32Config(
        command_timeout_s=command_timeout_s,
        degraded_timeout_s=degraded_timeout_s,
        max_consecutive_timeouts=max_consecutive_timeouts,
        degraded_poll_interval_s=degraded_poll_interval_s,
    )
    return SerialESP32Driver(cfg)


# -- Constructor / initial state --


def test_adaptive_fields_from_config():
    driver = _make_driver(degraded_timeout_s=0.03, max_consecutive_timeouts=10)
    assert driver._normal_timeout == 0.5
    assert driver._degraded_timeout == 0.03
    assert driver._max_consecutive_timeouts == 10
    assert driver._consecutive_timeouts == 0
    assert driver._is_degraded is False


def test_is_degraded_property_initially_false():
    driver = _make_driver()
    assert driver.is_degraded is False


def test_consecutive_timeouts_property_initially_zero():
    driver = _make_driver()
    assert driver.consecutive_timeouts == 0


# -- should_skip_read --


def test_should_skip_read_false_when_not_degraded():
    driver = _make_driver()
    assert driver.should_skip_read() is False


def test_should_skip_read_true_when_degraded_and_interval_not_elapsed():
    driver = _make_driver(degraded_poll_interval_s=10.0)
    driver._is_degraded = True
    driver._last_probe_time = 1e18  # far in the future
    assert driver.should_skip_read() is True


# -- _enter_degraded / _exit_degraded --


def test_enter_degraded_sets_serial_timeout():
    driver = _make_driver(degraded_timeout_s=0.02)
    mock_serial = MagicMock()
    mock_serial.timeout = 0.5
    driver._serial = mock_serial
    driver._consecutive_timeouts = 5

    driver._enter_degraded()

    assert driver._is_degraded is True
    assert mock_serial.timeout == 0.02


def test_enter_degraded_idempotent():
    driver = _make_driver()
    mock_serial = MagicMock()
    driver._serial = mock_serial
    driver._is_degraded = True

    # Should not re-log or re-set
    driver._enter_degraded()
    assert driver._is_degraded is True


def test_exit_degraded_restores_timeout():
    driver = _make_driver(command_timeout_s=0.5)
    mock_serial = MagicMock()
    mock_serial.timeout = 0.05
    driver._serial = mock_serial
    driver._is_degraded = True
    driver._consecutive_timeouts = 10

    driver._exit_degraded()

    assert driver._is_degraded is False
    assert driver._consecutive_timeouts == 0
    assert mock_serial.timeout == 0.5


def test_exit_degraded_noop_when_not_degraded():
    driver = _make_driver()
    driver._is_degraded = False
    driver._exit_degraded()
    assert driver._is_degraded is False


# -- _record_timeout / _record_success --


def test_record_timeout_increments_counter():
    driver = _make_driver(max_consecutive_timeouts=10)
    driver._record_timeout()
    assert driver._consecutive_timeouts == 1
    assert driver._is_degraded is False


def test_record_timeout_enters_degraded_at_threshold():
    driver = _make_driver(max_consecutive_timeouts=3)
    mock_serial = MagicMock()
    driver._serial = mock_serial
    for _ in range(3):
        driver._record_timeout()
    assert driver._is_degraded is True
    assert driver._consecutive_timeouts == 3


def test_record_success_resets_counter():
    driver = _make_driver()
    driver._consecutive_timeouts = 4
    driver._record_success()
    assert driver._consecutive_timeouts == 0
    assert driver._is_degraded is False


def test_record_success_exits_degraded():
    driver = _make_driver()
    mock_serial = MagicMock()
    driver._serial = mock_serial
    driver._is_degraded = True
    driver._consecutive_timeouts = 10

    driver._record_success()

    assert driver._is_degraded is False
    assert driver._consecutive_timeouts == 0


# -- _read_json integration with adaptive timeout --


async def test_read_json_empty_increments_timeout():
    driver = _make_driver(max_consecutive_timeouts=5)
    driver._serial = MagicMock()
    driver._serial.readline.return_value = b""

    result = await driver._read_json()

    assert result == {}
    assert driver._consecutive_timeouts == 1


async def test_read_json_success_resets_timeout():
    driver = _make_driver()
    driver._serial = MagicMock()
    payload = {"v": 12.0}
    driver._serial.readline.return_value = json.dumps(payload).encode() + b"\n"
    driver._consecutive_timeouts = 3

    result = await driver._read_json()

    assert result == payload
    assert driver._consecutive_timeouts == 0


async def test_read_json_triggers_degraded_after_threshold():
    driver = _make_driver(max_consecutive_timeouts=2)
    mock_serial = MagicMock()
    mock_serial.readline.return_value = b""
    driver._serial = mock_serial

    await driver._read_json()
    assert driver._is_degraded is False

    await driver._read_json()
    assert driver._is_degraded is True
    assert mock_serial.timeout == driver._degraded_timeout


async def test_read_json_recovers_from_degraded():
    driver = _make_driver(max_consecutive_timeouts=1)
    mock_serial = MagicMock()
    driver._serial = mock_serial

    # Enter degraded
    mock_serial.readline.return_value = b""
    await driver._read_json()
    assert driver._is_degraded is True

    # Recover
    payload = {"left": 0.1, "right": 0.2}
    mock_serial.readline.return_value = json.dumps(payload).encode() + b"\n"
    result = await driver._read_json()

    assert result == payload
    assert driver._is_degraded is False
    assert mock_serial.timeout == driver._normal_timeout


# -- connect resets adaptive state --


async def test_connect_resets_adaptive_state():
    driver = _make_driver()
    driver._consecutive_timeouts = 10
    driver._is_degraded = True

    mock_serial = MagicMock()
    with patch("mousedroid.comms.serial_driver._serial_mod", mock_serial):
        await driver.connect()

    assert driver._consecutive_timeouts == 0
    assert driver._is_degraded is False


# -- Hypothesis property-based tests --


@given(
    sequence=st.lists(
        st.booleans(),
        min_size=1,
        max_size=100,
    ),
    threshold=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=200)
def test_state_machine_invariants(sequence: list[bool], threshold: int) -> None:
    """Property: after any sequence of successes/failures, invariants hold.

    - consecutive_timeouts is always >= 0
    - is_degraded implies consecutive_timeouts >= threshold
    - !is_degraded after a success implies consecutive_timeouts == 0
    """
    driver = _make_driver(max_consecutive_timeouts=threshold)
    mock_serial = MagicMock()
    driver._serial = mock_serial

    for success in sequence:
        if success:
            driver._record_success()
        else:
            driver._record_timeout()

    assert driver._consecutive_timeouts >= 0

    if driver._is_degraded:
        assert driver._consecutive_timeouts >= threshold

    # If the last event was a success, we must not be degraded
    if sequence and sequence[-1]:
        assert driver._consecutive_timeouts == 0
        assert driver._is_degraded is False
