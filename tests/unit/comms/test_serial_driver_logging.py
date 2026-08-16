"""Verify the central command_dispatch logging helper emits structlog events."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import structlog

from mousedroid.comms.base_driver import log_command_dispatch
from mousedroid.comms.mock_driver import MockESP32Driver
from mousedroid.config.schema import Settings


@pytest.fixture
def capture_log_events() -> Iterator[list[dict[str, Any]]]:
    """Capture structlog events for one test and restore configuration after."""
    captured: list[dict[str, Any]] = []

    def _capture(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        captured.append(dict(event_dict))
        return event_dict

    prior = structlog.get_config()
    structlog.configure(
        processors=[
            _capture,
            # Final renderer collapses the event dict to a string so the
            # default PrintLogger.msg() does not receive kwargs.
            structlog.processors.KeyValueRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        cache_logger_on_first_use=False,
    )
    try:
        yield captured
    finally:
        structlog.configure(**prior)


async def test_log_command_dispatch_emits_structured_event(
    capture_log_events: list[dict[str, Any]],
) -> None:
    driver = MockESP32Driver(cfg=Settings().esp32)
    await driver.connect()
    try:
        log_command_dispatch(driver_name="mock", vx=0.1, vy=0.0, omega=0.0)
    finally:
        await driver.disconnect()

    matching = [e for e in capture_log_events if e.get("event") == "command_dispatch"]
    assert matching, "command_dispatch event was not emitted"
    assert matching[0]["driver"] == "mock"
    assert matching[0]["vx"] == 0.1
    assert matching[0]["vy"] == 0.0
    assert matching[0]["omega"] == 0.0


def test_log_command_dispatch_does_not_require_a_driver_instance(
    capture_log_events: list[dict[str, Any]],
) -> None:
    """The helper is a pure function — drivers can call it from any context."""
    log_command_dispatch(driver_name="resilient", vx=-0.05, vy=0.0, omega=0.2)
    events = [e for e in capture_log_events if e.get("event") == "command_dispatch"]
    assert events
    assert events[-1]["driver"] == "resilient"
    assert events[-1]["omega"] == 0.2


async def test_send_velocity_emits_command_dispatch_via_base_driver(
    capture_log_events: list[dict[str, Any]],
) -> None:
    """Regression — wiring the helper into ``BaseESP32Driver.send_velocity``.

    Previously ``log_command_dispatch`` was defined but never called from
    any driver (Gemini code-review finding 1: dead code). The wiring in
    ``base_driver.send_velocity`` now means every driver that inherits the
    base — Mock, Serial, WiFi — emits the same ``command_dispatch`` event
    shape on every velocity dispatch, so smoke-time triage greps land
    uniform records regardless of which transport fielded the call.
    """
    driver = MockESP32Driver(cfg=Settings().esp32)
    await driver.connect()
    try:
        await driver.send_velocity(0.07, 0.02, -0.1)
    finally:
        await driver.disconnect()

    matching = [e for e in capture_log_events if e.get("event") == "command_dispatch"]
    assert matching, "send_velocity must emit command_dispatch via base driver wiring"
    last = matching[-1]
    assert last["driver"] == "MockESP32Driver"
    assert last["vx"] == 0.07
    assert last["vy"] == 0.02
    assert last["omega"] == -0.1


def test_esp32_debug_log_max_chars_default_and_range() -> None:
    """Schema field ``debug_log_max_chars`` replaces the prior hardcoded 200.

    Pinned default (200) preserves backwards-compat with existing log-line
    consumers; the ``ge=16`` and ``le=4096`` bounds prevent operator
    misconfiguration that would either truncate the JSON framing entirely
    or balloon log files during triage.
    """
    import pytest
    from pydantic import ValidationError

    from mousedroid.config.schema import ESP32Config

    cfg = ESP32Config()
    assert cfg.debug_log_max_chars == 200

    # Operator widens for triage — still valid.
    widened = ESP32Config(debug_log_max_chars=1024)
    assert widened.debug_log_max_chars == 1024

    # Lower bound: 16 chars must at least carry the JSON framing.
    with pytest.raises(ValidationError):
        ESP32Config(debug_log_max_chars=8)

    # Upper bound: 4096 caps log-volume blowup.
    with pytest.raises(ValidationError):
        ESP32Config(debug_log_max_chars=10_000)
