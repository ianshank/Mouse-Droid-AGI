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
