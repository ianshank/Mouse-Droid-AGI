"""Unit tests for the structlog setup module.

Covers the level-string mapping, ``get_logger`` binding, and that
``configure_logging`` wires optional processors (log buffer / cloud sink) and
robot-id contextvars into the emit chain. The unit-tier ``conftest`` resets
structlog after each test, so reconfiguring here is isolated.
"""

from __future__ import annotations

from typing import Any

import pytest

from mousedroid.config.schema import LoggingConfig
from mousedroid.logging.setup import (
    _level_to_int,
    configure_logging,
    get_logger,
)


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("DEBUG", 10),
        ("INFO", 20),
        ("WARNING", 30),
        ("ERROR", 40),
        ("CRITICAL", 50),
        ("debug", 10),  # case-insensitive
        ("Warning", 30),
        ("NOTAREALLEVEL", 20),  # unknown falls back to INFO
    ],
)
def test_level_to_int(level: str, expected: int) -> None:
    """String levels map to their syslog integers; unknown falls back to INFO."""
    assert _level_to_int(level) == expected


def test_get_logger_returns_usable_logger() -> None:
    """``get_logger`` returns a bound logger exposing the standard methods."""
    log = get_logger("mousedroid.test")
    assert hasattr(log, "info")
    assert hasattr(log, "warning")
    # Emitting must not raise.
    log.info("smoke_event", value=1)


def _capturing_processor(sink: list[dict[str, Any]]) -> Any:
    def _proc(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        sink.append(dict(event_dict))
        return event_dict

    return _proc


def test_configure_logging_routes_through_log_buffer() -> None:
    """An injected log-buffer processor sees every emitted event."""
    captured: list[dict[str, Any]] = []
    configure_logging(
        LoggingConfig(level="DEBUG", format="json"),
        log_buffer=_capturing_processor(captured),
    )
    get_logger("t").info("hello", k=1)
    assert captured
    assert captured[-1]["event"] == "hello"
    assert captured[-1]["k"] == 1


def test_configure_logging_binds_robot_id_into_context() -> None:
    """``robot_id`` is bound into contextvars and surfaces on every event."""
    captured: list[dict[str, Any]] = []
    configure_logging(
        LoggingConfig(level="DEBUG", format="json"),
        cloud_logging_sink=_capturing_processor(captured),
        robot_id="rover-7",
    )
    get_logger("t").info("with_context")
    assert captured[-1]["robot_id"] == "rover-7"


def test_configure_logging_respects_level_filter() -> None:
    """A DEBUG event is filtered out when the configured level is WARNING."""
    captured: list[dict[str, Any]] = []
    configure_logging(
        LoggingConfig(level="WARNING", format="console"),
        log_buffer=_capturing_processor(captured),
    )
    log = get_logger("t")
    log.debug("suppressed")
    log.warning("surfaced")
    events = [entry["event"] for entry in captured]
    assert "surfaced" in events
    assert "suppressed" not in events
