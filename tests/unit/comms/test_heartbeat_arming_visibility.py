"""The chassis failsafe must never be silently absent.

``ESP32Config.heartbeat_enabled`` defaults to ``True`` and
``ESP32Config.command_set`` defaults to ``"legacy"`` — but legacy firmware
has no ``CMD_HEART_BEAT_SET``, so ``LegacyCommandCodec.connect_commands``
correctly returns ``[]`` and nothing is armed. That combination is not a
bug in the codec; it is a deployment whose operator believes there is a
firmware-side failsafe and has none.

Before this, ``_arm_command_set`` logged ``esp32_heartbeat_armed`` only
when commands existed and said *nothing at all* otherwise, so the gap was
invisible in container logs. These tests pin the warning that makes it
visible, and pin that the armed path stays quiet about it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest
import structlog

from mousedroid.comms.base_driver import BaseESP32Driver
from mousedroid.comms.protocol import EncoderReading
from mousedroid.config.schema import ESP32Config

_UNARMED_EVENT = "esp32_heartbeat_unavailable"
_ARMED_EVENT = "esp32_heartbeat_armed"


@pytest.fixture
def capture_log_events() -> Iterator[list[dict[str, Any]]]:
    """Capture structlog events for one test and restore configuration after."""
    captured: list[dict[str, Any]] = []

    def _capture(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        captured.append(dict(event_dict))
        return event_dict

    prior = structlog.get_config()
    structlog.configure(
        processors=[_capture, structlog.processors.KeyValueRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        cache_logger_on_first_use=False,
    )
    try:
        yield captured
    finally:
        structlog.configure(**prior)


class _RecordingDriver(BaseESP32Driver):
    """Minimal concrete driver that records the frames it would send."""

    def __init__(self, cfg: ESP32Config) -> None:
        super().__init__(cfg)
        self.commands_sent: list[dict[str, float]] = []

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def _send_command(self, cmd: Mapping[str, float]) -> None:
        self.commands_sent.append(dict(cmd))

    async def _query_data(
        self,
        resource: str,
        cmd: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        return {}

    async def read_encoders(self) -> EncoderReading:  # pragma: no cover - unused here
        raise NotImplementedError


def _events(captured: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [e for e in captured if e.get("event") == name]


async def test_legacy_command_set_warns_that_heartbeat_cannot_arm(
    capture_log_events: list[dict[str, Any]],
) -> None:
    """The shipped defaults must say out loud that there is no failsafe."""
    cfg = ESP32Config(command_set="legacy", heartbeat_enabled=True)
    driver = _RecordingDriver(cfg)

    await driver._arm_command_set()

    warnings = _events(capture_log_events, _UNARMED_EVENT)
    assert len(warnings) == 1, capture_log_events
    assert warnings[0]["command_set"] == "legacy"


async def test_warning_names_the_config_that_would_fix_it(
    capture_log_events: list[dict[str, Any]],
) -> None:
    """A warning an operator cannot act on is noise — name the selector."""
    cfg = ESP32Config(command_set="legacy", heartbeat_enabled=True)
    driver = _RecordingDriver(cfg)

    await driver._arm_command_set()

    warning = _events(capture_log_events, _UNARMED_EVENT)[0]
    assert "waveshare_stock" in str(warning.get("remedy", ""))


async def test_no_warning_when_operator_disabled_the_heartbeat(
    capture_log_events: list[dict[str, Any]],
) -> None:
    """``heartbeat_enabled=False`` is the explicit way to opt out — stay quiet."""
    cfg = ESP32Config(command_set="legacy", heartbeat_enabled=False)
    driver = _RecordingDriver(cfg)

    await driver._arm_command_set()

    assert _events(capture_log_events, _UNARMED_EVENT) == []


async def test_armed_path_does_not_emit_the_warning(
    capture_log_events: list[dict[str, Any]],
) -> None:
    """Under waveshare_stock the failsafe really is armed — no warning."""
    cfg = ESP32Config(command_set="waveshare_stock", heartbeat_enabled=True)
    driver = _RecordingDriver(cfg)

    await driver._arm_command_set()

    assert _events(capture_log_events, _UNARMED_EVENT) == []
    assert len(_events(capture_log_events, _ARMED_EVENT)) == 1


async def test_legacy_connect_sequence_stays_byte_identical() -> None:
    """The warning must be log-only — no new frame may reach legacy firmware."""
    cfg = ESP32Config(command_set="legacy", heartbeat_enabled=True)
    driver = _RecordingDriver(cfg)

    await driver._arm_command_set()

    assert driver.commands_sent == []
