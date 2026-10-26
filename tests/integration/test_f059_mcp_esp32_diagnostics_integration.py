"""F-059 4.3: the MCP ``esp32_diagnostics`` live branch over the REAL driver stack and a fake port.

The unit tier hands the tool an ``AsyncMock`` driver. This tier hands it the real
``ResilientESP32Driver`` -> ``SerialESP32Driver`` pair, connected the way the orchestrator
connects it, over the F-025 ``FakeSerialPort``: the stock codec, the T-gate behind
``imu_valid``, the adaptive-timeout counters and the exact bytes that reach the UART all
participate. What the tool puts on the wire is the claim under test, so the constants below
are restated on purpose rather than imported from ``mousedroid.comms.command_set``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from mousedroid.common.tools.registry import ToolRegistry, create_default_registry
from tests.integration.test_f025_integration import (
    FakeSerialPort,
    _build_with_fake_port,
    _settings,
)

_CMD_ROS_CTRL = 13  # velocity, and on stock also the zero-velocity stop
_CMD_BASE_FEEDBACK = 130
_FEEDBACK_BASE_INFO = 1001
_POLL_BUDGET = 3

_BASE_INFO = json.dumps(
    {"T": _FEEDBACK_BASE_INFO, "L": 0.0, "R": 0.0, "r": 0.0, "p": 0.0, "y": 0.1, "v": 11.8}
)
_POLL = {"T": _CMD_BASE_FEEDBACK}


async def _live_tool(
    monkeypatch: pytest.MonkeyPatch, responses: list[str], **esp32: Any
) -> tuple[ToolRegistry, FakeSerialPort]:
    """The orchestrator connects its driver; the tool is then built around that same driver.

    The connect-time heartbeat is cleared from the log: only what the diagnostics call
    itself writes is under test.
    """
    cfg = _settings(command_set="waveshare_stock", max_consecutive_timeouts=_POLL_BUDGET, **esp32)
    driver, port = _build_with_fake_port(monkeypatch, cfg, responses)
    await driver.connect()
    port.written.clear()
    return create_default_registry(settings=cfg, esp32_driver=driver), port


def _never_moves(port: FakeSerialPort) -> None:
    """No velocity and no stop frame (on stock the stop is a zero-velocity CMD_ROS_CTRL)."""
    assert _CMD_ROS_CTRL not in {command["T"] for command in port.written_commands}


async def test_a_frame_costs_one_feedback_poll_and_one_battery_poll_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, port = await _live_tool(monkeypatch, [_BASE_INFO, _BASE_INFO])

    result = await registry.dispatch("esp32_diagnostics")

    assert result == {
        "status": "ok",
        "source": "live_driver",
        "imu_valid": True,
        "battery_v": pytest.approx(11.8),
        "polls": 1,
    }
    assert port.written_commands == [_POLL, _POLL]
    _never_moves(port)
    assert not port.closed  # the orchestrator's port stays open: the tool never disconnects


async def test_a_wrong_typed_line_is_not_silence_and_polls_reports_the_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stock shares one stream across frame types; the second read finds the frame."""
    registry, port = await _live_tool(
        monkeypatch, [json.dumps({"T": 1003}), _BASE_INFO, _BASE_INFO]
    )

    result = await registry.dispatch("esp32_diagnostics")

    assert result["status"] == "ok"
    assert result["polls"] == 2
    assert result["battery_v"] == pytest.approx(11.8)
    assert port.written_commands == [_POLL, _POLL, _POLL]  # two encoder polls, one battery poll
    _never_moves(port)


async def test_wrong_typed_lines_for_the_whole_budget_are_no_frame_and_read_no_battery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The board talks, but never in the witness frame: the budget is spent on polls alone.

    Lines keep arriving, so the driver never degrades and every read reaches the wire. A
    battery read after a verdict of ``no_frame`` would therefore show up as a fourth frame.
    """
    registry, port = await _live_tool(monkeypatch, [json.dumps({"T": 1003})] * _POLL_BUDGET)

    result = await registry.dispatch("esp32_diagnostics")

    assert result == {
        "status": "no_frame",
        "source": "live_driver",
        "imu_valid": False,
        "battery_v": None,
        "polls": _POLL_BUDGET,
    }
    assert port.written_commands == [_POLL] * _POLL_BUDGET
    _never_moves(port)


async def test_silence_spends_the_whole_budget_on_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, port = await _live_tool(monkeypatch, [])

    result = await registry.dispatch("esp32_diagnostics")

    assert result["status"] == "no_frame"
    assert result["polls"] == _POLL_BUDGET
    assert port.written_commands == [_POLL] * _POLL_BUDGET
    _never_moves(port)


async def test_no_frame_is_the_links_current_view_not_a_fresh_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent board flips the shared driver to degraded; a repeat call writes nothing.

    The first call spends the budget and trips the driver's degraded mode. The second call,
    inside ``degraded_poll_interval_s``, is throttled by the driver: it issues its reads,
    puts nothing on the wire, and still answers ``no_frame``. That is why ``no_frame`` is
    not proof of a dead board, and why the operator guide says so.
    """
    registry, port = await _live_tool(monkeypatch, [], degraded_poll_interval_s=3600.0)

    first = await registry.dispatch("esp32_diagnostics")
    assert port.written_commands == [_POLL] * _POLL_BUDGET
    port.written.clear()

    second = await registry.dispatch("esp32_diagnostics")

    assert second == first
    assert second["status"] == "no_frame"
    assert second["polls"] == _POLL_BUDGET
    assert port.written == []
