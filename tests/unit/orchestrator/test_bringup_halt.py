"""Bring-up halt: the wheels are commanded to zero before the first tick.

Peer review C4(d). The chassis firmware latches the last velocity it was
given, so a previous unclean stop -- ``SIGKILL``, a power cut, a dropped USB
link -- leaves the wheels driving. ``_LifecycleMixin.start`` connected the
ESP32 and then went straight on to sensors, telemetry, MCP and the LLM
gateway without contradicting that latched velocity, so nothing commanded a
stop until the first tick produced an action.
:mod:`mousedroid.orchestrator`'s own ``CLAUDE.md`` states the consequence
outright: *"the rover can be moving throughout bring-up"*.

This is the one chassis-failsafe item from the review's C4 list that needs
no firmware change and no working ESP32, which is why it defaults on.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mousedroid.config.schema import Settings


def _orchestrator_with_mock_esp32(*, stop_on_connect: bool | None = None):  # type: ignore[no-untyped-def]
    """Build a factory orchestrator whose ESP32 is an assertion surface."""
    from mousedroid.factory import build_orchestrator

    cfg = Settings(mock_hardware=True)
    if stop_on_connect is not None:
        cfg.esp32.stop_on_connect = stop_on_connect  # type: ignore[misc]

    orch = build_orchestrator(cfg)

    esp32 = AsyncMock()
    esp32.connect = AsyncMock()
    esp32.send_velocity = AsyncMock()
    esp32.emergency_stop = AsyncMock()
    esp32.disconnect = AsyncMock()
    esp32.read_encoders = AsyncMock(
        return_value=MagicMock(left_velocity_mps=0.0, right_velocity_mps=0.0, heading_rad=0.0)
    )
    orch._esp32 = esp32  # type: ignore[attr-defined]
    return orch, esp32


async def test_zero_velocity_is_commanded_on_connect() -> None:
    """The defect: nothing contradicted a latched velocity at bring-up."""
    orch, esp32 = _orchestrator_with_mock_esp32()
    await orch._halt_on_connect()  # type: ignore[attr-defined]
    esp32.send_velocity.assert_awaited_once_with(0.0, 0.0, 0.0)


async def test_halt_uses_send_velocity_not_emergency_stop() -> None:
    """Reserve the e-stop path, and its log event, for genuine faults.

    Every boot emitting ``esp32_emergency_stop`` would make that event
    useless as an incident signal, which is the reason this is a plain
    zero-velocity command.
    """
    orch, esp32 = _orchestrator_with_mock_esp32()
    await orch._halt_on_connect()  # type: ignore[attr-defined]
    esp32.emergency_stop.assert_not_awaited()


async def test_toggle_off_restores_the_legacy_connect_sequence() -> None:
    """``stop_on_connect: false`` must be byte-identical to pre-fix behaviour."""
    orch, esp32 = _orchestrator_with_mock_esp32(stop_on_connect=False)
    await orch._halt_on_connect()  # type: ignore[attr-defined]
    esp32.send_velocity.assert_not_awaited()
    esp32.emergency_stop.assert_not_awaited()


async def test_a_failing_driver_does_not_block_startup() -> None:
    """A best-effort safety measure must not become a new way to fail boot.

    ``connect()`` already reports a driver that cannot be reached; raising
    here would trade a fail-safer improvement for a startup failure.
    """
    orch, esp32 = _orchestrator_with_mock_esp32()
    esp32.send_velocity = AsyncMock(side_effect=OSError("port went away"))
    await orch._halt_on_connect()  # type: ignore[attr-defined]  # must not raise


async def test_halt_runs_inside_start_after_connect() -> None:
    """Order matters: the stop is useless before the link is open."""
    orch, esp32 = _orchestrator_with_mock_esp32()
    calls: list[str] = []
    esp32.connect = AsyncMock(side_effect=lambda: calls.append("connect"))
    esp32.send_velocity = AsyncMock(side_effect=lambda *a: calls.append("send_velocity"))
    try:
        await orch.start()  # type: ignore[attr-defined]
    finally:
        await orch.stop()  # type: ignore[attr-defined]
    assert "connect" in calls, "connect() never ran"
    assert "send_velocity" in calls, "bring-up halt never ran inside start()"
    assert calls.index("connect") < calls.index("send_velocity")


@pytest.mark.parametrize("value", [True, False])
def test_toggle_is_schema_backed_and_overridable(value: bool) -> None:
    """The knob is config, not a literal (invariant 2)."""
    cfg = Settings.model_validate({"mock_hardware": True, "esp32": {"stop_on_connect": value}})
    assert cfg.esp32.stop_on_connect is value


def test_toggle_defaults_on() -> None:
    """Fail-safer default: existing YAML gains the stop without being edited."""
    assert Settings(mock_hardware=True).esp32.stop_on_connect is True
