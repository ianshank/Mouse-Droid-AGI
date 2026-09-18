"""S-1 end-to-end: a SIGTERM must reach ``emergency_stop()``.

This is the finding itself, pinned at the level it actually failed. Every
piece below the entry point already worked — ``stop()`` calls
``_halt_actuators`` calls ``ESP32CommProtocol.emergency_stop`` — but
nothing ever *reached* them under ``docker stop`` or ``systemctl stop``,
because SIGTERM's default disposition terminates the process before any
``finally`` can unwind.

So the orchestrator here is the real factory-built one and ``stop()`` is
the real teardown; only the ESP32 transport is a mock, and only because it
is the assertion surface. Mocking the orchestrator would make the test
vacuous — the bug was never in a collaborator, it was in whether the
teardown ran at all.

Reverting ``main.py`` to ``await orch_obj.run()`` does not merely fail
these tests: the unhandled SIGTERM kills the pytest process outright,
which is precisely the production symptom.
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mousedroid.config.schema import Settings
from mousedroid.main import _run

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX signal delivery; Windows has no loop.add_signal_handler",
)

_READY_TIMEOUT_S = 10.0
_SHUTDOWN_TIMEOUT_S = 15.0


def _instrumented_orchestrator(cfg: Settings) -> tuple[object, AsyncMock]:
    """Factory-build the real orchestrator with the ESP32 transport mocked."""
    from mousedroid.factory import build_orchestrator

    orch = build_orchestrator(cfg)

    esp32 = AsyncMock()
    esp32.emergency_stop = AsyncMock()
    esp32.send_velocity = AsyncMock()
    esp32.connect = AsyncMock()
    esp32.disconnect = AsyncMock()
    esp32.read_encoders = AsyncMock(
        return_value=MagicMock(
            left_velocity_mps=0.0,
            right_velocity_mps=0.0,
            heading_rad=0.0,
        )
    )
    esp32.get_battery_voltage = AsyncMock(return_value=12.0)
    orch._esp32 = esp32

    orch._sensor_manager = AsyncMock()
    orch._sensor_manager.read_all = AsyncMock(
        return_value=MagicMock(valid_mask=0xFF, valid_sensor_count=4)
    )
    orch._sensor_manager.start = AsyncMock()
    orch._sensor_manager.stop = AsyncMock()
    orch._sensor_manager.recovery_attempt = AsyncMock(return_value=0)

    orch.tick = AsyncMock()
    return orch, esp32


async def _await_loop_running(orch: object) -> None:
    """Block until ``start()`` has flipped the orchestrator to running."""
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while not orch._running and time.monotonic() < deadline:  # type: ignore[attr-defined]
        await asyncio.sleep(0.01)
    assert orch._running is True, "orchestrator never reached its run loop"  # type: ignore[attr-defined]


async def test_sigterm_during_run_halts_the_motors() -> None:
    """The S-1 regression: SIGTERM must halt the wheels, not kill the process."""
    cfg = Settings(mock_hardware=True)
    cfg.loop.tick_timeout_s = 0.1  # type: ignore[misc]
    orch, esp32 = _instrumented_orchestrator(cfg)

    with patch("mousedroid.factory.build_orchestrator", return_value=orch):
        task = asyncio.ensure_future(_run(cfg))
        await _await_loop_running(orch)

        signal.raise_signal(signal.SIGTERM)

        await asyncio.wait_for(task, timeout=_SHUTDOWN_TIMEOUT_S)

    esp32.emergency_stop.assert_awaited()


async def test_sigint_during_run_halts_the_motors() -> None:
    """SIGINT takes the same path, so the interactive case cannot drift."""
    cfg = Settings(mock_hardware=True)
    cfg.loop.tick_timeout_s = 0.1  # type: ignore[misc]
    orch, esp32 = _instrumented_orchestrator(cfg)

    with patch("mousedroid.factory.build_orchestrator", return_value=orch):
        task = asyncio.ensure_future(_run(cfg))
        await _await_loop_running(orch)

        signal.raise_signal(signal.SIGINT)

        await asyncio.wait_for(task, timeout=_SHUTDOWN_TIMEOUT_S)

    esp32.emergency_stop.assert_awaited()


async def test_shutdown_also_releases_the_serial_transport() -> None:
    """``_halt_actuators`` disconnects too — a held port blocks the restart."""
    cfg = Settings(mock_hardware=True)
    cfg.loop.tick_timeout_s = 0.1  # type: ignore[misc]
    orch, esp32 = _instrumented_orchestrator(cfg)

    with patch("mousedroid.factory.build_orchestrator", return_value=orch):
        task = asyncio.ensure_future(_run(cfg))
        await _await_loop_running(orch)
        signal.raise_signal(signal.SIGTERM)
        await asyncio.wait_for(task, timeout=_SHUTDOWN_TIMEOUT_S)

    esp32.disconnect.assert_awaited()


async def test_signal_handlers_are_released_after_run_returns() -> None:
    """A leaked handler would outlive the loop and fire against a dead orchestrator."""
    cfg = Settings(mock_hardware=True)
    cfg.loop.tick_timeout_s = 0.1  # type: ignore[misc]
    orch, _esp32 = _instrumented_orchestrator(cfg)

    with patch("mousedroid.factory.build_orchestrator", return_value=orch):
        task = asyncio.ensure_future(_run(cfg))
        await _await_loop_running(orch)
        signal.raise_signal(signal.SIGTERM)
        await asyncio.wait_for(task, timeout=_SHUTDOWN_TIMEOUT_S)

    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
