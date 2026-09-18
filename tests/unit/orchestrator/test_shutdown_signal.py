"""Graceful-shutdown wiring for the 30 Hz mission loop (S-1).

``run()`` is ``while self._running:``, so flipping that flag lets the
in-flight tick finish and returns control to the caller's ``finally``,
which is where ``stop()`` -> ``_halt_actuators`` -> ``emergency_stop()``
lives. These tests pin that cooperative exit and the bounded escalation
that backs it up when the loop will not wind down on its own.

The wedged-loop escalation is driven through :class:`MockClock` so the
grace window is exercised deterministically without a multi-second test.
The clean-exit cases deliberately keep the real clock: ``run()`` sleeps on
``self._clock`` for its control period, so a mock nobody advances would
wedge the very loop those tests need to exit normally.
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from mousedroid.common.time.protocol import MockClock
from mousedroid.config.schema import Settings

_LOOP_SETTLE_S = 0.05
_DELIVERY_TIMEOUT_S = 5.0

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32",
    reason="loop.add_signal_handler is POSIX-only; Windows degrades to plain run()",
)


async def _deliver_sigterm(orch: object) -> None:
    """Raise a real SIGTERM and wait for the handler to observe it.

    asyncio routes POSIX signals through a self-pipe the selector drains on
    a loop iteration, so delivery is not synchronous with ``raise_signal``.
    ``_running`` flipping is the observable proof the handler ran.
    """
    signal.raise_signal(signal.SIGTERM)
    deadline = time.monotonic() + _DELIVERY_TIMEOUT_S
    while orch._running and time.monotonic() < deadline:  # type: ignore[attr-defined]
        await asyncio.sleep(0.01)
    assert orch._running is False, "SIGTERM never reached request_shutdown"  # type: ignore[attr-defined]


async def _advance_until_done(
    clock: MockClock,
    task: asyncio.Future[None],
    *,
    step_s: float,
) -> None:
    """Drive *clock* forward until *task* finishes or the timeout expires.

    Advancing repeatedly (rather than once) avoids racing the escalation
    task: it is spawned synchronously by the handler but only registers its
    ``clock.sleep`` waiter once the event loop actually runs its body.
    """
    deadline = time.monotonic() + _DELIVERY_TIMEOUT_S
    while not task.done() and time.monotonic() < deadline:
        clock.advance(step_s)
        await asyncio.sleep(0.01)


def _build_orchestrator(*, shutdown_grace_s: float = 2.0) -> tuple[object, AsyncMock]:
    """Build a factory orchestrator with hardware mocked out.

    Returns:
        ``(orchestrator, esp32_mock)`` — the mock is the assertion surface
        for ``emergency_stop``.
    """
    from mousedroid.factory import build_orchestrator

    cfg = Settings(mock_hardware=True)
    cfg.loop.tick_timeout_s = 0.1  # type: ignore[misc]
    cfg.loop.shutdown_grace_s = shutdown_grace_s  # type: ignore[misc]

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


# ---------------------------------------------------------------------------
# request_shutdown — the cooperative flag flip
# ---------------------------------------------------------------------------


async def test_request_shutdown_exits_the_run_loop() -> None:
    """The loop must wind down on its own, without being cancelled."""
    orch, _esp32 = _build_orchestrator()
    orch._running = True

    run_task = asyncio.ensure_future(orch.run())
    await asyncio.sleep(_LOOP_SETTLE_S)

    orch.request_shutdown("SIGTERM")
    await asyncio.wait_for(run_task, timeout=5.0)

    assert orch._running is False
    assert not run_task.cancelled()


async def test_request_shutdown_is_idempotent() -> None:
    """Both SIGTERM and SIGINT may arrive; the second must not raise."""
    orch, _esp32 = _build_orchestrator()
    orch._running = True

    orch.request_shutdown("SIGTERM")
    orch.request_shutdown("SIGINT")

    assert orch._running is False


async def test_request_shutdown_before_start_is_safe() -> None:
    """A signal racing bring-up must not explode; it just stays stopped."""
    orch, _esp32 = _build_orchestrator()

    orch.request_shutdown("SIGTERM")

    assert orch._running is False


# ---------------------------------------------------------------------------
# run_until_shutdown — the wiring main.py calls
# ---------------------------------------------------------------------------


async def test_run_until_shutdown_returns_after_cooperative_stop() -> None:
    """A signalled shutdown returns normally so the caller's finally runs."""
    orch, _esp32 = _build_orchestrator()
    orch._running = True

    task = asyncio.ensure_future(orch.run_until_shutdown())
    await asyncio.sleep(_LOOP_SETTLE_S)

    orch.request_shutdown("SIGTERM")
    await asyncio.wait_for(task, timeout=5.0)

    assert orch._running is False


async def test_run_until_shutdown_does_not_swallow_loop_errors() -> None:
    """A crash inside run() must still surface, not be masked as a shutdown."""
    orch, _esp32 = _build_orchestrator()
    orch._running = True

    boom = RuntimeError("loop exploded")
    orch.run = AsyncMock(side_effect=boom)

    with pytest.raises(RuntimeError, match="loop exploded"):
        await orch.run_until_shutdown()


async def test_outer_cancel_does_not_leave_the_loop_running() -> None:
    """Cancelling this coroutine must take the inner run task down with it.

    ``await task`` does not propagate an outer cancellation into the awaited
    task — it only raises at the await point. Without an explicit teardown
    the control loop would survive its own supervisor and keep issuing
    velocity commands, which is the exact failure class S-1 is about.
    """
    orch, _esp32 = _build_orchestrator()
    orch._running = True

    started = asyncio.Event()
    loop_cancelled = False

    async def _long_loop() -> None:
        nonlocal loop_cancelled
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            loop_cancelled = True
            raise

    orch.run = _long_loop

    task = asyncio.ensure_future(orch.run_until_shutdown())
    await asyncio.wait_for(started.wait(), timeout=5.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert loop_cancelled, "run task outlived an outer cancel"


# ---------------------------------------------------------------------------
# Escalation — the bounded backstop
# ---------------------------------------------------------------------------


@_POSIX_ONLY
async def test_wedged_loop_is_cancelled_after_grace_window() -> None:
    """A loop that ignores the flag is cancelled before SIGKILL can land.

    The whole point of the grace window: ``docker stop`` escalates to
    SIGKILL after 10 s, and SIGKILL cannot be handled — so the halt has to
    happen before it, even when the loop will not cooperate.

    Driven by a real signal because the escalation is armed by the signal
    handler, not by :meth:`request_shutdown` — calling the latter directly
    would exercise a path the rover never takes.
    """
    orch, _esp32 = _build_orchestrator(shutdown_grace_s=2.0)
    clock = MockClock()
    orch._clock = clock
    orch._running = True

    wedged: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def _never_returns() -> None:
        await wedged

    orch.run = _never_returns

    task = asyncio.ensure_future(orch.run_until_shutdown())
    await asyncio.sleep(_LOOP_SETTLE_S)

    await _deliver_sigterm(orch)
    await _advance_until_done(clock, task, step_s=2.0)

    assert task.done(), "wedged loop was never escalated to a cancel"
    await task


@_POSIX_ONLY
async def test_escalation_is_dropped_when_loop_exits_cleanly() -> None:
    """A clean exit must not leave an escalation task sleeping on the loop.

    Deliberately uses the real clock: ``run()`` sleeps on ``self._clock``
    for its control period, so a :class:`MockClock` that nobody advances
    would wedge the very loop this test needs to exit normally.

    The 30 s grace is far longer than the test, so an escalation that was
    not drained would still be pending at the assertion.
    """
    orch, _esp32 = _build_orchestrator(shutdown_grace_s=30.0)
    orch._running = True

    task = asyncio.ensure_future(orch.run_until_shutdown())
    await asyncio.sleep(_LOOP_SETTLE_S)
    await _deliver_sigterm(orch)
    await asyncio.wait_for(task, timeout=5.0)

    leaked = [
        t for t in asyncio.all_tasks() if t.get_name() == "shutdown_escalation" and not t.done()
    ]
    assert leaked == [], f"escalation task outlived the loop: {leaked}"
