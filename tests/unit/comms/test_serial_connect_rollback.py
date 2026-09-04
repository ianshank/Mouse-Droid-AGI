"""``connect()`` rollback must complete even when cancelled mid-rollback.

``connect()`` opens the port, then arms the codec's connect-time commands. If
arming fails it rolls back — close the handle, clear ``_serial``/``_connected``,
release the I/O executor — so ``ResilientESP32Driver``'s retry starts clean
instead of leaking one fd per attempt behind a ``_connected=True`` flag.

The rollback itself contains an ``await``. ``contextlib.suppress(Exception)``
around that await cannot hold a ``CancelledError``, which is a ``BaseException``
— so a cancellation delivered *during* rollback skipped the state reset and the
executor shutdown entirely and recreated the exact leak the block exists to
prevent, through a narrower window. ``orchestrator.start()`` awaits
``connect()``, so that window is reachable in production.

The fix clears driver state synchronously before the await and shields the
close, so every rollback step runs on every path.

Only the two seams that would touch real hardware are patched. ``_run_io``
stays the production implementation, so these tests exercise the real
single-worker executor — which is the thing under test.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from mousedroid.comms import serial_driver
from mousedroid.comms.serial_driver import SerialESP32Driver
from mousedroid.config.schema import ESP32Config


class _BlockingCloseSerial:
    """A fake handle whose ``close()`` blocks until the test releases it.

    Blocking inside ``close()`` is what makes the rollback's ``await`` a real
    suspension point, which is the only place the cancellation can land.
    """

    def __init__(self, close_entered: threading.Event, release: threading.Event) -> None:
        self.timeout = 0.5
        self.closed = False
        self._close_entered = close_entered
        self._release = release

    def write(self, payload: bytes) -> int:  # pragma: no cover - not exercised
        return len(payload)

    def readline(self) -> bytes:  # pragma: no cover - not exercised
        return b""

    def close(self) -> None:
        self._close_entered.set()
        self._release.wait(timeout=5.0)
        self.closed = True


def _driver_that_fails_to_arm(
    monkeypatch: pytest.MonkeyPatch, handle: _BlockingCloseSerial
) -> SerialESP32Driver:
    """A driver whose ``connect()`` opens *handle* and then fails to arm."""

    async def _boom() -> None:
        raise RuntimeError("codec arming failed")

    # connect() short-circuits when pyserial is absent; this environment has
    # no pyserial, and _open_serial is patched out anyway.
    monkeypatch.setattr(serial_driver, "_serial_mod", object())
    driver = SerialESP32Driver(ESP32Config())
    monkeypatch.setattr(driver, "_open_serial", lambda: handle)
    monkeypatch.setattr(driver, "_arm_command_set", _boom)
    return driver


@pytest.mark.asyncio
async def test_cancellation_during_rollback_still_clears_state_and_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Red before the fix: state reset and executor shutdown were skipped."""
    close_entered = threading.Event()
    release_close = threading.Event()
    handle = _BlockingCloseSerial(close_entered, release_close)
    driver = _driver_that_fails_to_arm(monkeypatch, handle)

    task = asyncio.create_task(driver.connect())

    # Wait until rollback has reached the blocking close, then cancel it there.
    await asyncio.to_thread(close_entered.wait, 5.0)
    assert close_entered.is_set(), "rollback never reached the close; test is not exercising it"
    task.cancel()
    release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert driver._connected is False, (
        "cancellation during rollback left _connected=True over a handle the "
        "driver no longer tracks — a retrying ResilientESP32Driver would open a "
        "second port while advertising the first as live"
    )
    assert driver._serial is None, "the stale handle was still referenced after rollback"
    assert driver._io_executor is None, (
        "the I/O executor survived the rollback; each retried connect() would "
        "strand another serial worker thread"
    )


@pytest.mark.asyncio
async def test_uncancelled_rollback_closes_the_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordinary path must still actually close, not merely drop the handle."""
    close_entered = threading.Event()
    release_close = threading.Event()
    release_close.set()  # never block
    handle = _BlockingCloseSerial(close_entered, release_close)
    driver = _driver_that_fails_to_arm(monkeypatch, handle)

    with pytest.raises(RuntimeError, match="codec arming failed"):
        await driver.connect()

    assert handle.closed is True, "rollback dropped the handle without closing the fd"
    assert driver._connected is False
    assert driver._serial is None
    assert driver._io_executor is None
