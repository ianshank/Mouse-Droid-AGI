"""Serial writes must never overlap an in-flight read on the same handle.

``SerialESP32Driver`` talks to a single pyserial handle that is not
thread-safe, and its ``_query_data`` performs a *send-then-read* pair whose
atomicity the reply framing depends on. Two independent defects broke that:

1. ``_send_command`` — the path behind ``send_velocity`` and
   ``emergency_stop`` — took no lock at all, so a write could land between a
   query's send and its read and have its ACK consumed as the query's reply.
2. Taking the lock on both paths is *still* not enough. Cancelling a coroutine
   parked on an executor future releases the ``async with`` immediately while
   the OS thread keeps running, so after a tick timeout an emergency-stop write
   could reach the port while ``readline()`` still owned it.

The driver now holds ``_io_lock`` on both paths (atomicity) *and* routes every
blocking call through a single-worker executor (ordering that survives
cancellation). These tests pin both halves.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from mousedroid.comms.serial_driver import SerialESP32Driver
from mousedroid.config.schema import ESP32Config


class _RecordingSerial:
    """A fake pyserial handle that records the order operations *complete*."""

    def __init__(self, events: list[str], read_block: threading.Event) -> None:
        self.timeout = 0.5
        self._events = events
        self._read_block = read_block
        self.closed = False

    def write(self, payload: bytes) -> int:
        self._events.append("write")
        return len(payload)

    def readline(self) -> bytes:
        self._events.append("read-start")
        # Hold the handle the way a real blocking readline() would.
        self._read_block.wait(timeout=5.0)
        self._events.append("read-end")
        return b'{"ok": true}\n'

    def close(self) -> None:
        self.closed = True


def _make_driver(events: list[str], read_block: threading.Event) -> SerialESP32Driver:
    driver = SerialESP32Driver(ESP32Config())
    driver._serial = _RecordingSerial(events, read_block)
    driver._connected = True
    return driver


@pytest.mark.asyncio
async def test_emergency_stop_write_never_overlaps_an_inflight_read() -> None:
    """A write issued after a cancelled read must land *after* it, not during.

    Red against the pre-fix driver: ``_send_command`` used the shared default
    executor with no lock, so the write completed while ``readline()`` was
    still holding the handle — the ordering below came out as
    ``read-start, write, read-end``.
    """
    events: list[str] = []
    read_block = threading.Event()
    driver = _make_driver(events, read_block)

    query = asyncio.create_task(driver._query_data("battery"))
    # Let the read reach the blocking call on the I/O thread.
    deadline = time.monotonic() + 2.0
    while "read-start" not in events and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert "read-start" in events, "read never started"

    # Simulate the tick-timeout path: the awaiting coroutine is cancelled while
    # the OS thread is still inside readline().
    query.cancel()
    with pytest.raises(asyncio.CancelledError):
        await query

    stop = asyncio.create_task(driver.emergency_stop())
    await asyncio.sleep(0.05)

    # The write must not have completed yet — the single I/O worker is busy.
    assert "write" not in events, (
        f"emergency-stop write reached the port while readline() still owned "
        f"it (events={events}). The lock alone does not cover this: "
        f"cancellation releases it while the OS thread keeps running."
    )

    read_block.set()
    await asyncio.wait_for(stop, timeout=5.0)

    assert events.index("read-end") < events.index("write"), (
        f"write must be ordered after the in-flight read completes: {events}"
    )


@pytest.mark.asyncio
async def test_send_command_holds_the_io_lock() -> None:
    """A write must not split a concurrent query's send-then-read pair.

    Red against the pre-fix driver, where ``_send_command`` took no lock.
    """
    events: list[str] = []
    read_block = threading.Event()
    driver = _make_driver(events, read_block)

    query = asyncio.create_task(driver._query_data("battery", {"T": 1}))
    deadline = time.monotonic() + 2.0
    while "read-start" not in events and time.monotonic() < deadline:
        await asyncio.sleep(0.01)

    assert driver._io_lock.locked(), "_query_data must hold _io_lock across send+read"

    send = asyncio.create_task(driver.send_velocity(0.1, 0.0, 0.0))
    await asyncio.sleep(0.05)
    assert not send.done(), "_send_command must wait on _io_lock, not interleave"

    read_block.set()
    await asyncio.wait_for(query, timeout=5.0)
    await asyncio.wait_for(send, timeout=5.0)


@pytest.mark.asyncio
async def test_disconnect_releases_the_io_executor() -> None:
    """The I/O thread must not outlive the driver, and reconnect must work.

    Backwards-compatibility half: the executor is an implementation detail, so
    connect/disconnect cycles must behave exactly as before.
    """
    events: list[str] = []
    read_block = threading.Event()
    read_block.set()  # never block in this test
    driver = _make_driver(events, read_block)

    await driver._query_data("battery")
    assert driver._io_executor is not None, "an I/O call must start the executor"

    await driver.disconnect()
    assert driver._io_executor is None, "disconnect must release the executor"
    assert driver._connected is False

    # A subsequent operation transparently starts a fresh executor.
    driver._serial = _RecordingSerial(events, read_block)
    await driver._query_data("battery")
    assert driver._io_executor is not None


@pytest.mark.asyncio
async def test_unconnected_driver_starts_no_thread() -> None:
    """Constructing a driver must not spawn an I/O thread.

    Mock-mode wiring and unit tests build drivers they never connect; a thread
    per construction would leak.
    """
    driver = SerialESP32Driver(ESP32Config())
    assert driver._io_executor is None
