"""Tests for ``mousedroid.common.async_utils``.

These tests validate the strong-reference guarantee of
``spawn_tracked`` and the drain semantics of ``cancel_and_drain``. The
goal is to prevent regressions where fire-and-forget tasks are GC'd
before completion or where cleanup deadlocks on in-flight work.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect

import pytest

from mousedroid.common.async_utils import TIMEOUT_ERRORS, cancel_and_drain, spawn_tracked


@pytest.mark.asyncio
async def test_spawn_tracked_retains_task_until_done() -> None:
    """Task must stay in the tracking set until completion."""
    tasks: set[asyncio.Task[int]] = set()

    async def produce() -> int:
        await asyncio.sleep(0)
        return 42

    task = spawn_tracked(tasks, produce(), name="produce")
    assert task in tasks
    assert task.get_name() == "produce"

    result = await task
    # Allow done-callback to run and remove the task
    await asyncio.sleep(0)

    assert result == 42
    assert task not in tasks


@pytest.mark.asyncio
async def test_spawn_tracked_removes_on_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failing tasks must still be removed from the tracking set."""
    tasks: set[asyncio.Task[None]] = set()

    async def boom() -> None:
        raise RuntimeError("boom")

    task = spawn_tracked(tasks, boom(), name="boom")
    with pytest.raises(RuntimeError, match="boom"):
        await task
    await asyncio.sleep(0)

    assert task not in tasks


@pytest.mark.asyncio
async def test_spawn_tracked_survives_cancel() -> None:
    """Cancellation must not leak entries into the tracking set."""
    tasks: set[asyncio.Task[None]] = set()

    async def never() -> None:
        await asyncio.sleep(3600)

    task = spawn_tracked(tasks, never())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert task not in tasks


@pytest.mark.asyncio
async def test_cancel_and_drain_clears_pending_tasks() -> None:
    """``cancel_and_drain`` must cancel and await every pending task."""
    tasks: set[asyncio.Task[None]] = set()

    async def never() -> None:
        await asyncio.sleep(3600)

    for _ in range(3):
        spawn_tracked(tasks, never())

    cancelled = await cancel_and_drain(tasks, timeout_s=1.0)
    assert cancelled == 3
    # After drain every tracked task is done (and thus removed)
    await asyncio.sleep(0)
    assert not tasks


@pytest.mark.asyncio
async def test_cancel_and_drain_with_empty_set_is_noop() -> None:
    """An empty tracking set must return immediately with zero."""
    tasks: set[asyncio.Task[None]] = set()
    assert await cancel_and_drain(tasks, timeout_s=0.1) == 0


@pytest.mark.asyncio
async def test_cancel_and_drain_swallows_a_drain_timeout() -> None:
    """A task that outlives ``timeout_s`` must not propagate out of the drain.

    The timeout branch had no coverage, and on the Python 3.10 floor it was
    dead code: ``asyncio.wait_for`` raises ``asyncio.TimeoutError``, which is a
    distinct class from the builtin ``TimeoutError`` before 3.11, so the
    original bare ``except TimeoutError`` never matched. ``cancel_and_drain``
    then raised out of ``_lifecycle_mixin.stop()`` *before* the emergency stop.

    This asserts the documented contract directly — cleanup never raises — so
    the guarantee holds on every supported interpreter rather than only on the
    one the developer happens to run.
    """
    tasks: set[asyncio.Task[None]] = set()
    released = asyncio.Event()

    async def stubborn() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # Outlive the drain deadline while unwinding, the way a task with
            # slow cleanup in its own ``finally`` would.
            await released.wait()
            raise

    spawn_tracked(tasks, stubborn(), log_exceptions=False)
    await asyncio.sleep(0)

    # Must return normally rather than raising the drain timeout.
    assert await cancel_and_drain(tasks, timeout_s=0.05) == 1

    released.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_timeout_errors_catches_every_timeout_flavour() -> None:
    """``TIMEOUT_ERRORS`` must match all three timeout classes.

    On Python 3.11+ the three names denote one class and this is trivially
    true; on the 3.10 floor they are distinct, and a handler spelled with any
    single one of them is dead code for the other two.
    """
    for raised in (TimeoutError, asyncio.TimeoutError, concurrent.futures.TimeoutError):
        try:
            raise raised()
        except TIMEOUT_ERRORS:
            continue
        finally:
            pass
        pytest.fail(f"TIMEOUT_ERRORS did not catch {raised!r}")


@pytest.mark.asyncio
async def test_cancel_and_drain_timeout_is_caught_by_the_shared_tuple() -> None:
    """The drain's ``except`` clause must be the shared tuple, not one name.

    A source-level assertion, because the behavioural test above passes on
    3.11+ even with the buggy single-name spelling — the interpreter this
    suite usually runs on cannot distinguish them. This one can.
    """
    source = inspect.getsource(cancel_and_drain)
    assert "except TIMEOUT_ERRORS:" in source, (
        "cancel_and_drain must catch TIMEOUT_ERRORS. A bare `except "
        "TimeoutError` (or `except asyncio.TimeoutError` alone) is dead code "
        "for at least one timeout class on the Python 3.10 floor, which would "
        "let the drain raise out of orchestrator shutdown before the "
        "emergency stop."
    )
