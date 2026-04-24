"""Tests for ``mousedroid.common.async_utils``.

These tests validate the strong-reference guarantee of
``spawn_tracked`` and the drain semantics of ``cancel_and_drain``. The
goal is to prevent regressions where fire-and-forget tasks are GC'd
before completion or where cleanup deadlocks on in-flight work.
"""

from __future__ import annotations

import asyncio

import pytest

from mousedroid.common.async_utils import cancel_and_drain, spawn_tracked


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
