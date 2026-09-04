"""Reusable asyncio helpers for fire-and-forget task tracking.

Background coroutines spawned with :func:`asyncio.create_task` or
:func:`asyncio.ensure_future` must retain a strong reference or the task
may be garbage-collected mid-flight, silently cancelling the work.  This
module exposes :func:`spawn_tracked` which adds the new task to a
caller-owned set and wires an idempotent removal callback, so concrete
modules (orchestrator, cloud sinks, telemetry server) never have to
re-implement the pattern.

The helper is intentionally framework-agnostic — it does not know about
cloud, telemetry, or hardware — so it can be reused by any future module
that needs fire-and-forget semantics with bounded memory.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any, Final, TypeVar

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

_TaskResult = TypeVar("_TaskResult")

#: Every exception type a timed-out wait can raise, for use in ``except``.
#:
#: ``asyncio.TimeoutError`` and ``concurrent.futures.TimeoutError`` only became
#: aliases of the builtin ``TimeoutError`` in Python 3.11.  On the 3.10 floor
#: this project still supports — and which the Jetson rover image ships
#: (``dustynv/l4t-pytorch:r36.4.0`` provides ``python3.10``) — all three are
#: *distinct* classes, and the builtin is an ``OSError`` subclass.  So on 3.10 a
#: bare ``except TimeoutError`` catches neither an ``asyncio.wait_for`` timeout
#: nor a ``Future.result(timeout)`` timeout, and the handler is dead code.
#:
#: Catching this tuple is correct on every supported interpreter; on 3.11+ the
#: three names denote one class and the duplicates are harmless.  Use this
#: instead of writing any single name, so the 3.10 divergence cannot be
#: reintroduced one call site at a time.
TIMEOUT_ERRORS: Final[tuple[type[BaseException], ...]] = (
    TimeoutError,
    asyncio.TimeoutError,
    concurrent.futures.TimeoutError,
)


def spawn_tracked(
    tasks: set[asyncio.Task[Any]],
    coro: Coroutine[Any, Any, _TaskResult],
    *,
    name: str | None = None,
    log_exceptions: bool = True,
) -> asyncio.Task[_TaskResult]:
    """Spawn *coro* as a tracked :class:`asyncio.Task`.

    The new task is added to *tasks* immediately and removed when it
    completes (success, failure, or cancellation).  This guarantees that
    the task has at least one strong reference throughout its lifetime,
    preventing premature garbage collection of fire-and-forget work.

    When *log_exceptions* is true, any non-cancellation exception raised
    by the coroutine is logged via structlog at ``warning`` level.  The
    exception itself is still observed on the task object so callers that
    ``await`` it continue to see the error; this helper only ensures
    unobserved exceptions do not go to ``sys.stderr``.

    Args:
        tasks: Caller-owned set used to retain strong references.  Typically
            an attribute on a long-lived object (orchestrator, server).
        coro: Coroutine to schedule on the current event loop.
        name: Optional task name for debugging.  Passed through to
            :func:`asyncio.create_task`.
        log_exceptions: When ``True`` (default) log non-cancel errors via
            structlog with ``exc_info``.  Set to ``False`` when the caller
            awaits the task elsewhere and will observe the error itself.

    Returns:
        The newly created :class:`asyncio.Task`.

    Raises:
        RuntimeError: If called without a running event loop.
    """
    task: asyncio.Task[_TaskResult] = asyncio.create_task(coro, name=name)
    tasks.add(task)

    def _on_done(done: asyncio.Task[_TaskResult]) -> None:
        tasks.discard(done)
        if not log_exceptions:
            return
        if done.cancelled():
            return
        exc = done.exception()
        if exc is not None:
            _log.warning(
                "tracked_task_failed",
                task_name=done.get_name(),
                error=str(exc),
                error_type=type(exc).__name__,
            )

    task.add_done_callback(_on_done)
    return task


async def cancel_and_drain(tasks: set[asyncio.Task[Any]], *, timeout_s: float = 1.0) -> int:
    """Cancel every task in *tasks* and wait for completion.

    Used by shutdown paths that need to drain tracked background work.
    Exceptions from drained tasks are suppressed so cleanup never raises.

    Args:
        tasks: Set returned by a previous :func:`spawn_tracked` call.
            The set is not mutated by this helper; ``_on_done`` clears
            entries individually as each task resolves.
        timeout_s: Maximum seconds to wait for all tasks to finish.

    Returns:
        The number of tasks that were pending at the start of the call.
    """
    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    if not pending:
        return 0
    try:
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True),
            timeout=timeout_s,
        )
    except TIMEOUT_ERRORS:
        _log.warning("cancel_and_drain_timeout", pending=len(pending), timeout_s=timeout_s)
    return len(pending)
