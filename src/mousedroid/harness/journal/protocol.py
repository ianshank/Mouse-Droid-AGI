"""Protocol and shared dataclass for the harness journal."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class JournalEntry:
    """One immutable entry written to the agent ledger.

    Attributes:
        ts_ns: Capture time, ``time.monotonic_ns()`` recommended.
        task_id: Optional task this entry belongs to.
        phase: Free-form phase string (e.g. ``"pre_tick"`` or ``"action"``).
        event: Stable event identifier (e.g. ``"task_submitted"``).
        payload: Arbitrary JSON-serialisable data.
        agent_id: Optional id of the (sub-)agent that produced the entry.
    """

    ts_ns: int = field(default_factory=time.monotonic_ns)
    task_id: str | None = None
    phase: str = ""
    event: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    agent_id: str | None = None


@runtime_checkable
class JournalProtocol(Protocol):
    """Append-only persistent ledger of agent activity."""

    async def start(self) -> None:
        """Initialise backing store + spawn the writer task (if any)."""
        ...

    async def stop(self) -> None:
        """Flush pending writes and release resources."""
        ...

    async def append(self, entry: JournalEntry) -> None:
        """Enqueue ``entry`` for non-blocking persistence.

        Implementations must never block the caller (the orchestrator's
        30 Hz tick) on I/O. Drop-with-warning is preferred over backpressure.
        """
        ...

    def read_all(self) -> AsyncIterator[JournalEntry]:
        """Iterate every entry currently persisted, oldest first.

        Implementations are typically async generators (``async def`` +
        ``yield``) so callers iterate via ``async for entry in j.read_all()``.
        """
        ...

    @property
    def is_running(self) -> bool:
        """True after ``start()`` and before ``stop()``."""
        ...


__all__ = ["JournalEntry", "JournalProtocol"]
