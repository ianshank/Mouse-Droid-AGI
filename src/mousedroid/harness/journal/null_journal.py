"""No-op journal — the default backend when the harness is disabled."""

from __future__ import annotations

from collections.abc import AsyncIterator

from mousedroid.harness.journal.protocol import JournalEntry, JournalProtocol
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class NullJournal:
    """Implements :class:`JournalProtocol` with constant-time no-ops."""

    def __init__(self) -> None:
        self._running = False

    async def start(self) -> None:
        self._running = True
        _log.debug("null_journal_started")

    async def stop(self) -> None:
        self._running = False
        _log.debug("null_journal_stopped")

    async def append(self, entry: JournalEntry) -> None:
        return None

    async def read_all(self) -> AsyncIterator[JournalEntry]:
        # Empty async iterator without using yield-from semantics.
        if False:  # pragma: no cover - structural async generator hint
            yield JournalEntry()
        return

    @property
    def is_running(self) -> bool:
        return self._running


_PROTOCOL_CHECK: JournalProtocol = NullJournal()
del _PROTOCOL_CHECK


__all__ = ["NullJournal"]
