"""JSONL-backed harness journal with non-blocking writer task.

The orchestrator only ever calls ``put_nowait`` on the internal queue
during a tick; an :class:`asyncio.Task` drains the queue and flushes to
disk via :func:`asyncio.to_thread`. Path and queue sizing come from
:class:`mousedroid.config.schema.HarnessJournalConfig` — nothing is
hardcoded.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from mousedroid.harness.journal.protocol import JournalEntry
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import HarnessJournalConfig

_log = get_logger(__name__)


def _serialize(entry: JournalEntry) -> str:
    """Serialise a :class:`JournalEntry` into a single JSON line.

    Falls back to ``str(payload)`` for non-JSON-encodable payloads so the
    writer never raises during the hot loop.
    """
    return json.dumps(
        {
            "ts_ns": entry.ts_ns,
            "task_id": entry.task_id,
            "phase": entry.phase,
            "event": entry.event,
            "payload": entry.payload,
            "agent_id": entry.agent_id,
        },
        default=str,
        separators=(",", ":"),
    )


def _deserialize(line: str) -> JournalEntry:
    raw = json.loads(line)
    return JournalEntry(
        ts_ns=int(raw["ts_ns"]),
        task_id=raw.get("task_id"),
        phase=raw.get("phase", ""),
        event=raw.get("event", ""),
        payload=raw.get("payload", {}),
        agent_id=raw.get("agent_id"),
    )


class JSONLJournal:
    """Append-only JSONL journal with a background writer task."""

    def __init__(self, cfg: HarnessJournalConfig) -> None:
        self._cfg = cfg
        self._path = Path(cfg.path)
        self._queue: asyncio.Queue[JournalEntry] = asyncio.Queue(maxsize=cfg.queue_max)
        self._writer_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Touch the file so ``read_all`` can be called even before any writes.
        if not self._path.exists():
            await asyncio.to_thread(self._path.touch)
        self._running = True
        self._writer_task = asyncio.create_task(self._writer_loop(), name="jsonl-journal-writer")
        _log.info("jsonl_journal_started", path=str(self._path), queue_max=self._cfg.queue_max)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        # Wait for queued entries to be persisted, then cancel the writer.
        await self._queue.join()
        if self._writer_task is not None:
            self._writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._writer_task
            self._writer_task = None
        _log.info("jsonl_journal_stopped", path=str(self._path))

    async def append(self, entry: JournalEntry) -> None:
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            # Drop the OLDEST entry (excluding the trailing sentinel slot)
            # so the journal never exerts back-pressure on the tick loop.
            try:
                dropped = self._queue.get_nowait()
                self._queue.task_done()
                _log.warning(
                    "journal_overflow",
                    dropped_event=getattr(dropped, "event", None),
                    queue_max=self._cfg.queue_max,
                )
            except asyncio.QueueEmpty:  # pragma: no cover - defensive
                pass
            try:
                self._queue.put_nowait(entry)
            except asyncio.QueueFull:  # pragma: no cover - defensive
                _log.error("journal_dropped_after_overflow", event=entry.event)

    async def read_all(self) -> AsyncIterator[JournalEntry]:
        await self._flush_queue()
        if not self._path.exists():
            return
        # File reads happen in a worker thread to avoid blocking the loop.
        contents = await asyncio.to_thread(self._path.read_text, encoding="utf-8")
        for line in contents.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            yield _deserialize(stripped)

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------ helpers
    async def _flush_queue(self) -> None:
        """Wait for the writer to drain the queue (used by ``read_all``)."""
        if not self._running:
            return
        await self._queue.join()

    async def _writer_loop(self) -> None:
        try:
            while True:
                entry = await self._queue.get()
                try:
                    await self._write_entry(entry)
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            _log.debug("jsonl_journal_writer_cancelled")
            raise
        except Exception:  # pragma: no cover - defensive
            _log.error("jsonl_journal_writer_crashed", exc_info=True)
            raise

    async def _write_entry(self, entry: JournalEntry) -> None:
        line = _serialize(entry) + "\n"
        await asyncio.to_thread(self._append_to_disk, line)

    def _append_to_disk(self, line: str) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)


__all__ = ["JSONLJournal"]
