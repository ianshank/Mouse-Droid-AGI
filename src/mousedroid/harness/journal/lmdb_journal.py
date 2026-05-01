"""LMDB-backed harness journal — append-only ledger using msgpack.

Mirrors the pattern of :class:`mousedroid.experience.logger.ExperienceLogger`
but lives in its own database directory so it never collides with the
experience replay store. All knobs (path, map_size_gb, flush_every_n,
queue_max) come from :class:`HarnessJournalConfig`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import struct
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import lmdb

from mousedroid.constants import GB_TO_BYTES
from mousedroid.harness.journal.protocol import JournalEntry
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import HarnessJournalConfig

_log = get_logger(__name__)


def _serialize(entry: JournalEntry) -> bytes:
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
    ).encode("utf-8")


def _deserialize(blob: bytes) -> JournalEntry:
    raw = json.loads(blob.decode("utf-8"))
    return JournalEntry(
        ts_ns=int(raw["ts_ns"]),
        task_id=raw.get("task_id"),
        phase=raw.get("phase", ""),
        event=raw.get("event", ""),
        payload=raw.get("payload", {}),
        agent_id=raw.get("agent_id"),
    )


class LMDBJournal:
    """Append-only LMDB-backed agent ledger with a background writer."""

    def __init__(self, cfg: HarnessJournalConfig) -> None:
        self._cfg = cfg
        self._path = Path(cfg.path)
        self._flush_every_n = cfg.flush_every_n
        self._map_size = max(1, math.ceil(cfg.map_size_gb * GB_TO_BYTES))
        self._env: lmdb.Environment | None = None
        self._sequence = 0
        self._write_count = 0
        self._queue: asyncio.Queue[JournalEntry] = asyncio.Queue(maxsize=cfg.queue_max)
        self._writer_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        await asyncio.to_thread(self._open_env)
        self._running = True
        self._writer_task = asyncio.create_task(self._writer_loop(), name="lmdb-journal-writer")
        _log.info(
            "lmdb_journal_started",
            path=str(self._path),
            map_size=self._map_size,
            queue_max=self._cfg.queue_max,
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        await self._queue.join()
        if self._writer_task is not None:
            self._writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._writer_task
            self._writer_task = None
        await asyncio.to_thread(self._close_env)
        _log.info("lmdb_journal_stopped")

    async def append(self, entry: JournalEntry) -> None:
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
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
        if self._env is None:
            return
        entries = await asyncio.to_thread(self._read_all_blocking)
        for entry in entries:
            yield entry

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------ helpers
    def _open_env(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        self._env = lmdb.open(str(self._path), map_size=self._map_size, max_dbs=1)

    def _close_env(self) -> None:
        if self._env is not None:
            self._env.sync()
            self._env.close()
            self._env = None

    def _make_key(self) -> bytes:
        self._sequence += 1
        ts = time.monotonic_ns()
        return struct.pack(">QQ", ts, self._sequence)

    async def _flush_queue(self) -> None:
        if not self._running:
            return
        await self._queue.join()

    async def _writer_loop(self) -> None:
        try:
            while True:
                entry = await self._queue.get()
                try:
                    await asyncio.to_thread(self._write_entry_blocking, entry)
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            _log.debug("lmdb_journal_writer_cancelled")
            raise
        except Exception:  # pragma: no cover - defensive
            _log.error("lmdb_journal_writer_crashed", exc_info=True)
            raise

    def _write_entry_blocking(self, entry: JournalEntry) -> None:
        if self._env is None:  # pragma: no cover - defensive
            return
        key = self._make_key()
        data = _serialize(entry)
        with self._env.begin(write=True) as txn:
            txn.put(key, data)
        self._write_count += 1
        if self._write_count >= self._flush_every_n:
            self._env.sync()
            self._write_count = 0

    def _read_all_blocking(self) -> list[JournalEntry]:
        if self._env is None:
            return []
        out: list[JournalEntry] = []
        with self._env.begin() as txn, txn.cursor() as cursor:
            for _key, blob in cursor:
                out.append(_deserialize(bytes(blob)))
        return out


__all__ = ["LMDBJournal"]
