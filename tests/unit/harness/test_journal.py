"""Tests for the harness journal — null, JSONL and LMDB backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.schema import HarnessJournalConfig
from mousedroid.harness.journal.jsonl_journal import JSONLJournal
from mousedroid.harness.journal.lmdb_journal import LMDBJournal
from mousedroid.harness.journal.null_journal import NullJournal
from mousedroid.harness.journal.protocol import JournalEntry, JournalProtocol


def _entry(event: str, **payload: object) -> JournalEntry:
    return JournalEntry(event=event, payload=dict(payload))


# ---------------------------------------------------------------------------
# NullJournal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_null_journal_implements_protocol() -> None:
    j = NullJournal()
    assert isinstance(j, JournalProtocol)
    await j.start()
    assert j.is_running
    await j.append(_entry("x"))
    await j.stop()
    assert not j.is_running


@pytest.mark.asyncio
async def test_null_journal_read_all_is_empty() -> None:
    j = NullJournal()
    await j.start()
    entries = [e async for e in j.read_all()]
    assert entries == []
    await j.stop()


# ---------------------------------------------------------------------------
# JSONLJournal
# ---------------------------------------------------------------------------


@pytest.fixture
def jsonl_cfg(tmp_path: Path) -> HarnessJournalConfig:
    return HarnessJournalConfig(
        backend="jsonl",
        path=tmp_path / "journal.jsonl",
        queue_max=8,
    )


@pytest.mark.asyncio
async def test_jsonl_implements_protocol(jsonl_cfg: HarnessJournalConfig) -> None:
    j = JSONLJournal(jsonl_cfg)
    assert isinstance(j, JournalProtocol)


@pytest.mark.asyncio
async def test_jsonl_append_and_read_round_trip(jsonl_cfg: HarnessJournalConfig) -> None:
    j = JSONLJournal(jsonl_cfg)
    await j.start()
    try:
        await j.append(_entry("submitted", task="t1"))
        await j.append(_entry("completed", task="t1"))
        entries = [e async for e in j.read_all()]
    finally:
        await j.stop()
    assert [e.event for e in entries] == ["submitted", "completed"]
    assert entries[0].payload["task"] == "t1"


@pytest.mark.asyncio
async def test_jsonl_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "deeper" / "log.jsonl"
    cfg = HarnessJournalConfig(backend="jsonl", path=nested, queue_max=4)
    j = JSONLJournal(cfg)
    await j.start()
    await j.append(_entry("x"))
    await j.stop()
    assert nested.parent.is_dir()
    assert nested.exists()


@pytest.mark.asyncio
async def test_jsonl_overflow_drops_oldest(tmp_path: Path) -> None:
    """When the queue saturates, oldest entries are dropped with a warn-log.

    We synchronously fill the queue without yielding to the loop so the
    writer task cannot drain it; the second-overflow append must succeed
    and the oldest entry must be missing from the final read.
    """
    cfg = HarnessJournalConfig(
        backend="jsonl",
        path=tmp_path / "journal.jsonl",
        queue_max=2,
    )
    j = JSONLJournal(cfg)
    await j.start()
    # Fill the queue + trigger an overflow without yielding to the loop.
    j._queue.put_nowait(_entry("a"))  # type: ignore[attr-defined]
    j._queue.put_nowait(_entry("b"))  # type: ignore[attr-defined]
    await j.append(_entry("c"))  # should drop "a" and enqueue "c"
    await j.stop()
    entries = [e async for e in j.read_all()]
    events = {e.event for e in entries}
    assert "a" not in events
    assert {"b", "c"}.issubset(events)


@pytest.mark.asyncio
async def test_jsonl_writer_task_recovers_after_stop_start(
    jsonl_cfg: HarnessJournalConfig,
) -> None:
    j = JSONLJournal(jsonl_cfg)
    await j.start()
    await j.append(_entry("first"))
    await j.stop()
    assert not j.is_running

    j2 = JSONLJournal(jsonl_cfg)
    await j2.start()
    await j2.append(_entry("second"))
    await j2.stop()
    raw = Path(jsonl_cfg.path).read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 2


# ---------------------------------------------------------------------------
# LMDBJournal
# ---------------------------------------------------------------------------


from tests import TEST_EXPERIENCE_MAP_SIZE_GB


@pytest.fixture
def lmdb_cfg(tmp_path: Path) -> HarnessJournalConfig:
    return HarnessJournalConfig(
        backend="lmdb",
        path=tmp_path / "journal_lmdb",
        map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB,
        queue_max=8,
        flush_every_n=1,
    )


@pytest.mark.asyncio
async def test_lmdb_implements_protocol(lmdb_cfg: HarnessJournalConfig) -> None:
    j = LMDBJournal(lmdb_cfg)
    assert isinstance(j, JournalProtocol)


@pytest.mark.asyncio
async def test_lmdb_round_trip(lmdb_cfg: HarnessJournalConfig) -> None:
    j = LMDBJournal(lmdb_cfg)
    await j.start()
    try:
        await j.append(_entry("submitted", task="alpha"))
        await j.append(_entry("completed", task="alpha"))
        entries = [e async for e in j.read_all()]
    finally:
        await j.stop()
    assert [e.event for e in entries] == ["submitted", "completed"]
    assert entries[1].payload["task"] == "alpha"


@pytest.mark.asyncio
async def test_lmdb_persists_across_runs(lmdb_cfg: HarnessJournalConfig) -> None:
    j1 = LMDBJournal(lmdb_cfg)
    await j1.start()
    await j1.append(_entry("first"))
    await j1.stop()

    j2 = LMDBJournal(lmdb_cfg)
    await j2.start()
    entries = [e async for e in j2.read_all()]
    await j2.stop()
    assert any(e.event == "first" for e in entries)


@pytest.mark.asyncio
async def test_lmdb_idempotent_double_start(lmdb_cfg: HarnessJournalConfig) -> None:
    j = LMDBJournal(lmdb_cfg)
    await j.start()
    await j.start()  # second call is a no-op
    assert j.is_running
    await j.stop()
