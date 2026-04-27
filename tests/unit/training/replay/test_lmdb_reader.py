"""Unit tests for ``mousedroid.training.replay.lmdb_reader``."""

from __future__ import annotations

import asyncio
import math
import struct
from pathlib import Path

import lmdb
import numpy as np
import pytest

from mousedroid.config.schema import ExperienceConfig
from mousedroid.experience.record import (
    SCHEMA_VERSION,
    MouseDroidExperienceRecord,
)
from mousedroid.training.replay.lmdb_reader import (
    GB_TO_BYTES,
    LmdbReplayReader,
    SchemaVersionMismatchError,
)

GB_PER_DB = 0.001  # 1 MiB-ish — plenty for tests


def _make_record(idx: int, *, ts: float = 0.0) -> MouseDroidExperienceRecord:
    return MouseDroidExperienceRecord(
        timestamp=ts,
        vision_features=np.full(8, float(idx), dtype=np.float32),
        distance_m=0.5 + 0.01 * idx,
        motor_state=np.zeros(4, dtype=np.float32),
        action=np.zeros(3, dtype=np.float32),
        reward=float(idx),
        surprise=0.0,
    )


def _populate_db(path: Path, n: int, *, ts_step: float = 0.1) -> None:
    map_size = max(1, math.ceil(GB_PER_DB * GB_TO_BYTES))
    env = lmdb.open(str(path), map_size=map_size, max_dbs=1)
    try:
        with env.begin(write=True) as txn:
            for i in range(n):
                rec = _make_record(i, ts=i * ts_step)
                txn.put(struct.pack(">Q", i), rec.serialize())
    finally:
        env.close()


def _populate_db_with_bad_record(path: Path) -> None:
    """Populate a DB with one valid and one schema-mismatched record."""
    import msgpack

    map_size = max(1, math.ceil(GB_PER_DB * GB_TO_BYTES))
    env = lmdb.open(str(path), map_size=map_size, max_dbs=1)
    try:
        with env.begin(write=True) as txn:
            rec = _make_record(0)
            txn.put(struct.pack(">Q", 0), rec.serialize())
            # Hand-rolled record with a future schema_version.
            bad = msgpack.packb({"schema_version": 999, "timestamp": 0.0})
            txn.put(struct.pack(">Q", 1), bad)
    finally:
        env.close()


# ----------------------------------------------------------------------
# Constructor validation
# ----------------------------------------------------------------------
def test_chunk_size_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        LmdbReplayReader(tmp_path, map_size_gb=GB_PER_DB, chunk_size=0)


def test_map_size_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="map_size_gb must be positive"):
        LmdbReplayReader(tmp_path, map_size_gb=0.0, chunk_size=64)


# ----------------------------------------------------------------------
# Open / close lifecycle
# ----------------------------------------------------------------------
def test_open_missing_path_raises(tmp_path: Path) -> None:
    reader = LmdbReplayReader(tmp_path / "does_not_exist", map_size_gb=GB_PER_DB, chunk_size=64)
    with pytest.raises(FileNotFoundError):
        reader.open()


def test_stream_before_open_raises(tmp_path: Path) -> None:
    db = tmp_path / "db"
    _populate_db(db, n=1)
    reader = LmdbReplayReader(db, map_size_gb=GB_PER_DB, chunk_size=64)
    with pytest.raises(RuntimeError, match="before open"):
        list(reader.stream_chunks())


def test_context_manager_opens_and_closes(tmp_path: Path) -> None:
    db = tmp_path / "db"
    _populate_db(db, n=3)
    reader = LmdbReplayReader(db, map_size_gb=GB_PER_DB, chunk_size=64)
    with reader as r:
        assert len(r) == 3
    # Second open after exit should still work.
    with reader as r:
        assert len(r) == 3


# ----------------------------------------------------------------------
# Chunking semantics
# ----------------------------------------------------------------------
def test_empty_db_yields_no_chunks(tmp_path: Path) -> None:
    db = tmp_path / "db"
    _populate_db(db, n=0)
    reader = LmdbReplayReader(db, map_size_gb=GB_PER_DB, chunk_size=8)
    with reader:
        chunks = list(reader.stream_chunks())
    assert chunks == []
    assert reader.stats.records_consumed == 0
    assert reader.stats.chunks_yielded == 0


def test_chunks_are_exact_size_until_remainder(tmp_path: Path) -> None:
    db = tmp_path / "db"
    _populate_db(db, n=20)
    reader = LmdbReplayReader(db, map_size_gb=GB_PER_DB, chunk_size=8)
    with reader:
        chunks = list(reader.stream_chunks())
    assert [len(c) for c in chunks] == [8, 8, 4]
    assert reader.stats.chunks_yielded == 3
    assert reader.stats.records_consumed == 20


def test_stream_records_yields_in_key_order(tmp_path: Path) -> None:
    db = tmp_path / "db"
    _populate_db(db, n=10)
    reader = LmdbReplayReader(db, map_size_gb=GB_PER_DB, chunk_size=4)
    with reader:
        rewards = [r.reward for r in reader.stream_records()]
    assert rewards == [float(i) for i in range(10)]


def test_chunk_size_larger_than_db_yields_single_chunk(tmp_path: Path) -> None:
    db = tmp_path / "db"
    _populate_db(db, n=3)
    reader = LmdbReplayReader(db, map_size_gb=GB_PER_DB, chunk_size=64)
    with reader:
        chunks = list(reader.stream_chunks())
    assert len(chunks) == 1
    assert len(chunks[0]) == 3


# ----------------------------------------------------------------------
# Schema-version handling
# ----------------------------------------------------------------------
def test_schema_mismatch_skipped_in_lenient_mode(tmp_path: Path) -> None:
    db = tmp_path / "db"
    _populate_db_with_bad_record(db)
    reader = LmdbReplayReader(db, map_size_gb=GB_PER_DB, chunk_size=8, strict_schema=False)
    with reader:
        records = list(reader.stream_records())
    assert len(records) == 1  # bad record skipped
    assert reader.stats.schema_mismatches == 1
    assert reader.stats.schema_mismatch_versions.get(999) == 1


def test_schema_mismatch_raises_in_strict_mode(tmp_path: Path) -> None:
    db = tmp_path / "db"
    _populate_db_with_bad_record(db)
    reader = LmdbReplayReader(db, map_size_gb=GB_PER_DB, chunk_size=8, strict_schema=True)
    with reader, pytest.raises(SchemaVersionMismatchError) as excinfo:
        list(reader.stream_records())
    assert excinfo.value.expected == SCHEMA_VERSION
    assert excinfo.value.actual == 999


# ----------------------------------------------------------------------
# from_config
# ----------------------------------------------------------------------
def test_from_config_uses_experience_path_by_default(tmp_path: Path) -> None:
    db = tmp_path / "db"
    _populate_db(db, n=5)
    cfg = ExperienceConfig(path=str(db), map_size_gb=GB_PER_DB, flush_every_n=1)
    reader = LmdbReplayReader.from_config(cfg, chunk_size=2)
    with reader:
        records = list(reader.stream_records())
    assert len(records) == 5


def test_from_config_source_path_override(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    override = tmp_path / "override"
    _populate_db(primary, n=2)
    _populate_db(override, n=7)
    cfg = ExperienceConfig(path=str(primary), map_size_gb=GB_PER_DB, flush_every_n=1)
    reader = LmdbReplayReader.from_config(cfg, chunk_size=4, source_path=str(override))
    with reader:
        records = list(reader.stream_records())
    assert len(records) == 7


# ----------------------------------------------------------------------
# Async streaming
# ----------------------------------------------------------------------
def test_async_stream_yields_same_records_as_sync(tmp_path: Path) -> None:
    db = tmp_path / "db"
    _populate_db(db, n=12)

    sync_reader = LmdbReplayReader(db, map_size_gb=GB_PER_DB, chunk_size=5)
    with sync_reader:
        sync_chunks = [[r.reward for r in c] for c in sync_reader.stream_chunks()]

    async_reader = LmdbReplayReader(db, map_size_gb=GB_PER_DB, chunk_size=5)

    async def collect() -> list[list[float]]:
        out: list[list[float]] = []
        async_reader.open()
        try:
            async for chunk in async_reader.stream_chunks_async():
                out.append([r.reward for r in chunk])
        finally:
            async_reader.close()
        return out

    async_chunks = asyncio.run(collect())
    assert async_chunks == sync_chunks
