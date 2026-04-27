"""Unit tests for :class:`LMDBReplayReader`."""

from __future__ import annotations

import asyncio
import struct
import time
from pathlib import Path

import lmdb
import numpy as np
import pytest

from mousedroid.config.schema import ExperienceConfig
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.training.replay import LMDBReplayReader, ReplayReaderProtocol
from tests import TEST_EXPERIENCE_MAP_SIZE_GB


def _make_record(reward: float = 0.0) -> MouseDroidExperienceRecord:
    return MouseDroidExperienceRecord(
        timestamp=time.time(),
        vision_features=np.zeros(256, dtype=np.float32),
        distance_m=1.0,
        motor_state=np.zeros(4, dtype=np.float32),
        action=np.zeros(3, dtype=np.float32),
        reward=reward,
        surprise=0.0,
    )


def _populate(path: Path, n: int) -> None:
    env = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    base = time.time()
    with env.begin(write=True) as txn:
        for i in range(n):
            key = struct.pack(">Q", int(base * 1_000_000) + i)
            txn.put(key, _make_record(reward=float(i)).serialize())
    env.close()


def _cfg(path: Path) -> ExperienceConfig:
    return ExperienceConfig(
        path=str(path),
        map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB,
        flush_every_n=5,
    )


def _drain(reader: LMDBReplayReader, chunk_size: int) -> list[list[object]]:
    async def _go() -> list[list[object]]:
        out: list[list[object]] = []
        async for chunk in reader.stream(chunk_size):
            out.append(list(chunk))
        return out

    return asyncio.run(_go())


def test_reader_satisfies_protocol(tmp_path: Path) -> None:
    reader = LMDBReplayReader(_cfg(tmp_path))
    assert isinstance(reader, ReplayReaderProtocol)


def test_empty_db_yields_nothing_and_does_not_raise(tmp_path: Path) -> None:
    # tmp_path exists but contains no LMDB env files yet — empty case.
    reader = LMDBReplayReader(_cfg(tmp_path))
    chunks = _drain(reader, chunk_size=8)
    assert chunks == []
    assert reader.stats["read_records"] == 0
    assert reader.stats["chunks_yielded"] == 0


def test_missing_path_yields_nothing(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path / "does_not_exist")
    reader = LMDBReplayReader(cfg)
    assert _drain(reader, chunk_size=4) == []
    assert reader.stats["read_records"] == 0


def test_full_dataset_round_trip(tmp_path: Path) -> None:
    _populate(tmp_path, n=100)
    reader = LMDBReplayReader(_cfg(tmp_path))
    chunks = _drain(reader, chunk_size=32)
    flat = [r for chunk in chunks for r in chunk]
    assert len(flat) == 100
    # Last chunk should be the remainder (100 % 32 == 4).
    assert [len(c) for c in chunks] == [32, 32, 32, 4]
    assert reader.stats["read_records"] == 100
    assert reader.stats["chunks_yielded"] == 4


def test_chunk_size_must_be_positive(tmp_path: Path) -> None:
    reader = LMDBReplayReader(_cfg(tmp_path))

    async def _go() -> None:
        async for _ in reader.stream(0):
            pass

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        asyncio.run(_go())


def test_schema_mismatch_is_counted_and_skipped(tmp_path: Path) -> None:
    # 2 valid records + 1 corrupt record with bad schema_version byte.
    _populate(tmp_path, n=2)

    env = lmdb.open(str(tmp_path), map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        # Inject a payload that deserializes to schema mismatch:
        # an msgpack-packed dict with schema_version=999.
        import msgpack

        bad = msgpack.packb({"schema_version": 999, "payload": {}})
        txn.put(b"\xff" * 8, bad)
    env.close()

    reader = LMDBReplayReader(_cfg(tmp_path))
    chunks = _drain(reader, chunk_size=8)
    flat = [r for chunk in chunks for r in chunk]
    assert len(flat) == 2
    assert reader.stats["skipped_schema_mismatch"] == 1
    assert reader.stats["read_records"] == 2


def test_path_override_takes_precedence(tmp_path: Path) -> None:
    real_path = tmp_path / "real"
    real_path.mkdir()
    _populate(real_path, n=5)

    # Wrong cfg path; override points at populated env.
    wrong_cfg = _cfg(tmp_path / "ignored")
    reader = LMDBReplayReader(wrong_cfg, path_override=str(real_path))
    flat = [r for chunk in _drain(reader, 8) for r in chunk]
    assert len(flat) == 5
    assert reader.path == real_path
