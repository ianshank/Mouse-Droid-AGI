"""Smoke + behavioural tests for ``scripts/export_experience_to_training.py``.

The export CLI consumes the Phase 2 :class:`LMDBReplayReader` and writes
msgpack-gz shards. Tests use an inline LMDB store seeded with synthetic
records (mirrors ``tests/unit/training/replay/test_lmdb_reader.py``).
"""

from __future__ import annotations

import gzip
import struct
import sys
import time
from pathlib import Path

import lmdb
import msgpack
import numpy as np
import pytest

from mousedroid.experience.record import MouseDroidExperienceRecord

# Import the CLI module via its filesystem path so the test exercises the
# same script the operator runs.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "export_experience_to_training.py"
sys.path.insert(0, str(_SCRIPT.parent))
import export_experience_to_training as exporter


def _populate_lmdb(path: Path, n: int) -> None:
    env = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    base = time.time()
    with env.begin(write=True) as txn:
        for i in range(n):
            key = struct.pack(">Q", int(base * 1_000_000) + i)
            record = MouseDroidExperienceRecord(
                vision_features=np.zeros(256, dtype=np.float32),
                motor_state=np.zeros(4, dtype=np.float32),
                action=np.zeros(3, dtype=np.float32),
                reward=float(i),
            )
            txn.put(key, record.serialize())
    env.close()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_parse_args_requires_lmdb_and_dest() -> None:
    with pytest.raises(SystemExit):
        exporter._parse_args([])


def test_parse_args_defaults_match_module_constants(tmp_path: Path) -> None:
    args = exporter._parse_args(["--lmdb", str(tmp_path), "--dest", str(tmp_path / "out")])
    assert args.shard_size == exporter.DEFAULT_SHARD_SIZE
    assert args.chunk_size == exporter.DEFAULT_CHUNK_SIZE
    assert args.dry_run is False


def test_main_rejects_zero_shard_size(tmp_path: Path) -> None:
    rc = exporter.main(
        [
            "--lmdb",
            str(tmp_path),
            "--dest",
            str(tmp_path / "out"),
            "--shard-size",
            "0",
        ]
    )
    assert rc == exporter.EXIT_USAGE


def test_main_rejects_zero_chunk_size(tmp_path: Path) -> None:
    rc = exporter.main(
        [
            "--lmdb",
            str(tmp_path),
            "--dest",
            str(tmp_path / "out"),
            "--chunk-size",
            "0",
        ]
    )
    assert rc == exporter.EXIT_USAGE


# ---------------------------------------------------------------------------
# Missing LMDB
# ---------------------------------------------------------------------------


def test_missing_source_lmdb_returns_lmdb_err(tmp_path: Path) -> None:
    rc = exporter.main(
        [
            "--lmdb",
            str(tmp_path / "does-not-exist"),
            "--dest",
            str(tmp_path / "out"),
        ]
    )
    assert rc == exporter.EXIT_LMDB_ERR


# ---------------------------------------------------------------------------
# --dry-run path: no shards on disk, exit 0
# ---------------------------------------------------------------------------


def test_dry_run_does_not_create_dest(tmp_path: Path) -> None:
    src = tmp_path / "src.lmdb"
    src.mkdir()
    _populate_lmdb(src, n=8)
    dest = tmp_path / "shards"

    rc = exporter.main(
        [
            "--lmdb",
            str(src),
            "--dest",
            str(dest),
            "--dry-run",
            "--shard-size",
            "4",
        ]
    )

    assert rc == exporter.EXIT_OK
    assert not dest.exists(), "dry run must not create the dest directory"


# ---------------------------------------------------------------------------
# Real export: writes shards, payload round-trips
# ---------------------------------------------------------------------------


def test_real_export_writes_expected_shard_count(tmp_path: Path) -> None:
    src = tmp_path / "src.lmdb"
    src.mkdir()
    _populate_lmdb(src, n=10)
    dest = tmp_path / "shards"

    rc = exporter.main(
        [
            "--lmdb",
            str(src),
            "--dest",
            str(dest),
            "--shard-size",
            "4",
            "--chunk-size",
            "8",
        ]
    )

    assert rc == exporter.EXIT_OK
    shards = sorted(dest.glob("shard-*.msgpack.gz"))
    # 10 records / 4 per shard = 2 full + 1 tail = 3 shards
    assert len(shards) == 3


def test_shard_payload_msgpack_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "src.lmdb"
    src.mkdir()
    _populate_lmdb(src, n=6)
    dest = tmp_path / "shards"

    rc = exporter.main(
        [
            "--lmdb",
            str(src),
            "--dest",
            str(dest),
            "--shard-size",
            "3",
            "--chunk-size",
            "3",
        ]
    )
    assert rc == exporter.EXIT_OK

    shards = sorted(dest.glob("shard-*.msgpack.gz"))
    assert len(shards) == 2

    total_records = 0
    for shard in shards:
        with gzip.open(shard, "rb") as fh:
            payload = msgpack.unpackb(fh.read(), raw=False)
        assert payload["n_records"] == len(payload["records"])
        total_records += payload["n_records"]
        for rec in payload["records"]:
            for key in (
                "schema_version",
                "timestamp",
                "vision_features",
                "distance_m",
                "motor_state",
                "action",
                "reward",
            ):
                assert key in rec
    assert total_records == 6


def test_empty_lmdb_writes_no_shards(tmp_path: Path) -> None:
    """Empty LMDB store → 0 shards + clean exit (matches Phase 2 reader semantics)."""
    src = tmp_path / "src.lmdb"
    src.mkdir()  # empty dir, no data.mdb yet
    dest = tmp_path / "shards"

    rc = exporter.main(["--lmdb", str(src), "--dest", str(dest)])

    assert rc == exporter.EXIT_OK
    # Reader's empty-DB warning path: dest is created (real run), but holds nothing.
    if dest.exists():
        assert not list(dest.glob("shard-*.msgpack.gz"))
