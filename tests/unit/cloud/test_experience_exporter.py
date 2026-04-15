"""Unit tests for CloudExperienceExporter LMDB-to-GCS batch exporter."""

from __future__ import annotations

import gzip
import struct
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import lmdb
import msgpack
import pytest

from mousedroid.config.schema import (
    CircuitBreakerConfig,
    ExperienceConfig,
    GCPConfig,
    GCPStorageConfig,
    RetryConfig,
)
from mousedroid.experience.record import MouseDroidExperienceRecord


def _make_gcp_cfg(**overrides: Any) -> GCPConfig:
    """Create a GCPConfig with test defaults."""
    return GCPConfig(
        project_id="test-project",
        robot_id="droid-test",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_s=1.0,
            half_open_max_calls=1,
        ),
        retry=RetryConfig(max_attempts=1, base_delay_s=0.01),
        **overrides,
    )


def _populate_lmdb(path: Path, n_records: int = 5) -> list[bytes]:
    """Populate an LMDB database with test experience records."""
    env = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    keys: list[bytes] = []
    for i in range(n_records):
        record = MouseDroidExperienceRecord()
        key = struct.pack(">Q", 1000000 * (i + 1))
        with env.begin(write=True) as txn:
            txn.put(key, record.serialize())
        keys.append(key)
    env.close()
    return keys


def test_exporter_init() -> None:
    """Exporter should be constructable without starting."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    cfg = _make_gcp_cfg()
    exp_cfg = ExperienceConfig(path="/tmp/test_exp")
    exporter = CloudExperienceExporter(cfg, exp_cfg)
    assert exporter._gcs_client is None


def test_exporter_conforms_to_protocol() -> None:
    """CloudExperienceExporter should satisfy CloudExperienceExporterProtocol."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter
    from mousedroid.cloud.protocol import CloudExperienceExporterProtocol

    cfg = _make_gcp_cfg()
    exp_cfg = ExperienceConfig(path="/tmp/test_exp")
    exporter = CloudExperienceExporter(cfg, exp_cfg)
    assert isinstance(exporter, CloudExperienceExporterProtocol)


def test_build_shard_gzip() -> None:
    """build_shard with gzip compression produces valid gzipped msgpack."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    cfg = _make_gcp_cfg(
        storage=GCPStorageConfig(compression="gzip"),
    )
    exp_cfg = ExperienceConfig(path="/tmp/test_exp")
    exporter = CloudExperienceExporter(cfg, exp_cfg)

    records = [b"record1", b"record2", b"record3"]
    shard = exporter._build_shard(records)

    # Should be valid gzip
    decompressed = gzip.decompress(shard)
    unpacked = msgpack.unpackb(decompressed)
    assert unpacked == [b"record1", b"record2", b"record3"]


def test_build_shard_no_compression() -> None:
    """build_shard with no compression produces raw msgpack."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    cfg = _make_gcp_cfg(
        storage=GCPStorageConfig(compression="none"),
    )
    exp_cfg = ExperienceConfig(path="/tmp/test_exp")
    exporter = CloudExperienceExporter(cfg, exp_cfg)

    records = [b"record1", b"record2"]
    shard = exporter._build_shard(records)

    unpacked = msgpack.unpackb(shard)
    assert unpacked == [b"record1", b"record2"]


def test_hwm_read_write() -> None:
    """High-water mark should persist across read/write cycles."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_gcp_cfg()
        exp_cfg = ExperienceConfig(path=tmpdir)
        exporter = CloudExperienceExporter(cfg, exp_cfg)

        # Initially None
        assert exporter._read_hwm() is None

        # Write and read back
        test_key = struct.pack(">Q", 42)
        exporter._write_hwm(test_key)
        assert exporter._read_hwm() == test_key


@pytest.mark.asyncio
async def test_export_pending_empty_db() -> None:
    """export_pending on an empty LMDB returns 0."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    with tempfile.TemporaryDirectory() as tmpdir:
        lmdb_path = Path(tmpdir) / "exp"
        lmdb_path.mkdir()
        env = lmdb.open(str(lmdb_path), map_size=1024 * 1024)
        env.close()

        cfg = _make_gcp_cfg()
        exp_cfg = ExperienceConfig(path=str(lmdb_path))
        exporter = CloudExperienceExporter(cfg, exp_cfg)
        exporter._gcs_bucket = MagicMock()

        count = await exporter.export_pending()
        assert count == 0


@pytest.mark.asyncio
async def test_export_pending_no_lmdb() -> None:
    """export_pending when LMDB path doesn't exist returns 0."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    cfg = _make_gcp_cfg()
    exp_cfg = ExperienceConfig(path="/nonexistent/path")
    exporter = CloudExperienceExporter(cfg, exp_cfg)
    count = await exporter.export_pending()
    assert count == 0
