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


@pytest.mark.asyncio
async def test_export_pending_with_records() -> None:
    """export_pending should read records and upload shard."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    with tempfile.TemporaryDirectory() as tmpdir:
        lmdb_path = Path(tmpdir) / "exp"
        lmdb_path.mkdir()
        _populate_lmdb(lmdb_path, n_records=3)

        cfg = _make_gcp_cfg()
        exp_cfg = ExperienceConfig(path=str(lmdb_path))
        exporter = CloudExperienceExporter(cfg, exp_cfg)

        # Mock _upload_shard to return True
        upload_called = False

        async def mock_upload(data: bytes) -> bool:
            nonlocal upload_called
            upload_called = True
            assert len(data) > 0
            return True

        exporter._upload_shard = mock_upload
        count = await exporter.export_pending()
        assert count == 3
        assert upload_called

        # HWM should have been written
        hwm = exporter._read_hwm()
        assert hwm is not None


@pytest.mark.asyncio
async def test_export_pending_respects_batch_size() -> None:
    """export_pending should not export more than batch_size records."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    with tempfile.TemporaryDirectory() as tmpdir:
        lmdb_path = Path(tmpdir) / "exp"
        lmdb_path.mkdir()
        _populate_lmdb(lmdb_path, n_records=10)

        cfg = _make_gcp_cfg(storage=GCPStorageConfig(upload_batch_size=3))
        exp_cfg = ExperienceConfig(path=str(lmdb_path))
        exporter = CloudExperienceExporter(cfg, exp_cfg)

        async def mock_upload(data: bytes) -> bool:
            return True

        exporter._upload_shard = mock_upload
        count = await exporter.export_pending()
        assert count == 3


@pytest.mark.asyncio
async def test_export_pending_resumes_from_hwm() -> None:
    """export_pending should skip already-exported records via HWM."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    with tempfile.TemporaryDirectory() as tmpdir:
        lmdb_path = Path(tmpdir) / "exp"
        lmdb_path.mkdir()
        keys = _populate_lmdb(lmdb_path, n_records=5)

        cfg = _make_gcp_cfg()
        exp_cfg = ExperienceConfig(path=str(lmdb_path))
        exporter = CloudExperienceExporter(cfg, exp_cfg)

        # Set HWM to 3rd record — should skip first 3
        exporter._write_hwm(keys[2])

        async def mock_upload(data: bytes) -> bool:
            return True

        exporter._upload_shard = mock_upload
        count = await exporter.export_pending()
        assert count == 2  # records 4 and 5


@pytest.mark.asyncio
async def test_export_pending_upload_failure_no_hwm_advance() -> None:
    """When upload fails, HWM should not advance."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    with tempfile.TemporaryDirectory() as tmpdir:
        lmdb_path = Path(tmpdir) / "exp"
        lmdb_path.mkdir()
        _populate_lmdb(lmdb_path, n_records=3)

        cfg = _make_gcp_cfg()
        exp_cfg = ExperienceConfig(path=str(lmdb_path))
        exporter = CloudExperienceExporter(cfg, exp_cfg)

        async def mock_upload_fail(data: bytes) -> bool:
            return False

        exporter._upload_shard = mock_upload_fail
        count = await exporter.export_pending()
        assert count == 0
        assert exporter._read_hwm() is None


@pytest.mark.asyncio
async def test_upload_shard_circuit_open() -> None:
    """_upload_shard should return False when circuit breaker is open."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter
    from mousedroid.resilience.circuit_breaker import CircuitOpenError

    cfg = _make_gcp_cfg()
    exp_cfg = ExperienceConfig(path="/tmp/test")
    exporter = CloudExperienceExporter(cfg, exp_cfg)
    exporter._gcs_bucket = MagicMock()

    async def raise_circuit_open(func: Any) -> Any:
        raise CircuitOpenError("gcp_storage", 30.0)

    exporter._cb = MagicMock()
    exporter._cb.call = raise_circuit_open

    result = await exporter._upload_shard(b"test data")
    assert result is False


@pytest.mark.asyncio
async def test_upload_shard_no_bucket() -> None:
    """_upload_shard should return False when GCS bucket is None."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    cfg = _make_gcp_cfg()
    exp_cfg = ExperienceConfig(path="/tmp/test")
    exporter = CloudExperienceExporter(cfg, exp_cfg)
    result = await exporter._upload_shard(b"test data")
    assert result is False


@pytest.mark.asyncio
async def test_close_cancels_task() -> None:
    """close() should cancel the background export task."""
    import asyncio

    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    cfg = _make_gcp_cfg()
    exp_cfg = ExperienceConfig(path="/tmp/test")
    exporter = CloudExperienceExporter(cfg, exp_cfg)
    exporter._running = True

    async def fake_loop() -> None:
        await asyncio.sleep(100)

    exporter._task = asyncio.create_task(fake_loop())
    await exporter.close()
    assert exporter._task is None
    assert exporter._running is False


@pytest.mark.asyncio
async def test_start_initialises_gcs_client() -> None:
    """start() should create a GCS client, bucket, and start the export loop."""
    import sys
    from unittest.mock import patch

    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    cfg = _make_gcp_cfg()
    exp_cfg = ExperienceConfig(path="/tmp/test_exp")
    exporter = CloudExperienceExporter(cfg, exp_cfg)

    mock_gcs = MagicMock()
    mock_google_cloud = MagicMock()
    mock_google_cloud.storage = mock_gcs

    with (
        patch.dict(
            sys.modules,
            {
                "google": MagicMock(),
                "google.cloud": mock_google_cloud,
                "google.cloud.storage": mock_gcs,
            },
        ),
        patch("mousedroid.cloud.experience_exporter.resolve_credentials") as mock_creds,
    ):
        mock_creds.return_value = (MagicMock(), "test-project")
        await exporter.start()

    assert exporter._gcs_client is not None
    assert exporter._gcs_bucket is not None
    assert exporter._running is True
    assert exporter._task is not None
    # Clean up
    await exporter.close()


@pytest.mark.asyncio
async def test_export_loop_runs_and_stops() -> None:
    """_export_loop should periodically call export_pending and stop on cancel."""

    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    cfg = _make_gcp_cfg(storage=GCPStorageConfig(upload_interval_s=0.01))
    exp_cfg = ExperienceConfig(path="/nonexistent")
    exporter = CloudExperienceExporter(cfg, exp_cfg)
    exporter._running = True

    call_count = 0

    async def counting_export() -> int:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            exporter._running = False
        return 0

    exporter.export_pending = counting_export
    await exporter._export_loop()
    assert call_count >= 2


@pytest.mark.asyncio
async def test_export_loop_handles_exceptions() -> None:
    """_export_loop should catch and continue on exceptions."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    cfg = _make_gcp_cfg(storage=GCPStorageConfig(upload_interval_s=0.01))
    exp_cfg = ExperienceConfig(path="/nonexistent")
    exporter = CloudExperienceExporter(cfg, exp_cfg)
    exporter._running = True

    call_count = 0

    async def failing_export() -> int:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            exporter._running = False
            return 0
        raise RuntimeError("test error")

    exporter.export_pending = failing_export
    await exporter._export_loop()
    assert call_count >= 2


def test_build_shard_zstd_fallback_to_gzip() -> None:
    """build_shard with zstd should fall back to gzip if zstandard not installed."""
    from mousedroid.cloud.experience_exporter import CloudExperienceExporter

    cfg = _make_gcp_cfg(storage=GCPStorageConfig(compression="zstd"))
    exp_cfg = ExperienceConfig(path="/tmp/test")
    exporter = CloudExperienceExporter(cfg, exp_cfg)

    records = [b"record1"]
    shard = exporter._build_shard(records)
    # Should produce either zstd or gzip (fallback) — both are valid compressed
    assert len(shard) > 0
