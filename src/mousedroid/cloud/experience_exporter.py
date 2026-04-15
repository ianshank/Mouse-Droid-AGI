"""LMDB to GCS batch exporter — uploads experience shards on a periodic schedule.

Runs as a background ``asyncio.Task``.  Reads new records from the local
LMDB database since the last high-water mark, batches them into
msgpack-encoded shard files, and uploads to Cloud Storage with date-hour
partitioning.

GCS path layout::

    gs://{bucket}/{prefix}/{robot_id}/{YYYY-MM-DD}/{HH}/shard_{uuid}.msgpack.gz
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import io
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import lmdb
import msgpack

from mousedroid.cloud._auth import resolve_credentials
from mousedroid.logging.setup import get_logger
from mousedroid.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError

if TYPE_CHECKING:
    from mousedroid.config.schema import ExperienceConfig, GCPConfig

_log = get_logger(__name__)

_HWM_FILENAME = ".gcs_hwm"


class CloudExperienceExporter:
    """Batch exporter: LMDB experience records to GCS shards.

    Args:
        gcp_cfg: GCP configuration.
        exp_cfg: Experience configuration (LMDB path and settings).
    """

    def __init__(
        self,
        gcp_cfg: GCPConfig,
        exp_cfg: ExperienceConfig,
    ) -> None:
        self._gcp_cfg = gcp_cfg
        self._storage_cfg = gcp_cfg.storage
        self._robot_id = gcp_cfg.robot_id
        self._lmdb_path = Path(exp_cfg.path)
        self._batch_size = self._storage_cfg.upload_batch_size
        self._interval_s = self._storage_cfg.upload_interval_s
        self._compression = self._storage_cfg.compression

        self._cb = CircuitBreaker("gcp_storage", gcp_cfg.circuit_breaker)
        self._gcs_client: object | None = None
        self._gcs_bucket: object | None = None
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Open GCS client and begin the periodic export loop."""
        from google.cloud import storage as gcs

        creds, _project = resolve_credentials(self._gcp_cfg)
        self._gcs_client = gcs.Client(credentials=creds, project=self._gcp_cfg.project_id)
        self._gcs_bucket = self._gcs_client.bucket(self._storage_cfg.bucket)  # type: ignore[union-attr]
        self._running = True
        self._task = asyncio.create_task(self._export_loop())
        _log.info(
            "cloud_experience_exporter_started",
            bucket=self._storage_cfg.bucket,
            interval_s=self._interval_s,
        )

    async def export_pending(self) -> int:
        """Export pending experience records to GCS.

        Returns:
            Number of records exported in this batch.
        """
        if not self._lmdb_path.exists():
            _log.debug("cloud_exporter_lmdb_not_found", path=str(self._lmdb_path))
            return 0

        hwm = self._read_hwm()
        records: list[bytes] = []

        env = lmdb.open(str(self._lmdb_path), readonly=True, lock=False)
        try:
            with env.begin() as txn:
                cursor = txn.cursor()
                if hwm is not None:
                    if not cursor.set_range(hwm):
                        return 0
                    # Skip the HWM key itself (already exported)
                    if cursor.key() == hwm and not cursor.next():
                        return 0
                else:
                    if not cursor.first():
                        return 0

                last_key: bytes | None = None
                count = 0
                while True:
                    records.append(bytes(cursor.value()))
                    last_key = bytes(cursor.key())
                    count += 1
                    if count >= self._batch_size:
                        break
                    if not cursor.next():
                        break
        finally:
            env.close()

        if not records or last_key is None:
            return 0

        shard_data = self._build_shard(records)
        uploaded = await self._upload_shard(shard_data)
        if uploaded:
            self._write_hwm(last_key)
            _log.info("cloud_experience_exported", count=len(records))
            return len(records)
        return 0

    async def close(self) -> None:
        """Stop the export loop and release resources."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._gcs_client = None
        self._gcs_bucket = None
        _log.info("cloud_experience_exporter_closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _export_loop(self) -> None:
        """Periodic export loop — runs as a background task."""
        while self._running:
            try:
                await asyncio.sleep(self._interval_s)
                await self.export_pending()
            except asyncio.CancelledError:
                break
            except Exception:
                _log.warning("cloud_export_loop_error", exc_info=True)

    def _build_shard(self, records: list[bytes]) -> bytes:
        """Build a msgpack-encoded shard from raw record bytes.

        Args:
            records: List of msgpack-serialised experience record bytes.

        Returns:
            Shard file contents (optionally compressed).
        """
        payload: bytes = msgpack.packb(records)

        if self._compression == "gzip":
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                gz.write(payload)
            return buf.getvalue()
        if self._compression == "zstd":
            try:
                import zstandard as zstd

                cctx = zstd.ZstdCompressor()
                return cctx.compress(payload)
            except ImportError:
                _log.warning("zstd_not_available_falling_back_to_gzip")
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                    gz.write(payload)
                return buf.getvalue()
        return payload

    async def _upload_shard(self, data: bytes) -> bool:
        """Upload a shard to GCS with circuit-breaker protection.

        Args:
            data: Shard file contents.

        Returns:
            True if upload succeeded.
        """
        if self._gcs_bucket is None:
            return False

        now = datetime.now(tz=timezone.utc)
        ext = ".msgpack"
        if self._compression == "gzip":
            ext = ".msgpack.gz"
        elif self._compression == "zstd":
            ext = ".msgpack.zst"

        blob_path = (
            f"{self._storage_cfg.prefix}/{self._robot_id}/"
            f"{now.strftime('%Y-%m-%d')}/{now.strftime('%H')}/"
            f"shard_{uuid.uuid4().hex}{ext}"
        )

        try:

            async def _do_upload() -> None:
                loop = asyncio.get_running_loop()
                blob = self._gcs_bucket.blob(blob_path)  # type: ignore[union-attr]
                await loop.run_in_executor(None, blob.upload_from_string, data)

            await self._cb.call(_do_upload)
            _log.debug("cloud_shard_uploaded", path=blob_path, size_bytes=len(data))
            return True
        except CircuitOpenError:
            _log.debug("cloud_storage_circuit_open")
            return False
        except Exception:
            _log.warning("cloud_shard_upload_failed", path=blob_path, exc_info=True)
            return False

    def _hwm_path(self) -> Path:
        """Return the path to the high-water-mark file."""
        return self._lmdb_path / _HWM_FILENAME

    def _read_hwm(self) -> bytes | None:
        """Read the last-exported LMDB key from the high-water-mark file."""
        path = self._hwm_path()
        if path.exists():
            return path.read_bytes()
        return None

    def _write_hwm(self, key: bytes) -> None:
        """Write the high-water-mark key after a successful export."""
        self._hwm_path().write_bytes(key)
