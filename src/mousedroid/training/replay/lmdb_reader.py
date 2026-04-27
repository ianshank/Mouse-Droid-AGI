"""Chunked, schema-version-guarded LMDB replay reader.

This reader streams records from an LMDB experience database in fixed-size
chunks rather than loading every record into RAM. It is the foundational
component for Phase 2 of the Physical AI roadmap (Real-Episode Replay Loop).

Design invariants
-----------------
* Reader never loads the full database into RAM — all access is via a cursor
  iterating ``chunk_size`` records at a time.
* Schema-version mismatches are reported via a counter (`stats.schema_mismatches`)
  and surfaced as :class:`SchemaVersionMismatchError` when ``strict=True``.
* The async streaming wrapper offloads blocking LMDB cursor work to a thread
  pool via :func:`asyncio.to_thread`, keeping the event loop responsive.
* All thresholds and sizes are caller-injected — no hardcoded magic numbers.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import lmdb
import structlog

from mousedroid.experience.record import (
    SCHEMA_VERSION,
    MouseDroidExperienceRecord,
    deserialize_any,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import ExperienceConfig

_log = structlog.get_logger(__name__)

GB_TO_BYTES = 1024**3  # hardcoded-ok: SI/binary unit conversion constant


class SchemaVersionMismatchError(ValueError):
    """Raised when an LMDB record's ``schema_version`` does not match the runtime version."""

    def __init__(self, expected: int, actual: int | None) -> None:
        super().__init__(
            f"LMDB record schema_version mismatch: expected={expected} actual={actual}"
        )
        self.expected = expected
        self.actual = actual


@dataclass
class ReplayReaderStats:
    """Lightweight metrics surface for the streaming reader.

    These fields back the ``replay_records_consumed_total`` and
    ``replay_schema_mismatch_total`` Prometheus counters when metrics are enabled.
    """

    records_consumed: int = 0
    chunks_yielded: int = 0
    schema_mismatches: int = 0
    schema_mismatch_versions: dict[int, int] = field(default_factory=dict)


class LmdbReplayReader:
    """Streaming reader over an LMDB experience database.

    Args:
        path: LMDB environment path on disk.
        map_size_gb: Maximum memory-map size in GiB (read-only, but lmdb requires
            an upper bound).
        chunk_size: Number of records to deserialize per yielded chunk. Must be
            positive. ``64`` is a reasonable default for current hardware.
        strict_schema: When True, raise :class:`SchemaVersionMismatchError` on
            any version mismatch. When False, skip mismatched records and
            increment the counter — caller can introspect ``self.stats``.

    Backwards compatibility:
        This class is **only** invoked when callers explicitly opt in
        (e.g., ``TrainingReplayConfig.enabled=True``). Default code paths
        continue to use :class:`mousedroid.experience.dataset.OfflineRLDataset`.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        map_size_gb: float,
        chunk_size: int,
        strict_schema: bool = False,
    ) -> None:
        if chunk_size <= 0:
            msg = f"chunk_size must be positive, got {chunk_size}"
            raise ValueError(msg)
        if map_size_gb <= 0:
            msg = f"map_size_gb must be positive, got {map_size_gb}"
            raise ValueError(msg)

        self._path = Path(path)
        self._map_size = max(1, math.ceil(map_size_gb * GB_TO_BYTES))
        self._chunk_size = chunk_size
        self._strict_schema = strict_schema
        self._env: lmdb.Environment | None = None
        self.stats = ReplayReaderStats()

    @classmethod
    def from_config(
        cls,
        experience_cfg: ExperienceConfig,
        *,
        chunk_size: int,
        source_path: str | None = None,
        strict_schema: bool = False,
    ) -> LmdbReplayReader:
        """Construct a reader from an :class:`ExperienceConfig`.

        Args:
            experience_cfg: Experience storage config (provides ``path`` and ``map_size_gb``).
            chunk_size: Records per streaming chunk.
            source_path: Optional override of the LMDB path. Falls back to
                ``experience_cfg.path`` when ``None``.
            strict_schema: See :class:`LmdbReplayReader`.

        Returns:
            New reader. Caller must invoke :meth:`open` before iterating.
        """
        return cls(
            path=source_path or experience_cfg.path,
            map_size_gb=experience_cfg.map_size_gb,
            chunk_size=chunk_size,
            strict_schema=strict_schema,
        )

    def open(self) -> None:
        """Open the underlying LMDB environment in read-only mode."""
        if self._env is not None:
            return
        if not self._path.exists():
            msg = f"LMDB replay path not found: {self._path}"
            raise FileNotFoundError(msg)
        self._env = lmdb.open(
            str(self._path),
            map_size=self._map_size,
            readonly=True,
            lock=False,
        )
        _log.info(
            "lmdb_replay_reader_opened",
            path=str(self._path),
            chunk_size=self._chunk_size,
            strict_schema=self._strict_schema,
        )

    def close(self) -> None:
        """Close the LMDB environment if open."""
        if self._env is not None:
            self._env.close()
            self._env = None
            _log.info("lmdb_replay_reader_closed", path=str(self._path))

    def __enter__(self) -> LmdbReplayReader:
        """Open the environment and return ``self`` for use in a ``with`` block."""
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the LMDB environment on context-manager exit."""
        self.close()

    def __len__(self) -> int:
        """Return the number of records stored in the database."""
        if self._env is None:
            return 0
        with self._env.begin() as txn:
            entries: int = txn.stat()["entries"]
            return entries

    # ------------------------------------------------------------------
    # Sync streaming API
    # ------------------------------------------------------------------
    def stream_chunks(self) -> Iterator[list[MouseDroidExperienceRecord]]:
        """Yield successive chunks of deserialized experience records.

        Records are yielded in LMDB key order, which matches insertion order
        for the existing :class:`mousedroid.experience.logger.ExperienceLogger`.

        Yields:
            Lists of length up to ``chunk_size``. The final chunk may be shorter.

        Raises:
            RuntimeError: If the reader is not open.
            SchemaVersionMismatchError: If ``strict_schema=True`` and a record
                has an unsupported schema version.
        """
        if self._env is None:
            msg = "LmdbReplayReader.stream_chunks called before open()"
            raise RuntimeError(msg)

        chunk: list[MouseDroidExperienceRecord] = []
        with self._env.begin() as txn:
            cursor = txn.cursor()
            for _key, raw in cursor:
                record = self._safe_deserialize(bytes(raw))
                if record is None:
                    continue
                chunk.append(record)
                if len(chunk) >= self._chunk_size:
                    self.stats.records_consumed += len(chunk)
                    self.stats.chunks_yielded += 1
                    yield chunk
                    chunk = []

        if chunk:
            self.stats.records_consumed += len(chunk)
            self.stats.chunks_yielded += 1
            yield chunk

    def stream_records(self) -> Iterator[MouseDroidExperienceRecord]:
        """Yield individual records (chunked internally for efficiency)."""
        for chunk in self.stream_chunks():
            yield from chunk

    # ------------------------------------------------------------------
    # Async streaming API
    # ------------------------------------------------------------------
    async def stream_chunks_async(self) -> AsyncIterator[list[MouseDroidExperienceRecord]]:
        """Async wrapper around :meth:`stream_chunks`.

        Each chunk is materialized in a worker thread via :func:`asyncio.to_thread`
        so the event loop is not blocked by LMDB cursor work.

        Yields:
            Same chunks as :meth:`stream_chunks`.
        """
        iterator = self.stream_chunks()
        sentinel: object = object()

        def _next() -> list[MouseDroidExperienceRecord] | object:
            try:
                return next(iterator)
            except StopIteration:
                return sentinel

        while True:
            chunk = await asyncio.to_thread(_next)
            if chunk is sentinel:
                return
            assert isinstance(chunk, list)  # narrow for mypy
            yield chunk

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _safe_deserialize(self, raw: bytes) -> MouseDroidExperienceRecord | None:
        """Deserialize one record, honouring the schema-version policy."""
        try:
            return deserialize_any(raw)
        except ValueError as exc:
            actual = self._extract_version(exc)
            self.stats.schema_mismatches += 1
            self.stats.schema_mismatch_versions[actual or -1] = (
                self.stats.schema_mismatch_versions.get(actual or -1, 0) + 1
            )
            _log.warning(
                "lmdb_replay_schema_mismatch",
                expected=SCHEMA_VERSION,
                actual=actual,
                path=str(self._path),
            )
            if self._strict_schema:
                raise SchemaVersionMismatchError(SCHEMA_VERSION, actual) from exc
            return None

    @staticmethod
    def _extract_version(exc: ValueError) -> int | None:
        """Best-effort extraction of the offending schema version from an error message."""
        message = str(exc)
        # Messages look like ``Unknown schema version: <int|None>``
        marker = "Unknown schema version:"
        if marker in message:
            tail = message.split(marker, 1)[1].strip()
            if tail.startswith("None"):
                return None
            try:
                return int(tail.split()[0])
            except (ValueError, IndexError):
                return None
        return None
