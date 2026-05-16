"""Async, chunked LMDB replay reader.

Reads :class:`~mousedroid.experience.record.MouseDroidExperienceRecord`
instances from an LMDB store in bounded chunks. Schema-mismatched records
are counted and skipped (not raised) so a stale on-disk store cannot kill
training.

The reader is async-friendly: chunk reads run inside ``asyncio.to_thread``
so the LMDB cursor never blocks the event loop. The 8 GB Jetson Orin Nano
RAM budget is respected — no full-DB load.

The LMDB environment handle is opened **once per ``stream()`` call** and
reused across every chunk; opening per-chunk would be wasteful and
contradicts LMDB's design (envs are designed to be opened once and
reused across many transactions).
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import lmdb

from mousedroid.constants import GB_TO_BYTES
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mousedroid.config.schema import ExperienceConfig
    from mousedroid.telemetry.metrics import MetricsRegistry

_log = get_logger(__name__)


@runtime_checkable
class ReplayReaderProtocol(Protocol):
    """Async, chunked replay reader contract."""

    def stream(
        self,
        chunk_size: int,
    ) -> AsyncIterator[list[MouseDroidExperienceRecord]]:
        """Yield chunks of replay records from disk."""
        ...

    @property
    def stats(self) -> dict[str, int]:
        """Return cumulative reader statistics."""
        ...


class LMDBReplayReader:
    """Stream replay records from an LMDB store in bounded chunks.

    Args:
        experience_cfg: LMDB store configuration. ``path`` is the LMDB env
            directory; ``map_size_gb`` is converted to bytes via
            :data:`mousedroid.constants.GB_TO_BYTES`.
        path_override: Optional explicit path. When ``None``,
            ``experience_cfg.path`` is used.

    Notes:
        Records whose ``schema_version`` differs from the current code's
        :data:`mousedroid.experience.record.SCHEMA_VERSION` are silently
        dropped, but counted under ``stats["skipped_schema_mismatch"]``.
        Operators tail the structured log line ``replay_schema_mismatch``
        for visibility.

        An empty or missing LMDB store is **not** an error — the reader
        logs a single ``replay_empty_db`` warning and yields no chunks.
        This makes ``--use-real-replay`` safe to enable on a fresh Jetson.
    """

    def __init__(
        self,
        experience_cfg: ExperienceConfig,
        *,
        path_override: str | None = None,
        debug_log_every_n: int = 0,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        """Initialise the reader; defer LMDB env open until ``stream()``.

        Args:
            experience_cfg: LMDB store configuration (``path`` +
                ``map_size_gb``).
            path_override: Optional explicit path; ``None`` uses
                ``experience_cfg.path``.
            debug_log_every_n: Emit a DEBUG ``replay_chunk_decoded`` event
                every N chunks; ``0`` disables debug logs entirely.
            metrics: Optional :class:`MetricsRegistry`. When provided, each
                successful decode increments
                ``mousedroid_replay_records_total{outcome="ok"}`` and each
                schema-mismatch drop increments
                ``{outcome="schema_mismatch"}`` alongside the existing
                ``replay_schema_mismatch`` structured log. ``None`` (default)
                preserves byte-identical pre-PR-A2.1 behavior.
        """
        path_str = path_override if path_override is not None else experience_cfg.path
        self._path = Path(path_str)
        self._map_size = max(1, math.ceil(experience_cfg.map_size_gb * GB_TO_BYTES))
        self._debug_log_every_n = max(0, int(debug_log_every_n))
        self._metrics = metrics
        self._read_records = 0
        self._skipped_schema = 0
        self._chunks_yielded = 0

    @property
    def stats(self) -> dict[str, int]:
        """Cumulative reader statistics (records, skipped, chunks)."""
        return {
            "read_records": self._read_records,
            "skipped_schema_mismatch": self._skipped_schema,
            "chunks_yielded": self._chunks_yielded,
        }

    @property
    def path(self) -> Path:
        """LMDB env path."""
        return self._path

    def _open_env(self) -> lmdb.Environment:
        """Open the LMDB environment in read-only, no-lock mode."""
        return lmdb.open(
            str(self._path),
            map_size=self._map_size,
            readonly=True,
            lock=False,
        )

    def _read_keys_with_env(self, env: lmdb.Environment) -> list[bytes]:
        """Snapshot the key list inside ``env``. Sync; off the loop."""
        with env.begin() as txn:
            cursor = txn.cursor()
            return [bytes(k) for k, _ in cursor]

    def _read_chunk_with_env(
        self,
        env: lmdb.Environment,
        keys: list[bytes],
    ) -> list[MouseDroidExperienceRecord]:
        """Read this chunk's records inside ``env``. Sync; off the loop."""
        out: list[MouseDroidExperienceRecord] = []
        with env.begin() as txn:
            for key in keys:
                blob = txn.get(key)
                if blob is None:
                    continue
                try:
                    record = MouseDroidExperienceRecord.deserialize(bytes(blob))
                except ValueError as exc:
                    # Schema mismatch — count, log once per occurrence,
                    # do not crash training.
                    self._skipped_schema += 1
                    if self._metrics is not None:
                        self._metrics.inc_replay_record("schema_mismatch")
                    _log.warning(
                        "replay_schema_mismatch",
                        error=str(exc),
                        cumulative_skipped=self._skipped_schema,
                    )
                    continue
                out.append(record)
                if self._metrics is not None:
                    self._metrics.inc_replay_record("ok")
        return out

    async def stream(
        self,
        chunk_size: int,
    ) -> AsyncIterator[list[MouseDroidExperienceRecord]]:
        """Yield chunks of records of size ``chunk_size`` (last chunk may be short).

        Args:
            chunk_size: Maximum records per yielded chunk. Must be positive.

        Yields:
            Lists of decoded records of length up to ``chunk_size``.

        Raises:
            ValueError: If ``chunk_size`` is not positive.

        Notes:
            The LMDB env is opened **once** at the start and held for the
            duration of the stream. Both the key snapshot and every chunk
            read share the same env handle (closed in the ``finally`` block
            on a worker thread to avoid blocking the event loop).
        """
        if chunk_size <= 0:
            msg = f"chunk_size must be positive, got {chunk_size}"
            raise ValueError(msg)

        # An LMDB env is a directory containing ``data.mdb`` + ``lock.mdb``.
        # Treat a missing dir or a dir without ``data.mdb`` as an empty store
        # so a fresh Jetson can opt-in to real replay safely.
        if not self._path.exists() or not (self._path / "data.mdb").exists():
            _log.warning("replay_empty_db", path=str(self._path))
            return

        env = await asyncio.to_thread(self._open_env)
        try:
            keys = await asyncio.to_thread(self._read_keys_with_env, env)
            if not keys:
                _log.warning("replay_empty_db", path=str(self._path))
                return

            _log.info(
                "replay_stream_open",
                path=str(self._path),
                n_keys=len(keys),
                chunk_size=chunk_size,
            )

            for start in range(0, len(keys), chunk_size):
                window = keys[start : start + chunk_size]
                chunk = await asyncio.to_thread(self._read_chunk_with_env, env, window)
                if not chunk:
                    continue
                self._read_records += len(chunk)
                self._chunks_yielded += 1
                # Throttled DEBUG-level live triage. Operator sets
                # `cfg.training.replay_mixer.debug_log_every_n` and the factory
                # forwards it via the constructor; 0 disables completely.
                if (
                    self._debug_log_every_n > 0
                    and self._chunks_yielded % self._debug_log_every_n == 0
                ):
                    _log.debug(
                        "replay_chunk_decoded",
                        chunks_yielded=self._chunks_yielded,
                        read_records=self._read_records,
                        skipped_schema=self._skipped_schema,
                        last_chunk_len=len(chunk),
                    )
                yield chunk
        finally:
            await asyncio.to_thread(env.close)
