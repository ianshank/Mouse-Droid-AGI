"""Experience logger — LMDB-backed experience storage."""

from __future__ import annotations

import math
import struct
import time
from pathlib import Path
from typing import TYPE_CHECKING

import lmdb

from mousedroid.constants import GB_TO_BYTES
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ExperienceConfig

_log = get_logger(__name__)


class ExperienceLogger:
    """LMDB-backed experience logger for MouseDroid.

    Writes experience records to LMDB with auto-flushing.
    """

    def __init__(self, cfg: ExperienceConfig) -> None:
        """Initialise experience logger.

        Args:
            cfg: Experience storage configuration.
        """
        self._cfg = cfg
        self._path = Path(cfg.path)
        self._flush_every_n = cfg.flush_every_n
        self._map_size = max(1, math.ceil(cfg.map_size_gb * GB_TO_BYTES))
        self._env: lmdb.Environment | None = None
        self._write_count = 0
        self._sequence = 0

    def open(self) -> None:
        """Open LMDB environment."""
        self._path.mkdir(parents=True, exist_ok=True)
        self._env = lmdb.open(
            str(self._path),
            map_size=self._map_size,
            max_dbs=1,
        )
        _log.info("experience_logger_opened", path=str(self._path))

    def close(self) -> None:
        """Close LMDB environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
            _log.info("experience_logger_closed")

    def log(self, record: MouseDroidExperienceRecord) -> None:
        """Write an experience record to LMDB.

        Args:
            record: Experience record to store.
        """
        if self._env is None:
            _log.warning("experience_logger_not_open")
            return

        key = self._make_key()
        data = record.serialize()

        with self._env.begin(write=True) as txn:
            txn.put(key, data)

        self._write_count += 1
        if self._write_count >= self._flush_every_n:
            self._env.sync()
            self._write_count = 0
            _log.debug("experience_flushed", sequence=self._sequence)

    def read(self, key: bytes) -> MouseDroidExperienceRecord | None:
        """Read an experience record by key.

        Args:
            key: LMDB key bytes.

        Returns:
            Deserialized record or None if not found.
        """
        if self._env is None:
            return None

        with self._env.begin() as txn:
            data = txn.get(key)
            if data is None:
                return None
            return MouseDroidExperienceRecord.deserialize(bytes(data))

    def count(self) -> int:
        """Return number of records in the database.

        Returns:
            Record count.
        """
        if self._env is None:
            return 0
        with self._env.begin() as txn:
            entries: int = txn.stat()["entries"]
            return entries

    def _make_key(self) -> bytes:
        """Generate monotonically increasing key.

        Returns:
            8-byte key from timestamp + sequence.
        """
        self._sequence += 1
        ts = int(time.time() * 1_000_000)
        return struct.pack(">Q", ts + self._sequence)
