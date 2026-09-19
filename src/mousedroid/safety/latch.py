"""Emergency-stop latch that survives a process restart (peer review D-5).

``MouseDroidSafetyMonitor.evaluate`` recomputes ``is_emergency`` from
``False`` every tick, so the instant a triggering condition clears the next
tick resumes driving with no human in the loop. The shipped units compound
it -- ``scripts/mousedroid.service`` and ``scripts/mousedroid-docker.service``
both set ``Restart=on-failure``, and ``docker-compose.jetson.yml`` sets
``restart: unless-stopped`` -- so a fault can be cleared by a restart too.
ISO 3691-4 requires that an emergency stop is reset only by deliberate human
action.

**Hot-loop discipline.** :meth:`FileEmergencyLatch.trip` is synchronous and
touches no filesystem: ``evaluate`` runs at 30 Hz *and* is called from the
MCP tool bridge on an async request path, so a blocking write there would
land on both. Persistence, loading and re-arm are ``async`` and push their
syscalls through ``asyncio.to_thread``.

**Fail-closed on damage.** This deliberately inverts the precedent in
``learning/on_device/slot_store.py::load_active``, which returns ``None`` on
a corrupt manifest. There, "no active slot" is the safe answer. Here the safe
answer is the opposite: a corrupt or unreadable record is evidence that
*something wrote a latch* and the write or the media failed, so reading it as
"not latched" would be exactly the silent re-arm this module exists to
prevent.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from mousedroid.logging.setup import get_logger

__all__ = [
    "EmergencyLatchProtocol",
    "EmergencyLatchStoreError",
    "FileEmergencyLatch",
    "LatchRecord",
]

_log = get_logger(__name__)

LATCH_SCHEMA_VERSION: Final[int] = 1
"""Bumped only for an incompatible record shape. An unknown version latches."""

LATCH_FILENAME: Final[str] = "estop_latch.json"
_TMP_SUFFIX: Final[str] = ".tmp"

REASON_CORRUPT: Final[str] = "corrupt_latch_file"
REASON_UNREADABLE: Final[str] = "latch_store_unreadable"
REASON_UNKNOWN_VERSION: Final[str] = "unknown_latch_schema_version"


class EmergencyLatchStoreError(RuntimeError):
    """Raised when the latch record cannot be written.

    A named exception rather than an ``assert``: ruff ``S101`` blocks
    asserts in ``src/`` and ``PYTHONOPTIMIZE=1`` in ``Dockerfile.jetson``
    strips them, so a guard would become no guard on the rover.
    """


@dataclass(frozen=True, slots=True)
class LatchRecord:
    """What tripped the latch, and when."""

    latched: bool
    reason: str
    causes: tuple[str, ...] = ()
    tripped_at_unix_s: float = 0.0
    tripped_at_iso: str = ""
    tick_index: int | None = None

    def to_json_obj(self) -> dict[str, Any]:
        """Serialise for the on-disk record.

        Carries no hostname, no paths and no credentials -- CHARTER
        invariant 11.
        """
        return {
            "schema_version": LATCH_SCHEMA_VERSION,
            "latched": self.latched,
            "reason": self.reason,
            "causes": list(self.causes),
            "tripped_at_unix_s": self.tripped_at_unix_s,
            "tripped_at_iso": self.tripped_at_iso,
            "tick_index": self.tick_index,
        }


_CLEAR: Final[LatchRecord] = LatchRecord(latched=False, reason="")


@runtime_checkable
class EmergencyLatchProtocol(Protocol):
    """A latch the safety monitor can set and an operator must clear."""

    @property
    def is_latched(self) -> bool:
        """Whether an emergency stop is currently held."""
        ...

    @property
    def record(self) -> LatchRecord:
        """The current record, for logs and the operator CLI."""
        ...

    def trip(
        self, reason: str, *, causes: tuple[str, ...] = (), tick_index: int | None = None
    ) -> bool:
        """Latch. Synchronous and allocation-light -- runs at 30 Hz.

        Returns:
            ``True`` if this call was the transition, ``False`` if already
            latched. Lets the caller log the trip exactly once.
        """
        ...

    async def persist(self) -> None:
        """Write the record if it changed. Off the hot loop."""
        ...

    async def load(self) -> bool:
        """Read the record at startup. Returns whether latched."""
        ...

    async def rearm(self, *, operator: str) -> bool:
        """Clear the latch on deliberate operator action.

        Returns:
            ``True`` if a latch was cleared, ``False`` if already clear.
        """
        ...


class FileEmergencyLatch:
    """Latch backed by one small JSON file under the experience root."""

    def __init__(self, state_dir: Path, *, fsync: bool = True) -> None:
        self._path = state_dir / LATCH_FILENAME
        self._fsync = fsync
        self._record = _CLEAR
        self._dirty = False

    @property
    def is_latched(self) -> bool:
        """Whether an emergency stop is currently held."""
        return self._record.latched

    @property
    def record(self) -> LatchRecord:
        """The current record."""
        return self._record

    @property
    def path(self) -> Path:
        """Where the record lives, for the operator CLI and preflight."""
        return self._path

    def trip(
        self, reason: str, *, causes: tuple[str, ...] = (), tick_index: int | None = None
    ) -> bool:
        """Latch in memory. No file I/O -- see the module docstring."""
        if self._record.latched:
            return False
        now = time.time()
        self._record = LatchRecord(
            latched=True,
            reason=reason,
            causes=causes,
            tripped_at_unix_s=now,
            tripped_at_iso=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            tick_index=tick_index,
        )
        self._dirty = True
        return True

    async def persist(self) -> None:
        """Write the record when it changed, atomically and durably."""
        if not self._dirty:
            return
        await asyncio.to_thread(self._write_sync, self._record)
        self._dirty = False
        _log.error(
            "estop_latch_persisted",
            reason=self._record.reason,
            causes=list(self._record.causes),
            tripped_at_iso=self._record.tripped_at_iso,
        )

    async def load(self) -> bool:
        """Read the record at startup; fail closed on anything unexpected."""
        self._record = await asyncio.to_thread(self._read_sync)
        self._dirty = False
        return self._record.latched

    async def rearm(self, *, operator: str) -> bool:
        """Clear the latch. The only path back to motion."""
        if not self._record.latched and not self._path.exists():
            return False
        cleared = self._record
        await asyncio.to_thread(self._remove_sync)
        self._record = _CLEAR
        self._dirty = False
        _log.warning(
            "estop_latch_rearmed",
            operator=operator,
            cleared_reason=cleared.reason,
            was_tripped_at_iso=cleared.tripped_at_iso,
        )
        return True

    # -- sync halves, always called via asyncio.to_thread -------------------

    def _write_sync(self, record: LatchRecord) -> None:
        """Atomic + durable replace. See the module docstring on fsync."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + _TMP_SUFFIX)
            payload = json.dumps(record.to_json_obj(), indent=2, sort_keys=True)
            with tmp.open("w", encoding="utf-8", errors="replace") as handle:
                handle.write(payload)
                handle.flush()
                if self._fsync:
                    os.fsync(handle.fileno())
            os.replace(tmp, self._path)
            if self._fsync:
                dir_fd = os.open(self._path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        except OSError as exc:
            msg = f"could not write the emergency-stop latch to {self._path}"
            raise EmergencyLatchStoreError(msg) from exc

    def _read_sync(self) -> LatchRecord:
        """Every unexpected state resolves to LATCHED. See module docstring."""
        if not self._path.exists():
            return _CLEAR
        try:
            raw = self._path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _log.error("estop_latch_unreadable", path=str(self._path), error=str(exc))
            return LatchRecord(latched=True, reason=REASON_UNREADABLE)
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError:
            _log.error("estop_latch_corrupt", path=str(self._path))
            return LatchRecord(latched=True, reason=REASON_CORRUPT)
        if not isinstance(doc, dict):
            _log.error("estop_latch_corrupt", path=str(self._path), detail="not_an_object")
            return LatchRecord(latched=True, reason=REASON_CORRUPT)
        version = doc.get("schema_version")
        if version != LATCH_SCHEMA_VERSION:
            _log.error("estop_latch_unknown_version", found=repr(version))
            return LatchRecord(latched=True, reason=REASON_UNKNOWN_VERSION)
        latched = doc.get("latched")
        if not isinstance(latched, bool):
            _log.error("estop_latch_corrupt", path=str(self._path), detail="latched_not_bool")
            return LatchRecord(latched=True, reason=REASON_CORRUPT)
        if not latched:
            _log.info("estop_latch_tombstone", path=str(self._path))
            return _CLEAR
        causes = doc.get("causes")
        return LatchRecord(
            latched=True,
            reason=str(doc.get("reason", "")),
            causes=tuple(str(c) for c in causes) if isinstance(causes, list) else (),
            tripped_at_unix_s=float(doc.get("tripped_at_unix_s", 0.0) or 0.0),
            tripped_at_iso=str(doc.get("tripped_at_iso", "")),
            tick_index=doc.get("tick_index") if isinstance(doc.get("tick_index"), int) else None,
        )

    def _remove_sync(self) -> None:
        """Remove the record; absence is the only unlatched state."""
        self._path.unlink(missing_ok=True)


_PROTOCOL_CHECK: EmergencyLatchProtocol = FileEmergencyLatch(Path("."))
del _PROTOCOL_CHECK
