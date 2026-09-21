"""Safety journal writer — convenience wrapper for structured safety entries.

Writes structured safety-state entries (e-stop triggers, sensor failures,
proximity events) through the existing ``JournalProtocol``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mousedroid.harness.journal.protocol import JournalEntry
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.harness.journal.protocol import JournalProtocol

_log = get_logger(__name__)


class SafetyJournalWriter:
    """Convenience wrapper that writes structured safety-state journal entries.

    Encapsulates the category and severity tagging so callers in the
    orchestrator / safety monitor don't need to know the JournalEntry
    field contract.

    Args:
        journal: The backing ``JournalProtocol`` instance. When ``None``,
            all write calls are no-ops (the writer degrades silently —
            matching the harness's null-journal pattern).
    """

    def __init__(self, *, journal: JournalProtocol | None = None) -> None:
        self._journal = journal

    async def record_estop(
        self,
        *,
        reason: str,
        clearance_m: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record an emergency stop event.

        Args:
            reason: Human-readable reason for the e-stop.
            clearance_m: Forward clearance at trigger time (if available).
            payload: Additional context data.
        """
        data: dict[str, Any] = {"reason": reason}
        if clearance_m is not None:
            data["clearance_m"] = clearance_m
        if payload:
            data.update(payload)

        entry = JournalEntry(
            event="estop_triggered",
            phase="safety",
            category="safety_state",
            severity="critical",
            payload=data,
        )
        await self._append(entry)
        _log.info("safety_journal_estop", reason=reason)

    async def record_sensor_failure(
        self,
        *,
        sensor: str,
        detail: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record a sensor failure event.

        Args:
            sensor: Name of the failed sensor.
            detail: Failure detail message.
            payload: Additional context data.
        """
        data: dict[str, Any] = {"sensor": sensor, "detail": detail}
        if payload:
            data.update(payload)

        entry = JournalEntry(
            event="sensor_failure",
            phase="safety",
            category="safety_state",
            severity="warning",
            payload=data,
        )
        await self._append(entry)
        _log.info("safety_journal_sensor_failure", sensor=sensor)

    async def record_proximity_event(
        self,
        *,
        clearance_m: float,
        threshold_m: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record a proximity warning event.

        Args:
            clearance_m: Current forward clearance in metres.
            threshold_m: Safety threshold that was breached.
            payload: Additional context data.
        """
        data: dict[str, Any] = {
            "clearance_m": clearance_m,
            "threshold_m": threshold_m,
        }
        if payload:
            data.update(payload)

        entry = JournalEntry(
            event="proximity_warning",
            phase="safety",
            category="safety_state",
            severity="warning",
            payload=data,
        )
        await self._append(entry)
        _log.debug(
            "safety_journal_proximity",
            clearance_m=clearance_m,
            threshold_m=threshold_m,
        )

    async def record_mission_transition(
        self,
        *,
        from_state: str,
        to_state: str,
        mission_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record a mission lifecycle state transition.

        Args:
            from_state: Previous lifecycle state.
            to_state: New lifecycle state.
            mission_id: Identifier of the mission.
            payload: Additional context data.
        """
        data: dict[str, Any] = {
            "from_state": from_state,
            "to_state": to_state,
            "mission_id": mission_id,
        }
        if payload:
            data.update(payload)

        entry = JournalEntry(
            event="mission_transition",
            phase="mission",
            category="mission",
            severity="info",
            payload=data,
        )
        await self._append(entry)
        _log.info(
            "safety_journal_mission_transition",
            from_state=from_state,
            to_state=to_state,
        )

    async def _append(self, entry: JournalEntry) -> None:
        """Append entry to journal, degrading silently on None journal."""
        if self._journal is None:
            return
        await self._journal.append(entry)


__all__ = ["SafetyJournalWriter"]
