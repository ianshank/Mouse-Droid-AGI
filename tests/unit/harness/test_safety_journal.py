"""Unit tests — SafetyJournalWriter.

Tests the convenience wrapper for writing structured safety journal
entries through the JournalProtocol.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from mousedroid.harness.journal.protocol import JournalEntry
from mousedroid.harness.journal.safety_journal import SafetyJournalWriter


class _FakeJournal:
    """In-memory journal for testing."""

    def __init__(self) -> None:
        self.entries: list[JournalEntry] = []
        self._running = True

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def append(self, entry: JournalEntry) -> None:
        self.entries.append(entry)

    async def read_all(self) -> AsyncIterator[JournalEntry]:
        for entry in self.entries:
            yield entry

    @property
    def is_running(self) -> bool:
        return self._running


@pytest.fixture
def journal() -> _FakeJournal:
    """Fresh in-memory journal."""
    return _FakeJournal()


@pytest.fixture
def writer(journal: _FakeJournal) -> SafetyJournalWriter:
    """SafetyJournalWriter with fake journal."""
    return SafetyJournalWriter(journal=journal)


@pytest.mark.asyncio
async def test_record_estop(writer: SafetyJournalWriter, journal: _FakeJournal) -> None:
    """E-stop events are logged with correct category and severity."""
    await writer.record_estop(reason="obstacle_detected", clearance_m=0.05)
    assert len(journal.entries) == 1
    entry = journal.entries[0]
    assert entry.event == "estop_triggered"
    assert entry.category == "safety_state"
    assert entry.severity == "critical"
    assert entry.payload["reason"] == "obstacle_detected"
    assert entry.payload["clearance_m"] == 0.05


@pytest.mark.asyncio
async def test_record_sensor_failure(writer: SafetyJournalWriter, journal: _FakeJournal) -> None:
    """Sensor failure events are logged with correct fields."""
    await writer.record_sensor_failure(sensor="lidar", detail="no data")
    assert len(journal.entries) == 1
    entry = journal.entries[0]
    assert entry.event == "sensor_failure"
    assert entry.category == "safety_state"
    assert entry.severity == "warning"
    assert entry.payload["sensor"] == "lidar"


@pytest.mark.asyncio
async def test_record_proximity_event(writer: SafetyJournalWriter, journal: _FakeJournal) -> None:
    """Proximity events are logged with clearance and threshold."""
    await writer.record_proximity_event(clearance_m=0.1, threshold_m=0.3)
    assert len(journal.entries) == 1
    entry = journal.entries[0]
    assert entry.event == "proximity_warning"
    assert entry.payload["clearance_m"] == 0.1
    assert entry.payload["threshold_m"] == 0.3


@pytest.mark.asyncio
async def test_record_mission_transition(writer: SafetyJournalWriter, journal: _FakeJournal) -> None:
    """Mission transitions are logged with state info."""
    await writer.record_mission_transition(
        from_state="PENDING",
        to_state="RUNNING",
        mission_id="m-123",
    )
    assert len(journal.entries) == 1
    entry = journal.entries[0]
    assert entry.event == "mission_transition"
    assert entry.category == "mission"
    assert entry.payload["from_state"] == "PENDING"
    assert entry.payload["to_state"] == "RUNNING"


@pytest.mark.asyncio
async def test_none_journal_noop() -> None:
    """Writer with None journal silently no-ops."""
    writer = SafetyJournalWriter(journal=None)
    # Must not raise
    await writer.record_estop(reason="test")
    await writer.record_sensor_failure(sensor="test")
    await writer.record_proximity_event(clearance_m=0.1, threshold_m=0.3)
    await writer.record_mission_transition(from_state="A", to_state="B")
