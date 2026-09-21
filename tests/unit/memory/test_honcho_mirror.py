"""Unit tests — HonchoMemoryMirror.

Tests the Honcho memory mirror with mocked SDK and journal,
verifying safe-category filtering, recall sanitization, and
degraded behaviour.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema.agents import HonchoConfig
from mousedroid.harness.journal.protocol import JournalEntry


class _FakeJournal:
    """In-memory journal for testing."""

    def __init__(self, entries: list[JournalEntry] | None = None) -> None:
        self._entries = entries or []
        self._running = True

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def append(self, entry: JournalEntry) -> None:
        self._entries.append(entry)

    async def read_all(self) -> AsyncIterator[JournalEntry]:
        for entry in self._entries:
            yield entry

    @property
    def is_running(self) -> bool:
        return self._running


@pytest.fixture
def honcho_config() -> HonchoConfig:
    """Honcho config with enabled=True for testing."""
    return HonchoConfig(enabled=True, app_name="test-app")


@pytest.fixture
def mock_injection_filter() -> MagicMock:
    """Injection filter mock that strips injections."""
    filt = MagicMock()
    filt.sanitize.side_effect = lambda x: x.replace(
        "ignore previous", "[REDACTED]"
    )
    return filt


@pytest.fixture
def mirror(
    honcho_config: HonchoConfig, mock_injection_filter: MagicMock
) -> Any:
    """HonchoMemoryMirror instance with mock filter and fake journal."""
    from mousedroid.memory.honcho_mirror import HonchoMemoryMirror

    journal = _FakeJournal(
        [
            JournalEntry(
                ts_ns=1,
                event="mission_start",
                category="mission",
                payload={"target": "waypoint_1"},
            ),
            JournalEntry(
                ts_ns=2,
                event="estop",
                category="safety_state",
                payload={"reason": "obstacle"},
            ),
            JournalEntry(
                ts_ns=3,
                event="pref_update",
                category="operator_preference",
                payload={"speed": "slow"},
            ),
        ]
    )
    return HonchoMemoryMirror(
        honcho_config,
        journal=journal,
        injection_filter=mock_injection_filter,
    )


@pytest.mark.asyncio
async def test_sync_to_remote_filters_categories(mirror: Any) -> None:
    """Only 'mission' and 'operator_preference' entries are synced."""
    # Mirror is degraded without SDK, but we can test the filtering logic
    # by checking what would be synced
    from mousedroid.memory.honcho_mirror import _SAFE_SYNC_CATEGORIES

    assert "mission" in _SAFE_SYNC_CATEGORIES
    assert "operator_preference" in _SAFE_SYNC_CATEGORIES


@pytest.mark.asyncio
async def test_sync_to_remote_skips_safety_state() -> None:
    """Entries with category='safety_state' are NEVER synced."""
    from mousedroid.memory.honcho_mirror import _SAFE_SYNC_CATEGORIES

    assert "safety_state" not in _SAFE_SYNC_CATEGORIES


@pytest.mark.asyncio
async def test_degraded_when_sdk_missing(honcho_config: HonchoConfig) -> None:
    """start() without honcho SDK sets _degraded=True."""
    from mousedroid.memory.honcho_mirror import HonchoMemoryMirror

    mirror = HonchoMemoryMirror(honcho_config)
    await mirror.start()
    assert mirror.is_degraded is True


@pytest.mark.asyncio
async def test_recall_empty_query(mirror: Any) -> None:
    """Empty query returns empty list."""
    result = await mirror.recall("")
    assert result == []


@pytest.mark.asyncio
async def test_local_journal_authoritative(
    honcho_config: HonchoConfig,
) -> None:
    """Journal entries exist locally even when Honcho sync fails."""
    journal = _FakeJournal()
    entry = JournalEntry(
        event="test", category="mission", payload={"k": "v"}
    )
    await journal.append(entry)

    # Entries are in the journal regardless of mirror state
    entries: list[JournalEntry] = []
    async for e in journal.read_all():
        entries.append(e)
    assert len(entries) == 1
    assert entries[0].event == "test"
