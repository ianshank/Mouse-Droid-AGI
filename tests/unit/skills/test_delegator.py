"""Tests for ``mousedroid.skills.delegator.SkillDelegator``."""

from __future__ import annotations

from typing import Any

import pytest

from mousedroid.config.schema import HarnessTrackerConfig
from mousedroid.harness.approval.auto import AutoApproveGate
from mousedroid.harness.approval.protocol import (
    ApprovalDecision,
    ApprovalRequest,
)
from mousedroid.harness.journal.null_journal import NullJournal
from mousedroid.harness.journal.protocol import JournalEntry, JournalProtocol
from mousedroid.harness.predicates import AlwaysFalse
from mousedroid.harness.protocol import TaskSpec, TaskStatus
from mousedroid.harness.task_tracker import InMemoryTaskTracker
from mousedroid.skills.delegator import SkillDelegationError, SkillDelegator
from mousedroid.skills.protocol import (
    SkillSpec,
    SubAgentResult,
)
from mousedroid.skills.registry import SkillRegistry


class _RecordingJournal:
    def __init__(self) -> None:
        self.entries: list[JournalEntry] = []
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def append(self, entry: JournalEntry) -> None:
        self.entries.append(entry)

    async def read_all(self):  # type: ignore[no-untyped-def]
        for e in self.entries:
            yield e

    @property
    def is_running(self) -> bool:
        return self._running


class _DenyGate:
    name = "deny"

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=False, reason="nope", decided_by=self.name)


class _RaisingSubAgent:
    name = "kaboom"
    is_busy = False

    async def invoke(self, spec: TaskSpec, parent_ctx: Any | None = None) -> SubAgentResult:
        raise RuntimeError("subagent crashed")

    def cancel(self) -> None:
        pass


@pytest.fixture
def tracker() -> InMemoryTaskTracker:
    return InMemoryTaskTracker(HarnessTrackerConfig(enabled=True, history_size=8, max_active=4))


@pytest.fixture
def registry_with_skill() -> SkillRegistry:
    r = SkillRegistry()
    r.register(SkillSpec(name="diag", description="diagnostics"))
    return r


@pytest.fixture
def journal() -> _RecordingJournal:
    return _RecordingJournal()


def _task() -> TaskSpec:
    return TaskSpec(id="t1", goal="run diagnostics", acceptance_predicate=AlwaysFalse())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delegate_happy_path(
    registry_with_skill: SkillRegistry,
    tracker: InMemoryTaskTracker,
    journal: _RecordingJournal,
) -> None:
    delegator = SkillDelegator(
        registry_with_skill,
        AutoApproveGate(),
        journal,
        tracker,
    )
    result = await delegator.delegate("diag", _task())
    assert result.status == "ok"
    events = [e.event for e in journal.entries]
    assert events[0] == "delegate_requested"
    assert "delegate_started" in events
    assert events[-1] == "delegate_completed"
    # Tracker must have moved the task to terminal state.
    state = tracker.get("t1")
    assert state is not None
    assert state.is_terminal
    assert state.status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Approval denial
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delegate_denied_by_approval_gate(
    registry_with_skill: SkillRegistry,
    tracker: InMemoryTaskTracker,
    journal: _RecordingJournal,
) -> None:
    delegator = SkillDelegator(
        registry_with_skill,
        _DenyGate(),
        journal,
        tracker,
    )
    result = await delegator.delegate("diag", _task())
    assert result.status == "denied"
    assert result.error == "nope"
    events = [e.event for e in journal.entries]
    assert events == ["delegate_requested", "delegate_denied"]
    # Tracker was not consulted.
    assert tracker.get("t1") is None


# ---------------------------------------------------------------------------
# Sub-agent failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delegate_handles_sub_agent_exception(
    registry_with_skill: SkillRegistry,
    tracker: InMemoryTaskTracker,
    journal: _RecordingJournal,
) -> None:
    delegator = SkillDelegator(
        registry_with_skill,
        AutoApproveGate(),
        journal,
        tracker,
        agent_factory=lambda _name: _RaisingSubAgent(),
    )
    result = await delegator.delegate("diag", _task())
    assert result.status == "error"
    assert "crashed" in (result.error or "")
    state = tracker.get("t1")
    assert state is not None
    assert state.status == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# Misconfiguration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delegate_unknown_skill_raises(
    tracker: InMemoryTaskTracker, journal: _RecordingJournal
) -> None:
    delegator = SkillDelegator(SkillRegistry(), AutoApproveGate(), journal, tracker)
    with pytest.raises(SkillDelegationError):
        await delegator.delegate("ghost", _task())


@pytest.mark.asyncio
async def test_delegate_invalid_agent_factory_raises(
    registry_with_skill: SkillRegistry,
    tracker: InMemoryTaskTracker,
    journal: _RecordingJournal,
) -> None:
    delegator = SkillDelegator(
        registry_with_skill,
        AutoApproveGate(),
        journal,
        tracker,
        agent_factory=lambda _: object(),  # not a SubAgentProtocol
    )
    with pytest.raises(SkillDelegationError):
        await delegator.delegate("diag", _task())


# ---------------------------------------------------------------------------
# NullJournal interaction (does not raise)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delegate_with_null_journal(
    registry_with_skill: SkillRegistry, tracker: InMemoryTaskTracker
) -> None:
    j: JournalProtocol = NullJournal()
    delegator = SkillDelegator(registry_with_skill, AutoApproveGate(), j, tracker)
    result = await delegator.delegate("diag", _task())
    assert result.status == "ok"
