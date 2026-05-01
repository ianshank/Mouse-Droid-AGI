"""Tests for ``mousedroid.skills.sub_agent``."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from mousedroid.harness.journal.protocol import JournalEntry
from mousedroid.harness.predicates import AlwaysFalse
from mousedroid.harness.protocol import TaskSpec
from mousedroid.skills.protocol import (
    SkillSpec,
    SubAgentProtocol,
)
from mousedroid.skills.sub_agent import (
    LLMBackedSubAgent,
    NoOpSubAgent,
)


def _spec() -> TaskSpec:
    return TaskSpec(id="t1", goal="go forward", acceptance_predicate=AlwaysFalse())


# ---------------------------------------------------------------------------
# NoOpSubAgent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_noop_sub_agent_implements_protocol() -> None:
    a = NoOpSubAgent()
    assert isinstance(a, SubAgentProtocol)


@pytest.mark.asyncio
async def test_noop_returns_ok() -> None:
    a = NoOpSubAgent("custom")
    assert a.name == "custom"
    assert not a.is_busy
    out = await a.invoke(_spec())
    assert out.status == "ok"
    assert out.task_id == "t1"
    assert out.latency_ms == 0.0


# ---------------------------------------------------------------------------
# LLMBackedSubAgent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_sub_agent_calls_translate_mission() -> None:
    gateway = AsyncMock()
    gateway.is_ready = True
    gateway.translate_mission = AsyncMock(return_value={"vx": 0.5})
    skill = SkillSpec(name="nav", description="navigate")

    appended: list[JournalEntry] = []

    async def journal(entry: JournalEntry) -> None:
        appended.append(entry)

    agent = LLMBackedSubAgent(skill, gateway, journal_append=journal)
    result = await agent.invoke(_spec())
    assert result.status == "ok"
    assert result.output == {"vx": 0.5}
    gateway.translate_mission.assert_awaited_once_with("go forward")
    events = [e.event for e in appended]
    assert events == ["started", "completed"]
    assert all(e.agent_id == "sub:nav" for e in appended)


@pytest.mark.asyncio
async def test_llm_sub_agent_handles_unready_gateway() -> None:
    gateway = AsyncMock()
    gateway.is_ready = False
    skill = SkillSpec(name="nav")

    agent = LLMBackedSubAgent(skill, gateway)
    result = await agent.invoke(_spec())
    assert result.status == "llm_unavailable"
    gateway.translate_mission.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_sub_agent_handles_gateway_exception() -> None:
    gateway = AsyncMock()
    gateway.is_ready = True
    gateway.translate_mission.side_effect = RuntimeError("boom")
    skill = SkillSpec(name="nav")

    appended: list[JournalEntry] = []

    async def journal(entry: JournalEntry) -> None:
        appended.append(entry)

    agent = LLMBackedSubAgent(skill, gateway, journal_append=journal)
    result = await agent.invoke(_spec())
    assert result.status == "error"
    assert result.error == "boom"
    events = [e.event for e in appended]
    assert events == ["started", "failed"]


@pytest.mark.asyncio
async def test_llm_sub_agent_journal_optional() -> None:
    gateway = AsyncMock()
    gateway.is_ready = True
    gateway.translate_mission = AsyncMock(return_value={"vx": 0.0})
    skill = SkillSpec(name="nav")

    agent = LLMBackedSubAgent(skill, gateway, journal_append=None)
    result = await agent.invoke(_spec())
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_llm_sub_agent_busy_flag_resets() -> None:
    gateway = AsyncMock()
    gateway.is_ready = True
    gateway.translate_mission = AsyncMock(return_value={})
    agent = LLMBackedSubAgent(SkillSpec(name="x"), gateway)
    assert not agent.is_busy
    await agent.invoke(_spec())
    assert not agent.is_busy
