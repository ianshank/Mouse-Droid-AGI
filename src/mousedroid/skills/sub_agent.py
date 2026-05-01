"""Concrete sub-agent implementations.

Two implementations are provided:

* :class:`NoOpSubAgent` — returns a deterministic ``status='ok'`` result;
  the default backend so the harness has zero external dependencies.
* :class:`LLMBackedSubAgent` — delegates one task to an injected
  :class:`LLMGatewayProtocol` via its ``translate_mission`` method,
  emitting structured journal entries through an injected callable.

Both classes are typed against :class:`SubAgentProtocol` (the harness-
level surface, distinct from the tensor-level ``AgentProtocol``).
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from mousedroid.harness.journal.protocol import JournalEntry
from mousedroid.harness.protocol import TaskSpec
from mousedroid.logging.setup import get_logger
from mousedroid.skills.protocol import (
    SkillSpec,
    SubAgentProtocol,
    SubAgentResult,
)

_log = get_logger(__name__)


JournalAppender = Callable[[JournalEntry], Awaitable[None]]
"""Async callable used to record sub-agent activity to the harness journal."""


class NoOpSubAgent:
    """Deterministic sub-agent returning a fixed result. Used as a default."""

    def __init__(self, name: str = "noop") -> None:
        self._name = name
        self._busy = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_busy(self) -> bool:
        return self._busy

    async def invoke(self, spec: TaskSpec, parent_ctx: Any | None = None) -> SubAgentResult:
        self._busy = True
        try:
            _log.info("noop_sub_agent_invoked", task_id=spec.id, name=self._name)
            return SubAgentResult(
                task_id=spec.id,
                status="ok",
                output=None,
                journal_refs=(),
                latency_ms=0.0,
            )
        finally:
            self._busy = False

    def cancel(self) -> None:
        self._busy = False


class LLMBackedSubAgent:
    """Sub-agent that delegates one task to an LLM via the gateway protocol.

    The gateway's ``translate_mission`` is consumed to keep the surface
    aligned with the existing ``LLMGatewayProtocol``. A ``Callable`` for
    journal appends is injected so the sub-agent never imports a journal
    backend directly (DI / no concrete imports outside the factory).
    """

    def __init__(
        self,
        skill: SkillSpec,
        llm_gateway: Any,  # LLMGatewayProtocol — typed Any to avoid import cycle
        *,
        journal_append: JournalAppender | None = None,
        agent_id: str | None = None,
    ) -> None:
        self._skill = skill
        self._gateway = llm_gateway
        self._journal_append = journal_append
        self._agent_id = agent_id or f"sub:{skill.name}"
        self._busy = False

    @property
    def name(self) -> str:
        return self._skill.name

    @property
    def is_busy(self) -> bool:
        return self._busy

    async def _journal(
        self,
        event: str,
        task_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._journal_append is None:
            return
        await self._journal_append(
            JournalEntry(
                task_id=task_id,
                phase="sub_agent",
                event=event,
                payload=payload or {},
                agent_id=self._agent_id,
            )
        )

    async def invoke(self, spec: TaskSpec, parent_ctx: Any | None = None) -> SubAgentResult:
        self._busy = True
        start = time.monotonic()
        try:
            _log.info(
                "llm_sub_agent_invoked",
                skill=self._skill.name,
                task_id=spec.id,
            )
            await self._journal("started", spec.id, {"goal": spec.goal})

            if not getattr(self._gateway, "is_ready", False):
                await self._journal("llm_unavailable", spec.id)
                latency_ms = (time.monotonic() - start) * 1000.0
                return SubAgentResult(
                    task_id=spec.id,
                    status="llm_unavailable",
                    latency_ms=latency_ms,
                    error="gateway not ready",
                )

            try:
                output = await self._gateway.translate_mission(spec.goal)
            except Exception as exc:  # pylint: disable=broad-except
                latency_ms = (time.monotonic() - start) * 1000.0
                await self._journal(
                    "failed",
                    spec.id,
                    {"error": str(exc), "latency_ms": latency_ms},
                )
                return SubAgentResult(
                    task_id=spec.id,
                    status="error",
                    error=str(exc),
                    latency_ms=latency_ms,
                )

            latency_ms = (time.monotonic() - start) * 1000.0
            await self._journal(
                "completed",
                spec.id,
                {"latency_ms": latency_ms},
            )
            return SubAgentResult(
                task_id=spec.id,
                status="ok",
                output=output,
                latency_ms=latency_ms,
            )
        finally:
            self._busy = False

    def cancel(self) -> None:
        self._busy = False


_PROTOCOL_CHECKS: tuple[SubAgentProtocol, ...] = (
    NoOpSubAgent(),
    LLMBackedSubAgent(SkillSpec(name="x"), llm_gateway=object()),
)
del _PROTOCOL_CHECKS


__all__ = ["JournalAppender", "LLMBackedSubAgent", "NoOpSubAgent"]
