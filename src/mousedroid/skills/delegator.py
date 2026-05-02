"""Skill delegator — coordinates sub-agent invocation, approval, journal.

The delegator is the public coordination point used by the orchestrator
(or any caller) to invoke a named skill against a :class:`TaskSpec`. It
threads four concerns:

1. Approval — consults the configured :class:`ApprovalGateProtocol`
   before invocation.
2. Sub-agent dispatch — finds the registered :class:`SubAgentProtocol`
   for the skill (defaulting to :class:`NoOpSubAgent`).
3. Tracking — registers the task with the :class:`TaskTrackerProtocol`
   and updates its terminal status post-invocation.
4. Journalling — appends ``delegate_*`` entries via the configured
   :class:`JournalProtocol`.
"""

from __future__ import annotations

from typing import Any

from mousedroid.harness.approval.protocol import (
    ApprovalGateProtocol,
    ApprovalRequest,
)
from mousedroid.harness.journal.protocol import JournalEntry, JournalProtocol
from mousedroid.harness.protocol import (
    TaskSpec,
    TaskStatus,
    TaskTrackerProtocol,
)
from mousedroid.logging.setup import get_logger
from mousedroid.skills.protocol import (
    SkillRegistryProtocol,
    SubAgentProtocol,
    SubAgentResult,
)
from mousedroid.skills.sub_agent import NoOpSubAgent

_log = get_logger(__name__)


class SkillDelegationError(RuntimeError):
    """Raised on misconfigurations the delegator cannot recover from."""


class SkillDelegator:
    """Coordinates a single skill-delegation request end-to-end.

    Args:
        skill_registry: Registry of available skills.
        approval_gate: Gate consulted before sub-agent invocation. The
            default :class:`mousedroid.harness.approval.auto.AutoApproveGate`
            wired by the factory makes this transparent when HITL is off.
        journal: Append-only journal; receives delegation events.
        task_tracker: Tracker that owns the lifecycle of submitted tasks.
        agent_factory: Optional ``(skill_name) -> SubAgentProtocol``
            callable used to instantiate the sub-agent for a skill. The
            default returns a :class:`NoOpSubAgent`.
    """

    def __init__(
        self,
        skill_registry: SkillRegistryProtocol,
        approval_gate: ApprovalGateProtocol,
        journal: JournalProtocol,
        task_tracker: TaskTrackerProtocol,
        *,
        agent_factory: Any = None,
    ) -> None:
        self._skills = skill_registry
        self._approval = approval_gate
        self._journal = journal
        self._tracker = task_tracker
        self._agent_factory = agent_factory or (lambda name: NoOpSubAgent(name))

    async def delegate(
        self,
        skill_name: str,
        spec: TaskSpec,
        *,
        parent_ctx: Any | None = None,
    ) -> SubAgentResult:
        """Delegate ``spec`` to the named skill's sub-agent."""
        skill = self._skills.get(skill_name)
        if skill is None:
            msg = f"Unknown skill: {skill_name!r}"
            raise SkillDelegationError(msg)

        await self._append(
            "delegate_requested",
            spec.id,
            {"skill": skill_name, "goal": spec.goal},
        )

        approval = await self._approval.decide(
            ApprovalRequest(
                skill_name=skill_name,
                task_id=spec.id,
                action="skill_delegate",
                payload={"goal": spec.goal},
            )
        )
        if not approval.approved:
            await self._append(
                "delegate_denied",
                spec.id,
                {
                    "skill": skill_name,
                    "decided_by": approval.decided_by,
                    "reason": approval.reason,
                },
            )
            _log.info(
                "skill_delegate_denied",
                skill=skill_name,
                task_id=spec.id,
                reason=approval.reason,
            )
            return SubAgentResult(
                task_id=spec.id,
                status="denied",
                error=approval.reason,
            )

        # Submit to tracker so the orchestrator's tick loop can evaluate it.
        # If the tracker rejects the submission (duplicate id, capacity cap,
        # ...), fail-closed: skip the sub-agent invocation entirely so we
        # never run a side-effecting skill against a task the tracker did
        # not accept.
        try:
            self._tracker.submit(spec)
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning(
                "skill_delegate_tracker_submit_failed",
                skill=skill_name,
                task_id=spec.id,
                error=str(exc),
            )
            await self._append(
                "delegate_rejected",
                spec.id,
                {"skill": skill_name, "error": str(exc)},
            )
            return SubAgentResult(
                task_id=spec.id,
                status="rejected",
                error=f"tracker rejected submission: {exc}",
            )

        sub_agent = self._build_sub_agent(skill_name)
        await self._append(
            "delegate_started",
            spec.id,
            {"skill": skill_name, "agent": sub_agent.name},
        )
        try:
            result = await sub_agent.invoke(spec, parent_ctx)
        except Exception as exc:  # pylint: disable=broad-except
            await self._append(
                "delegate_failed",
                spec.id,
                {"skill": skill_name, "error": str(exc)},
            )
            self._tracker.update(spec.id, TaskStatus.FAILED, error=str(exc))
            _log.warning(
                "skill_delegate_failed",
                skill=skill_name,
                task_id=spec.id,
                error=str(exc),
                exc_info=True,
            )
            return SubAgentResult(task_id=spec.id, status="error", error=str(exc))

        await self._append(
            "delegate_completed",
            spec.id,
            {"skill": skill_name, "status": result.status, "latency_ms": result.latency_ms},
        )

        # Mirror the sub-agent's terminal status onto the tracker (unless
        # the predicate has already flipped it via the orchestrator tick).
        current = self._tracker.get(spec.id)
        if current is not None and not current.is_terminal:
            mapped = TaskStatus.COMPLETED if result.status == "ok" else TaskStatus.FAILED
            self._tracker.update(spec.id, mapped, error=result.error)
        return result

    # ------------------------------------------------------------ helpers
    def _build_sub_agent(self, skill_name: str) -> SubAgentProtocol:
        agent = self._agent_factory(skill_name)
        if not isinstance(agent, SubAgentProtocol):
            msg = (
                f"agent_factory returned a non-SubAgentProtocol object "
                f"for skill {skill_name!r}: {type(agent).__name__}"
            )
            raise SkillDelegationError(msg)
        return agent

    async def _append(
        self,
        event: str,
        task_id: str,
        payload: dict[str, Any],
    ) -> None:
        await self._journal.append(
            JournalEntry(
                task_id=task_id,
                phase="delegator",
                event=event,
                payload=payload,
            )
        )


__all__ = ["SkillDelegationError", "SkillDelegator"]
