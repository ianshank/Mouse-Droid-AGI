"""Sandbox policy gate for OpenClaw/openshell constraints."""

from __future__ import annotations

import shutil
from collections import OrderedDict

from mousedroid.config.schema import OpenClawPolicyConfig
from mousedroid.harness.approval.protocol import (
    ApprovalDecision,
    ApprovalGateProtocol,
    ApprovalRequest,
)
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class SandboxPolicyGate:
    """Delegates policy decisions to openshell or static limits.

    If the ``openshell`` binary is present, this gate will (eventually)
    consult it for live policy enforcement. If missing, it falls back
    to static limits defined in ``OpenClawPolicyConfig``.
    """

    name = "sandbox_policy"

    def __init__(
        self,
        inner: ApprovalGateProtocol,
        cfg: OpenClawPolicyConfig,
    ) -> None:
        self._inner = inner
        self._cfg = cfg
        self._has_openshell = shutil.which("openshell") is not None
        # Bounded LRU-style dict for per-mission skill counts.
        # Evicts oldest entry when max_tracked_missions is reached.
        self._skill_counts: OrderedDict[str, int] = OrderedDict()
        self._actuation_set: frozenset[str] = frozenset(cfg.actuation_skill_names)

    def _get_skill_count(self, task_id: str) -> int:
        """Get current skill count for a task, maintaining LRU order."""
        if task_id in self._skill_counts:
            self._skill_counts.move_to_end(task_id)
            return self._skill_counts[task_id]
        return 0

    def _increment_skill_count(self, task_id: str) -> None:
        """Increment skill count, evicting oldest if at capacity."""
        if task_id in self._skill_counts:
            self._skill_counts.move_to_end(task_id)
            self._skill_counts[task_id] += 1
        else:
            if len(self._skill_counts) >= self._cfg.max_tracked_missions:
                self._skill_counts.popitem(last=False)
            self._skill_counts[task_id] = 1

    def reset_counts(self, task_id: str) -> None:
        """Remove a completed mission's count entry.

        Called by the orchestrator at mission completion to prevent
        stale entries from accumulating.

        Args:
            task_id: The mission task identifier to clear.
        """
        self._skill_counts.pop(task_id, None)

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        """Evaluate a skill delegation request against sandbox policy."""
        if request.action == "skill_delegate":
            task_id = request.task_id or "unknown"

            if not self._has_openshell:
                # Fallback: Static constraints
                count = self._get_skill_count(task_id)
                if count >= self._cfg.max_skills_per_mission:
                    _log.warning(
                        "sandbox_policy_rejected",
                        reason="max_skills_exceeded",
                        task_id=task_id,
                        skill=request.skill_name,
                        limit=self._cfg.max_skills_per_mission,
                    )
                    return ApprovalDecision(
                        approved=False,
                        reason="max_skills_exceeded",
                        decided_by=self.name,
                    )
                self._increment_skill_count(task_id)

                # Actuation check using config-driven set
                if not self._cfg.allow_actuation and request.skill_name in self._actuation_set:
                    _log.warning(
                        "sandbox_policy_rejected",
                        reason="actuation_disabled",
                        task_id=task_id,
                        skill=request.skill_name,
                    )
                    return ApprovalDecision(
                        approved=False,
                        reason="actuation_disabled",
                        decided_by=self.name,
                    )
            else:
                # Spike: if openshell is mature, we would shell out or RPC to it here.
                _log.debug("sandbox_policy_checked_openshell", skill=request.skill_name)

        return await self._inner.decide(request)
