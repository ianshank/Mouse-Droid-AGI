"""Pattern-based gate that wraps an inner gate.

The policy gate is the default wiring; it consults config-driven
``fnmatch`` pattern lists to decide whether approval is even required.
When the pending request does not match any pattern, the gate auto-
approves and never wakes the inner gate.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence

from mousedroid.harness.approval.protocol import (
    ApprovalDecision,
    ApprovalGateProtocol,
    ApprovalRequest,
)
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class PolicyApprovalGate:
    """Selectively delegates to an inner gate based on name patterns.

    Args:
        inner: The wrapped gate consulted only for matched requests.
        tool_patterns: ``fnmatch`` patterns over ``request.tool_name``.
        skill_patterns: ``fnmatch`` patterns over ``request.skill_name``.

    The gate is a no-op (auto-approve) when both pattern lists are empty,
    matching the documented ``HarnessApprovalConfig`` default.
    """

    name = "policy"

    def __init__(
        self,
        inner: ApprovalGateProtocol,
        *,
        tool_patterns: Sequence[str] = (),
        skill_patterns: Sequence[str] = (),
    ) -> None:
        self._inner = inner
        self._tool_patterns = tuple(tool_patterns)
        self._skill_patterns = tuple(skill_patterns)

    def requires_approval(self, request: ApprovalRequest) -> bool:
        """Return True when ``request`` matches any configured pattern."""
        if request.tool_name is not None:
            for pattern in self._tool_patterns:
                if fnmatch.fnmatchcase(request.tool_name, pattern):
                    return True
        if request.skill_name is not None:
            for pattern in self._skill_patterns:
                if fnmatch.fnmatchcase(request.skill_name, pattern):
                    return True
        return False

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if not self.requires_approval(request):
            _log.debug(
                "approval_policy_skipped",
                request_id=request.id,
                tool=request.tool_name,
                skill=request.skill_name,
            )
            return ApprovalDecision(
                approved=True,
                reason="policy auto-approved (no match)",
                decided_by=self.name,
            )
        decision = await self._inner.decide(request)
        _log.info(
            "approval_policy_delegated",
            request_id=request.id,
            inner=self._inner.name,
            approved=decision.approved,
        )
        return decision


__all__ = ["PolicyApprovalGate"]
