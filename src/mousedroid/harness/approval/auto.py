"""Auto-approve gate — the no-op default."""

from __future__ import annotations

from mousedroid.harness.approval.protocol import (
    ApprovalDecision,
    ApprovalGateProtocol,
    ApprovalRequest,
)
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class AutoApproveGate:
    """Approves every request unconditionally."""

    name = "auto"

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        _log.debug(
            "approval_auto_approved",
            request_id=request.id,
            tool=request.tool_name,
            skill=request.skill_name,
        )
        return ApprovalDecision(
            approved=True,
            reason="auto-approved",
            decided_by=self.name,
        )


_PROTOCOL_CHECK: ApprovalGateProtocol = AutoApproveGate()
del _PROTOCOL_CHECK


__all__ = ["AutoApproveGate"]
