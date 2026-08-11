"""OpenClaw safety envelope for gating skill and mission dispatches."""

from __future__ import annotations

from mousedroid.config.schema import OpenClawConfig
from mousedroid.harness.approval.protocol import (
    ApprovalDecision,
    ApprovalGateProtocol,
    ApprovalRequest,
)
from mousedroid.logging.setup import get_logger
from mousedroid.security.injection_filter import (
    InjectionRejected,
    PromptInjectionFilterProtocol,
)

_log = get_logger(__name__)


class OpenClawSafetyGate:
    """Envelope that enforces OpenClaw safety policies.

    Checks max_command_len and prompt injection across both mission
    dispatch (nl_command) and skill delegation (goal). Also enforces
    channel allow-listing for mission dispatch.
    """

    name = "openclaw_safety"

    def __init__(
        self,
        inner: ApprovalGateProtocol,
        filter_impl: PromptInjectionFilterProtocol,
        cfg: OpenClawConfig,
    ) -> None:
        self._inner = inner
        self._filter = filter_impl
        self._cfg = cfg
        self._allowed_channels = frozenset(cfg.allowed_channels)

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if request.action == "mission_dispatch":
            channel = request.payload.get("channel")
            if channel and channel not in self._allowed_channels:
                _log.warning(
                    "mission_dispatched_rejected",
                    reason="channel_not_allowed",
                    peer=request.payload.get("peer", "unknown"),
                    allowed=sorted(self._allowed_channels),
                )
                return ApprovalDecision(
                    approved=False,
                    reason="channel_not_allowed",
                    decided_by=self.name,
                )

        text_to_check = request.payload.get("nl_command") or request.payload.get("goal")
        if text_to_check:
            if len(text_to_check) > self._cfg.max_command_len:
                _log.warning(
                    "openclaw_dispatch_rejected",
                    reason="command_too_long",
                    action=request.action,
                    length=len(text_to_check),
                    limit=self._cfg.max_command_len,
                )
                return ApprovalDecision(
                    approved=False,
                    reason="command_too_long",
                    decided_by=self.name,
                )
            try:
                self._filter.sanitize(text_to_check)
            except InjectionRejected:
                _log.warning(
                    "openclaw_dispatch_rejected",
                    reason="injection_pattern",
                    action=request.action,
                )
                return ApprovalDecision(
                    approved=False,
                    reason="injection_pattern",
                    decided_by=self.name,
                )

        return await self._inner.decide(request)
