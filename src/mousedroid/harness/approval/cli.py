"""Stdin-prompt approval gate with timeout."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable

from mousedroid.harness.approval.protocol import (
    ApprovalDecision,
    ApprovalGateProtocol,
    ApprovalRequest,
)
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


_AcceptedAffirmatives: frozenset[str] = frozenset({"y", "yes"})


_DEFAULT_INPUT: Callable[[str], Awaitable[str]]


async def _stdin_input(prompt: str) -> str:
    """Read one line from stdin without blocking the event loop."""
    return await asyncio.to_thread(input, prompt)


_DEFAULT_INPUT = _stdin_input


class CLIApprovalGate:
    """Prompts the operator on stdin to approve / deny.

    The on-timeout decision is configurable so callers can fail-closed
    (default) or fail-open per :class:`HarnessApprovalConfig.on_timeout`.
    """

    name = "cli"

    def __init__(
        self,
        *,
        timeout_s: float,
        on_timeout: str = "deny",
        reader: Callable[[str], Awaitable[str]] | None = None,
    ) -> None:
        if on_timeout not in {"deny", "approve"}:
            msg = f"on_timeout must be 'deny' or 'approve'; got {on_timeout!r}"
            raise ValueError(msg)
        self._timeout_s = timeout_s
        self._on_timeout = on_timeout
        self._reader = reader if reader is not None else _DEFAULT_INPUT

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        prompt = (
            f"[approval] {request.action or 'action'} "
            f"tool={request.tool_name!r} skill={request.skill_name!r} "
            f"task={request.task_id!r} (y/N): "
        )
        try:
            answer = await asyncio.wait_for(
                self._reader(prompt),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            _log.warning(
                "approval_cli_timeout",
                request_id=request.id,
                fallback=self._on_timeout,
                timeout_s=self._timeout_s,
            )
            approved = self._on_timeout == "approve"
            return ApprovalDecision(
                approved=approved,
                reason=f"timeout after {self._timeout_s}s ({self._on_timeout})",
                decided_by=self.name,
            )
        approved = answer.strip().lower() in _AcceptedAffirmatives
        _log.info(
            "approval_cli_decision",
            request_id=request.id,
            approved=approved,
        )
        return ApprovalDecision(
            approved=approved,
            reason="cli accepted" if approved else "cli denied",
            decided_by=self.name,
        )


_PROTOCOL_CHECK: ApprovalGateProtocol = CLIApprovalGate(timeout_s=0.001)
del _PROTOCOL_CHECK


__all__ = ["CLIApprovalGate"]


# silence flake about unused import marker on some lint configurations
_ = sys
