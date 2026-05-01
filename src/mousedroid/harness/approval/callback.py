"""Async-callback approval gate.

Useful for wiring an external webhook, message-bus subscriber, or test
harness as the decision authority. The callback receives the
:class:`ApprovalRequest` and must return a bool (or raise to deny).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from mousedroid.harness.approval.protocol import (
    ApprovalDecision,
    ApprovalGateProtocol,
    ApprovalRequest,
)
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


ApprovalCallback = Callable[[ApprovalRequest], Awaitable[bool]]


class AsyncCallbackApprovalGate:
    """Delegates the decision to a caller-supplied async callable."""

    name = "callback"

    def __init__(
        self,
        callback: ApprovalCallback,
        *,
        timeout_s: float,
        on_timeout: str = "deny",
    ) -> None:
        if on_timeout not in {"deny", "approve"}:
            msg = f"on_timeout must be 'deny' or 'approve'; got {on_timeout!r}"
            raise ValueError(msg)
        self._callback = callback
        self._timeout_s = timeout_s
        self._on_timeout = on_timeout

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        try:
            approved = await asyncio.wait_for(
                self._callback(request),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            _log.warning(
                "approval_callback_timeout",
                request_id=request.id,
                fallback=self._on_timeout,
                timeout_s=self._timeout_s,
            )
            return ApprovalDecision(
                approved=self._on_timeout == "approve",
                reason=f"callback timeout after {self._timeout_s}s ({self._on_timeout})",
                decided_by=self.name,
            )
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning(
                "approval_callback_error",
                request_id=request.id,
                error=str(exc),
            )
            return ApprovalDecision(
                approved=False,
                reason=f"callback raised: {exc}",
                decided_by=self.name,
            )
        return ApprovalDecision(
            approved=bool(approved),
            reason="callback approved" if approved else "callback denied",
            decided_by=self.name,
        )


async def _noop_callback(_request: ApprovalRequest) -> bool:  # pragma: no cover - structural
    return False


_PROTOCOL_CHECK: ApprovalGateProtocol = AsyncCallbackApprovalGate(_noop_callback, timeout_s=0.001)
del _PROTOCOL_CHECK


__all__ = ["ApprovalCallback", "AsyncCallbackApprovalGate"]
