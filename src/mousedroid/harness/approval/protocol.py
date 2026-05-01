"""Approval-gate protocol and shared dataclasses."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ApprovalRequest:
    """A request for human or policy approval before an action runs.

    Attributes:
        id: Unique request id (auto-generated if not supplied).
        skill_name: Optional name of the skill that initiated the request.
        tool_name: Optional name of the tool whose dispatch is gated.
        task_id: Optional id of the harness task this request belongs to.
        action: Free-form description of the action being requested.
        payload: Arbitrary structured detail (e.g. tool kwargs).
        ts_ns: ``time.monotonic_ns()`` at request creation.
    """

    skill_name: str | None = None
    tool_name: str | None = None
    task_id: str | None = None
    action: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts_ns: int = field(default_factory=time.monotonic_ns)


@dataclass(frozen=True)
class ApprovalDecision:
    """The outcome of an approval gate.

    Attributes:
        approved: Whether the action may proceed.
        reason: Human-readable rationale (always populated for audit logs).
        decided_by: Identifier of the gate that produced the decision.
        decided_at_ns: Timestamp of the decision.
    """

    approved: bool
    reason: str = ""
    decided_by: str = ""
    decided_at_ns: int = field(default_factory=time.monotonic_ns)


@runtime_checkable
class ApprovalGateProtocol(Protocol):
    """Decides whether a pending action may proceed."""

    name: str

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        """Return an :class:`ApprovalDecision` for ``request``.

        Implementations MUST be async-safe and SHOULD never block longer
        than their configured timeout. Raising is forbidden — return a
        deny decision with the reason populated instead.
        """
        ...


__all__ = ["ApprovalDecision", "ApprovalGateProtocol", "ApprovalRequest"]
