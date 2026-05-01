"""Tests for ``mousedroid.harness.approval`` — auto / cli / callback / policy."""

from __future__ import annotations

import asyncio

import pytest

from mousedroid.harness.approval.auto import AutoApproveGate
from mousedroid.harness.approval.callback import (
    AsyncCallbackApprovalGate,
)
from mousedroid.harness.approval.cli import CLIApprovalGate
from mousedroid.harness.approval.policy import PolicyApprovalGate
from mousedroid.harness.approval.protocol import (
    ApprovalDecision,
    ApprovalGateProtocol,
    ApprovalRequest,
)


def _request(**overrides) -> ApprovalRequest:
    base: dict = {"tool_name": "esp32_diagnostics", "skill_name": "diagnose", "task_id": "t1"}
    base.update(overrides)
    return ApprovalRequest(**base)


# ---------------------------------------------------------------------------
# AutoApproveGate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_gate_implements_protocol() -> None:
    g = AutoApproveGate()
    assert isinstance(g, ApprovalGateProtocol)


@pytest.mark.asyncio
async def test_auto_gate_always_approves() -> None:
    g = AutoApproveGate()
    decision = await g.decide(_request())
    assert decision.approved is True
    assert decision.decided_by == "auto"
    assert decision.reason == "auto-approved"


# ---------------------------------------------------------------------------
# CLIApprovalGate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cli_gate_invalid_on_timeout_rejected() -> None:
    with pytest.raises(ValueError, match="on_timeout"):
        CLIApprovalGate(timeout_s=1.0, on_timeout="bogus")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_cli_gate_approves_on_yes() -> None:
    async def reader(_prompt: str) -> str:
        return "y"

    g = CLIApprovalGate(timeout_s=1.0, reader=reader)
    decision = await g.decide(_request())
    assert decision.approved is True
    assert decision.decided_by == "cli"


@pytest.mark.asyncio
async def test_cli_gate_denies_on_no() -> None:
    async def reader(_prompt: str) -> str:
        return "n"

    g = CLIApprovalGate(timeout_s=1.0, reader=reader)
    decision = await g.decide(_request())
    assert decision.approved is False


@pytest.mark.asyncio
async def test_cli_gate_timeout_fail_closed() -> None:
    async def slow(_prompt: str) -> str:
        await asyncio.sleep(10)
        return "y"

    g = CLIApprovalGate(timeout_s=0.05, on_timeout="deny", reader=slow)
    decision = await g.decide(_request())
    assert decision.approved is False
    assert "timeout" in decision.reason


@pytest.mark.asyncio
async def test_cli_gate_timeout_fail_open_when_configured() -> None:
    async def slow(_prompt: str) -> str:
        await asyncio.sleep(10)
        return "y"

    g = CLIApprovalGate(timeout_s=0.05, on_timeout="approve", reader=slow)
    decision = await g.decide(_request())
    assert decision.approved is True


# ---------------------------------------------------------------------------
# AsyncCallbackApprovalGate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_callback_gate_passes_request_to_callback() -> None:
    seen: list[ApprovalRequest] = []

    async def cb(req: ApprovalRequest) -> bool:
        seen.append(req)
        return True

    g = AsyncCallbackApprovalGate(cb, timeout_s=1.0)
    req = _request()
    decision = await g.decide(req)
    assert decision.approved is True
    assert seen[0] is req


@pytest.mark.asyncio
async def test_callback_gate_callback_exception_denies() -> None:
    async def cb(_req: ApprovalRequest) -> bool:
        raise RuntimeError("boom")

    g = AsyncCallbackApprovalGate(cb, timeout_s=1.0)
    decision = await g.decide(_request())
    assert decision.approved is False
    assert "boom" in decision.reason


@pytest.mark.asyncio
async def test_callback_gate_timeout_fail_closed() -> None:
    async def slow(_req: ApprovalRequest) -> bool:
        await asyncio.sleep(10)
        return True

    g = AsyncCallbackApprovalGate(slow, timeout_s=0.05)
    decision = await g.decide(_request())
    assert decision.approved is False


@pytest.mark.asyncio
async def test_callback_gate_invalid_on_timeout_rejected() -> None:
    async def cb(_req: ApprovalRequest) -> bool:
        return True

    with pytest.raises(ValueError, match="on_timeout"):
        AsyncCallbackApprovalGate(cb, timeout_s=1.0, on_timeout="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PolicyApprovalGate
# ---------------------------------------------------------------------------


class _SpyGate:
    name = "spy"

    def __init__(self, decision: ApprovalDecision) -> None:
        self._decision = decision
        self.calls: list[ApprovalRequest] = []

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.calls.append(request)
        return self._decision


@pytest.mark.asyncio
async def test_policy_no_patterns_auto_approves() -> None:
    spy = _SpyGate(ApprovalDecision(approved=False, reason="should not run"))
    gate = PolicyApprovalGate(spy)
    decision = await gate.decide(_request())
    assert decision.approved is True
    assert decision.decided_by == "policy"
    assert spy.calls == []


@pytest.mark.asyncio
async def test_policy_matched_tool_delegates_to_inner() -> None:
    inner_decision = ApprovalDecision(approved=False, reason="denied", decided_by="cli")
    spy = _SpyGate(inner_decision)
    gate = PolicyApprovalGate(spy, tool_patterns=("esp32_*",))
    decision = await gate.decide(_request(tool_name="esp32_diagnostics"))
    assert decision.approved is False
    assert decision.decided_by == "cli"
    assert len(spy.calls) == 1


@pytest.mark.asyncio
async def test_policy_matched_skill_delegates_to_inner() -> None:
    inner_decision = ApprovalDecision(approved=True, reason="approved")
    spy = _SpyGate(inner_decision)
    gate = PolicyApprovalGate(spy, skill_patterns=("dangerous_*",))
    decision = await gate.decide(_request(skill_name="dangerous_skill", tool_name=None))
    assert decision.approved is True
    assert len(spy.calls) == 1


@pytest.mark.asyncio
async def test_policy_unmatched_does_not_call_inner() -> None:
    spy = _SpyGate(ApprovalDecision(approved=False))
    gate = PolicyApprovalGate(spy, tool_patterns=("dangerous_*",))
    decision = await gate.decide(_request(tool_name="health_check"))
    assert decision.approved is True
    assert spy.calls == []


def test_policy_requires_approval_helper() -> None:
    spy = _SpyGate(ApprovalDecision(approved=True))
    gate = PolicyApprovalGate(spy, tool_patterns=("nuclear_*",))
    assert gate.requires_approval(_request(tool_name="nuclear_launch")) is True
    assert gate.requires_approval(_request(tool_name="health_check")) is False


@pytest.mark.asyncio
async def test_request_id_and_ts_auto_populated() -> None:
    req = _request()
    assert req.id  # uuid4 hex is non-empty
    assert req.ts_ns > 0
