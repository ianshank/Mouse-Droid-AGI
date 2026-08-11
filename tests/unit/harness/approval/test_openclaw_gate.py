"""Unit tests for OpenClawSafetyGate."""

from __future__ import annotations

import pytest

from mousedroid.config.schema import OpenClawConfig
from mousedroid.harness.approval.openclaw_gate import OpenClawSafetyGate
from mousedroid.harness.approval.protocol import ApprovalRequest

from .conftest import DummyFilter, DummyGate


@pytest.mark.asyncio
async def test_openclaw_gate_mission_dispatch_allowed() -> None:
    """Requests on an allowed channel pass through to the inner gate."""
    cfg = OpenClawConfig(allowed_channels=("rest",))
    gate = OpenClawSafetyGate(DummyGate(), DummyFilter(), cfg)

    req = ApprovalRequest(
        action="mission_dispatch",
        payload={"nl_command": "hello", "channel": "rest", "peer": "peer1"},
    )
    decision = await gate.decide(req)
    assert decision.approved
    assert decision.decided_by == "dummy"


@pytest.mark.asyncio
async def test_openclaw_gate_channel_rejected() -> None:
    """Requests on a disallowed channel are rejected."""
    cfg = OpenClawConfig(allowed_channels=("rest",))
    gate = OpenClawSafetyGate(DummyGate(), DummyFilter(), cfg)

    req = ApprovalRequest(
        action="mission_dispatch",
        payload={"nl_command": "hello", "channel": "mcp", "peer": "peer1"},
    )
    decision = await gate.decide(req)
    assert not decision.approved
    assert "channel_not_allowed" in decision.reason


@pytest.mark.asyncio
async def test_openclaw_gate_injection_rejected() -> None:
    """Injection patterns in the NL command are rejected."""
    cfg = OpenClawConfig(allowed_channels=("rest",))
    gate = OpenClawSafetyGate(DummyGate(), DummyFilter(), cfg)

    req = ApprovalRequest(
        action="mission_dispatch",
        payload={"nl_command": "this is bad", "channel": "rest", "peer": "peer1"},
    )
    decision = await gate.decide(req)
    assert not decision.approved
    assert "injection_pattern" in decision.reason


@pytest.mark.asyncio
async def test_openclaw_gate_command_too_long() -> None:
    """Overlong commands are rejected before injection checking."""
    cfg = OpenClawConfig(allowed_channels=("rest",), max_command_len=10)
    gate = OpenClawSafetyGate(DummyGate(), DummyFilter(), cfg)

    req = ApprovalRequest(
        action="mission_dispatch",
        payload={"nl_command": "this command is way too long", "channel": "rest", "peer": "peer1"},
    )
    decision = await gate.decide(req)
    assert not decision.approved
    assert "command_too_long" in decision.reason


@pytest.mark.asyncio
async def test_openclaw_gate_no_text_passes_through() -> None:
    """Requests without nl_command or goal skip text checks."""
    cfg = OpenClawConfig(allowed_channels=("rest",))
    gate = OpenClawSafetyGate(DummyGate(), DummyFilter(), cfg)

    req = ApprovalRequest(
        action="mission_dispatch",
        payload={"channel": "rest", "peer": "peer1"},
    )
    decision = await gate.decide(req)
    assert decision.approved


@pytest.mark.asyncio
async def test_openclaw_gate_skill_delegate_injection_check() -> None:
    """skill_delegate action also checks goal text for injection."""
    cfg = OpenClawConfig(allowed_channels=("rest",))
    gate = OpenClawSafetyGate(DummyGate(), DummyFilter(), cfg)

    req = ApprovalRequest(
        action="skill_delegate",
        payload={"goal": "do bad things"},
    )
    decision = await gate.decide(req)
    assert not decision.approved
    assert "injection_pattern" in decision.reason


@pytest.mark.asyncio
async def test_openclaw_gate_non_mission_no_channel_check() -> None:
    """Non-mission actions skip the channel allowlist check."""
    cfg = OpenClawConfig(allowed_channels=("rest",))
    gate = OpenClawSafetyGate(DummyGate(), DummyFilter(), cfg)

    req = ApprovalRequest(
        action="skill_delegate",
        payload={"goal": "safe goal"},
    )
    decision = await gate.decide(req)
    assert decision.approved
