"""Unit tests for SandboxPolicyGate."""

from __future__ import annotations

import pytest

from mousedroid.config.schema import OpenClawPolicyConfig
from mousedroid.harness.approval.protocol import ApprovalRequest
from mousedroid.harness.approval.sandbox_gate import SandboxPolicyGate

from .conftest import DummyGate


@pytest.mark.asyncio
async def test_sandbox_gate_static_limit_actuation_disabled() -> None:
    """Actuation skills are rejected when allow_actuation is False."""
    cfg = OpenClawPolicyConfig(allow_actuation=False)
    gate = SandboxPolicyGate(DummyGate(), cfg)
    gate._has_openshell = False

    req = ApprovalRequest(action="skill_delegate", payload={}, skill_name="move")
    decision = await gate.decide(req)

    assert not decision.approved
    assert decision.reason == "actuation_disabled"
    assert decision.decided_by == "sandbox_policy"


@pytest.mark.asyncio
async def test_sandbox_gate_static_limit_max_skills() -> None:
    """Skill count limit is enforced per task_id."""
    cfg = OpenClawPolicyConfig(max_skills_per_mission=2, allow_actuation=True)
    gate = SandboxPolicyGate(DummyGate(), cfg)
    gate._has_openshell = False

    req = ApprovalRequest(action="skill_delegate", payload={}, skill_name="say", task_id="t1")

    decision1 = await gate.decide(req)
    assert decision1.approved

    decision2 = await gate.decide(req)
    assert decision2.approved

    # 3rd invocation exceeds limit
    decision3 = await gate.decide(req)
    assert not decision3.approved
    assert decision3.reason == "max_skills_exceeded"


@pytest.mark.asyncio
async def test_sandbox_gate_openshell_mock() -> None:
    """When openshell is available, delegates to inner gate."""
    cfg = OpenClawPolicyConfig()
    gate = SandboxPolicyGate(DummyGate(), cfg)
    gate._has_openshell = True

    req = ApprovalRequest(action="skill_delegate", payload={}, skill_name="move", task_id="t1")
    decision = await gate.decide(req)

    assert decision.approved
    assert decision.decided_by == "dummy"


@pytest.mark.asyncio
async def test_sandbox_gate_task_id_none_defaults_to_unknown() -> None:
    """task_id=None falls back to 'unknown' for skill counting."""
    cfg = OpenClawPolicyConfig(max_skills_per_mission=1, allow_actuation=True)
    gate = SandboxPolicyGate(DummyGate(), cfg)
    gate._has_openshell = False

    req = ApprovalRequest(action="skill_delegate", payload={}, skill_name="say")
    # task_id defaults to None → "unknown"
    decision1 = await gate.decide(req)
    assert decision1.approved

    decision2 = await gate.decide(req)
    assert not decision2.approved
    assert decision2.reason == "max_skills_exceeded"


@pytest.mark.asyncio
async def test_sandbox_gate_non_skill_action_passes_through() -> None:
    """Non-skill_delegate actions are not checked by sandbox policy."""
    cfg = OpenClawPolicyConfig(allow_actuation=False)
    gate = SandboxPolicyGate(DummyGate(), cfg)
    gate._has_openshell = False

    req = ApprovalRequest(action="mission_dispatch", payload={})
    decision = await gate.decide(req)

    assert decision.approved
    assert decision.decided_by == "dummy"


@pytest.mark.asyncio
async def test_sandbox_gate_reset_counts() -> None:
    """reset_counts clears a mission's skill count."""
    cfg = OpenClawPolicyConfig(max_skills_per_mission=1, allow_actuation=True)
    gate = SandboxPolicyGate(DummyGate(), cfg)
    gate._has_openshell = False

    req = ApprovalRequest(action="skill_delegate", payload={}, skill_name="say", task_id="t1")
    decision1 = await gate.decide(req)
    assert decision1.approved

    # Would fail without reset
    gate.reset_counts("t1")

    decision2 = await gate.decide(req)
    assert decision2.approved


@pytest.mark.asyncio
async def test_sandbox_gate_bounded_eviction() -> None:
    """Oldest entries are evicted when max_tracked_missions is reached."""
    cfg = OpenClawPolicyConfig(
        max_skills_per_mission=10,
        max_tracked_missions=2,
        allow_actuation=True,
    )
    gate = SandboxPolicyGate(DummyGate(), cfg)
    gate._has_openshell = False

    # Fill the dict with 2 missions
    for task_id in ["t1", "t2"]:
        req = ApprovalRequest(
            action="skill_delegate", payload={}, skill_name="say", task_id=task_id
        )
        await gate.decide(req)

    assert len(gate._skill_counts) == 2

    # Adding a 3rd evicts the oldest (t1)
    req = ApprovalRequest(action="skill_delegate", payload={}, skill_name="say", task_id="t3")
    await gate.decide(req)

    assert len(gate._skill_counts) == 2
    assert "t1" not in gate._skill_counts
    assert "t3" in gate._skill_counts


@pytest.mark.asyncio
async def test_sandbox_gate_custom_actuation_skill_names() -> None:
    """Custom actuation_skill_names from config are enforced."""
    cfg = OpenClawPolicyConfig(
        allow_actuation=False,
        actuation_skill_names=("fire_laser", "deploy_arm"),
    )
    gate = SandboxPolicyGate(DummyGate(), cfg)
    gate._has_openshell = False

    # "move" is no longer in the actuation set
    req_move = ApprovalRequest(action="skill_delegate", payload={}, skill_name="move")
    decision = await gate.decide(req_move)
    assert decision.approved

    # "fire_laser" IS in the custom set
    req_laser = ApprovalRequest(action="skill_delegate", payload={}, skill_name="fire_laser")
    decision = await gate.decide(req_laser)
    assert not decision.approved
    assert decision.reason == "actuation_disabled"
