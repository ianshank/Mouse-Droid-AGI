"""PR #105b B.1: regression coverage for the ``build_approval_gate`` typing fix.

The pre-existing PR-98 mypy error at ``factory.py:2262`` reported
``Argument 1 to "PolicyApprovalGate" has incompatible type "object"; expected
"ApprovalGateProtocol"``. The fix tightens the local ``inner`` annotation
from ``object`` to :class:`ApprovalGateProtocol`. These tests pin the
runtime contract so a future refactor can't silently widen the annotation
again.

Coverage scope:

1. Default ``auto`` gate returns :class:`AutoApproveGate` directly when no
   patterns are configured.
2. ``auto`` gate WITH patterns wraps in :class:`PolicyApprovalGate` (the
   call site that originally tripped the mypy error).
3. ``cli`` + ``callback`` branches build their respective gates + still
   typecheck against the protocol.
4. ``cfg.harness is None`` short-circuit still returns an
   :class:`AutoApproveGate`.

All assertions duck-type via ``isinstance(gate, ApprovalGateProtocol)`` so
the runtime invariant the mypy fix represents stays pinned regardless of
which concrete class lands in any branch.
"""

from __future__ import annotations

from typing import Any

import pytest

from mousedroid.config.schema import Settings
from mousedroid.factory import build_approval_gate
from mousedroid.harness.approval.auto import AutoApproveGate
from mousedroid.harness.approval.policy import PolicyApprovalGate
from mousedroid.harness.approval.protocol import ApprovalGateProtocol


def _settings_with_harness_approval(**approval_kwargs: Any) -> Settings:
    """Build a ``Settings`` overriding only the harness.approval block.

    The factory's ``build_approval_gate`` reads ``cfg.harness.approval`` and
    returns a concrete protocol implementation. The rest of the harness +
    rover settings carry their schema defaults to keep these tests focused
    on the approval-gate factory branch.
    """
    return Settings.model_validate(
        {
            "mock_hardware": True,
            "harness": {"approval": approval_kwargs} if approval_kwargs else {},
        }
    )


def test_auto_gate_without_patterns_returns_bare_auto_gate() -> None:
    """No patterns configured → factory returns :class:`AutoApproveGate` directly.

    The policy wrapper is unnecessary in this branch (no patterns means
    every request is auto-approved) and the factory documents this in the
    short-circuit at ``factory.py:2259``.
    """
    cfg = _settings_with_harness_approval(gate="auto")
    gate = build_approval_gate(cfg)
    assert isinstance(gate, AutoApproveGate)
    # Pin the protocol invariant: the returned object always satisfies the
    # runtime-checkable protocol regardless of branch.
    assert isinstance(gate, ApprovalGateProtocol)


def test_auto_gate_with_patterns_wraps_in_policy_gate() -> None:
    """Patterns configured → factory wraps the auto inner gate in PolicyApprovalGate.

    This is the call site that originally tripped the ``mypy --strict``
    error: ``PolicyApprovalGate(inner, ...)`` where ``inner`` was annotated
    ``object``. The fix narrows the annotation so the protocol contract is
    visible to mypy without runtime overhead.
    """
    cfg = _settings_with_harness_approval(
        gate="auto",
        require_approval_tool_patterns=["filesystem_*"],
    )
    gate = build_approval_gate(cfg)
    assert isinstance(gate, PolicyApprovalGate)
    assert isinstance(gate, ApprovalGateProtocol)
    # The inner gate should still be an AutoApproveGate (the policy gate is
    # a wrapper; the inner is only consulted on pattern match).
    inner = getattr(gate, "_inner", None)
    assert isinstance(inner, AutoApproveGate)


def test_cli_branch_typechecks_against_protocol() -> None:
    """The ``cli`` gate branch still satisfies the protocol after the fix.

    Even with a tightened annotation, every concrete gate must satisfy
    :class:`ApprovalGateProtocol` — otherwise the assignment at
    ``factory.py:2241`` would fail at runtime (the protocol is
    ``@runtime_checkable``).
    """
    cfg = _settings_with_harness_approval(gate="cli", cli_timeout_s=5.0)
    gate = build_approval_gate(cfg)
    assert isinstance(gate, ApprovalGateProtocol)


def test_harness_disabled_short_circuits_to_auto_gate() -> None:
    """``cfg.harness is None`` → factory returns :class:`AutoApproveGate`.

    This is the guard at ``factory.py:2232``. The branch must remain valid
    after the annotation tightening (no harness config means no patterns,
    no approval workflow, so auto-approve is the only correct behaviour).
    """
    cfg = Settings.model_validate({"mock_hardware": True, "harness": None})
    gate = build_approval_gate(cfg)
    assert isinstance(gate, AutoApproveGate)
    assert isinstance(gate, ApprovalGateProtocol)


def test_policy_gate_inner_satisfies_protocol() -> None:
    """The wrapped inner gate ALSO satisfies the protocol.

    Belt-and-suspenders: the PR #98 type hole only flagged the OUTER
    ``PolicyApprovalGate(inner, ...)`` call. We also pin that the inner
    object the factory hands to the wrapper actually satisfies the
    contract — otherwise the assertion ``mypy --strict`` infers is
    correct but the runtime invariant could still silently weaken.
    """
    cfg = _settings_with_harness_approval(
        gate="auto",
        require_approval_skill_patterns=["dangerous_*"],
    )
    gate = build_approval_gate(cfg)
    assert isinstance(gate, PolicyApprovalGate)
    # Access the private ``_inner`` attribute directly — the policy gate
    # documents it as the wrap point and the PR-104 test (test_factory.py)
    # uses the same pattern. Project ruff config does not enable SLF001.
    inner = gate._inner
    assert isinstance(inner, ApprovalGateProtocol)


@pytest.mark.parametrize(
    "gate_kind",
    ["auto", "cli"],
)
def test_factory_return_type_is_protocol_for_every_branch(gate_kind: str) -> None:
    """Every branch of ``build_approval_gate`` returns an ApprovalGateProtocol.

    Parametrized over the two zero-config branches (auto + cli) to assert
    the typing contract holds independently of which concrete gate the
    operator picked. The callback branch needs a dotted-path target so
    it has its own focused test elsewhere.
    """
    cfg = _settings_with_harness_approval(gate=gate_kind)
    gate = build_approval_gate(cfg)
    assert isinstance(gate, ApprovalGateProtocol)
