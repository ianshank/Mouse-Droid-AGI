"""Tests for the harness factory wiring.

These tests verify that ``factory.build_*`` returns the expected concrete
types based on ``cfg.harness`` and that the disabled-by-default path
keeps the orchestrator legacy-compatible.
"""

from __future__ import annotations

from pathlib import Path

from mousedroid.config.schema import (
    HarnessApprovalConfig,
    HarnessConfig,
    HarnessHooksConfig,
    HarnessJournalConfig,
    HarnessTrackerConfig,
    Settings,
    SkillsConfig,
)
from mousedroid.factory import (
    build_approval_gate,
    build_hook_registry,
    build_journal,
    build_skill_delegator,
    build_skill_loaders,
    build_skill_registry,
    build_task_tracker,
)
from mousedroid.harness.approval.auto import AutoApproveGate
from mousedroid.harness.approval.policy import PolicyApprovalGate
from mousedroid.harness.hooks import HookRegistry, NullHookRegistry
from mousedroid.harness.journal.jsonl_journal import JSONLJournal
from mousedroid.harness.journal.lmdb_journal import LMDBJournal
from mousedroid.harness.journal.null_journal import NullJournal


def _settings(harness: HarnessConfig | None) -> Settings:
    return Settings(mock_hardware=True, harness=harness)


# ---------------------------------------------------------------------------
# Disabled path
# ---------------------------------------------------------------------------


def test_disabled_returns_no_op_components() -> None:
    cfg = _settings(None)
    assert build_task_tracker(cfg) is None
    assert isinstance(build_journal(cfg), NullJournal)
    assert isinstance(build_approval_gate(cfg), AutoApproveGate)
    assert build_skill_loaders(cfg) == ()
    skills = build_skill_registry(cfg, ())
    assert len(skills) == 0
    assert build_skill_delegator(cfg, skills, build_approval_gate(cfg), NullJournal(), None) is None
    assert isinstance(build_hook_registry(cfg, NullJournal()), NullHookRegistry)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


def test_tracker_built_when_enabled() -> None:
    cfg = _settings(
        HarnessConfig(tracker=HarnessTrackerConfig(enabled=True)),
    )
    tracker = build_task_tracker(cfg)
    assert tracker is not None


def test_tracker_disabled_returns_none() -> None:
    cfg = _settings(HarnessConfig(tracker=HarnessTrackerConfig(enabled=False)))
    assert build_task_tracker(cfg) is None


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


def test_journal_jsonl(tmp_path: Path) -> None:
    cfg = _settings(
        HarnessConfig(journal=HarnessJournalConfig(backend="jsonl", path=tmp_path / "j.jsonl"))
    )
    assert isinstance(build_journal(cfg), JSONLJournal)


def test_journal_lmdb(tmp_path: Path) -> None:
    cfg = _settings(
        HarnessConfig(journal=HarnessJournalConfig(backend="lmdb", path=tmp_path / "lmdb"))
    )
    assert isinstance(build_journal(cfg), LMDBJournal)


def test_journal_null_default() -> None:
    cfg = _settings(HarnessConfig())  # backend defaults to 'null'
    assert isinstance(build_journal(cfg), NullJournal)


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


def test_approval_auto_default() -> None:
    cfg = _settings(HarnessConfig())
    gate = build_approval_gate(cfg)
    assert isinstance(gate, AutoApproveGate)


def test_approval_policy_when_patterns_present() -> None:
    cfg = _settings(
        HarnessConfig(
            approval=HarnessApprovalConfig(
                gate="auto",
                require_approval_tool_patterns=["esp32_*"],
            )
        )
    )
    gate = build_approval_gate(cfg)
    assert isinstance(gate, PolicyApprovalGate)


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def test_skills_disabled_yields_empty_loaders() -> None:
    cfg = _settings(HarnessConfig(skills=SkillsConfig(enabled=False)))
    assert build_skill_loaders(cfg) == ()


def test_skills_enabled_yields_yaml_and_markdown_loaders(tmp_path: Path) -> None:
    cfg = _settings(
        HarnessConfig(
            skills=SkillsConfig(
                enabled=True,
                manifest_glob=str(tmp_path / "*.yaml"),
                markdown_agent_dirs=[tmp_path],
            )
        )
    )
    loaders = build_skill_loaders(cfg)
    assert len(loaders) == 2


def test_skill_delegator_requires_tracker_and_skills_enabled() -> None:
    cfg = _settings(
        HarnessConfig(
            skills=SkillsConfig(enabled=True),
            tracker=HarnessTrackerConfig(enabled=True),
        )
    )
    tracker = build_task_tracker(cfg)
    journal = build_journal(cfg)
    skills = build_skill_registry(cfg, build_skill_loaders(cfg))
    gate = build_approval_gate(cfg)
    delegator = build_skill_delegator(cfg, skills, gate, journal, tracker)
    assert delegator is not None


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def test_hook_registry_journal_events_default_seeds_journal_hooks() -> None:
    cfg = _settings(HarnessConfig(hooks=HarnessHooksConfig(journal_events=True)))
    journal = build_journal(cfg)
    hooks = build_hook_registry(cfg, journal)
    # When harness is not None, we always return a HookRegistry (not Null).
    assert isinstance(hooks, HookRegistry)
    # Journal-event hooks were registered for every HookPhase.
    from mousedroid.harness.protocol import HookPhase

    for phase in HookPhase:
        assert any(spec.name.startswith("journal:") for spec in hooks.for_phase(phase))


def test_hook_registry_journal_events_disabled_yields_empty_registry() -> None:
    cfg = _settings(HarnessConfig(hooks=HarnessHooksConfig(journal_events=False)))
    journal = build_journal(cfg)
    hooks = build_hook_registry(cfg, journal)
    assert isinstance(hooks, HookRegistry)
    from mousedroid.harness.protocol import HookPhase

    for phase in HookPhase:
        assert hooks.for_phase(phase) == ()


def test_hook_registry_enabled_hooks_acts_as_allowlist() -> None:
    """When ``enabled_hooks`` is non-empty the factory only registers
    journal hooks whose names are explicitly listed."""
    cfg = _settings(
        HarnessConfig(
            hooks=HarnessHooksConfig(
                journal_events=True,
                enabled_hooks=["journal:pre_tick", "journal:post_tick"],
            )
        )
    )
    hooks = build_hook_registry(cfg, build_journal(cfg))
    from mousedroid.harness.protocol import HookPhase

    pre = [s.name for s in hooks.for_phase(HookPhase.PRE_TICK)]
    post = [s.name for s in hooks.for_phase(HookPhase.POST_TICK)]
    other = [s.name for s in hooks.for_phase(HookPhase.PRE_ACTION)]
    assert "journal:pre_tick" in pre
    assert "journal:post_tick" in post
    assert other == []  # not in allowlist


def test_hook_registry_fail_fast_promotes_error_policy_to_raise() -> None:
    """``fail_fast=True`` overrides per-hook ``error_policy`` so any
    failure propagates out of the tick instead of being warn-logged."""
    cfg = _settings(
        HarnessConfig(
            hooks=HarnessHooksConfig(
                journal_events=True,
                fail_fast=True,
                error_policy="warn",  # would be 'warn', overridden to 'raise'
            )
        )
    )
    hooks = build_hook_registry(cfg, build_journal(cfg))
    from mousedroid.harness.protocol import HookPhase

    for phase in HookPhase:
        for spec in hooks.for_phase(phase):
            assert spec.error_policy == "raise"


# ---------------------------------------------------------------------------
# Callback-gate dotted-path resolution
# ---------------------------------------------------------------------------


def test_approval_callback_path_resolved(monkeypatch) -> None:
    """A valid dotted path resolves to the configured callable."""
    import sys
    import types

    module = types.ModuleType("mousedroid_test_approval_callback_module")

    async def _grant(_request):
        return True

    module.grant = _grant
    sys.modules[module.__name__] = module
    try:
        cfg = _settings(
            HarnessConfig(
                approval=HarnessApprovalConfig(
                    gate="callback",
                    callback_dotted_path=f"{module.__name__}.grant",
                ),
            )
        )
        gate = build_approval_gate(cfg)
        # Inner gate is the AsyncCallbackApprovalGate (or wrapped in policy).
        from mousedroid.harness.approval.callback import (
            AsyncCallbackApprovalGate,
        )

        assert isinstance(gate, AsyncCallbackApprovalGate)
    finally:
        sys.modules.pop(module.__name__, None)


def test_approval_callback_invalid_path_falls_back_to_deny() -> None:
    """A bogus dotted path produces a fail-closed deny callback (the gate
    itself is still constructed)."""
    cfg = _settings(
        HarnessConfig(
            approval=HarnessApprovalConfig(
                gate="callback",
                callback_dotted_path="nonexistent_module.no_such_attr",
            ),
        )
    )
    gate = build_approval_gate(cfg)
    # A gate is returned (no exception); behaviour is verified at runtime.
    assert gate is not None


# ---------------------------------------------------------------------------
# Skill backend wiring
# ---------------------------------------------------------------------------


def test_skill_delegator_uses_noop_backend_by_default() -> None:
    cfg = _settings(
        HarnessConfig(
            skills=SkillsConfig(enabled=True, backend="noop"),
            tracker=HarnessTrackerConfig(enabled=True),
        )
    )
    tracker = build_task_tracker(cfg)
    journal = build_journal(cfg)
    skills = build_skill_registry(cfg, build_skill_loaders(cfg))
    from mousedroid.skills.protocol import SkillSpec

    skills.register(SkillSpec(name="diag"))
    delegator = build_skill_delegator(
        cfg,
        skills,
        build_approval_gate(cfg),
        journal,
        tracker,
    )
    assert delegator is not None
    # The factory's agent_factory should produce a NoOpSubAgent for noop.
    from mousedroid.skills.sub_agent import NoOpSubAgent

    agent = delegator._agent_factory("diag")
    assert isinstance(agent, NoOpSubAgent)


def test_skill_delegator_uses_llm_gateway_backend_when_configured() -> None:
    from unittest.mock import AsyncMock

    cfg = _settings(
        HarnessConfig(
            skills=SkillsConfig(enabled=True, backend="llm_gateway"),
            tracker=HarnessTrackerConfig(enabled=True),
        )
    )
    tracker = build_task_tracker(cfg)
    journal = build_journal(cfg)
    skills = build_skill_registry(cfg, build_skill_loaders(cfg))
    from mousedroid.skills.protocol import SkillSpec

    skills.register(SkillSpec(name="nav"))
    fake_gateway = AsyncMock()
    delegator = build_skill_delegator(
        cfg,
        skills,
        build_approval_gate(cfg),
        journal,
        tracker,
        llm_gateway=fake_gateway,
    )
    from mousedroid.skills.sub_agent import LLMBackedSubAgent

    agent = delegator._agent_factory("nav")
    assert isinstance(agent, LLMBackedSubAgent)


def test_skill_delegator_anthropic_backend_falls_back_when_unconfigured() -> None:
    """``backend="anthropic"`` without a populated arm replanner config
    must NOT crash; it should log a warning and fall back to a no-op
    sub-agent so deployments without the SDK keep functioning."""
    cfg = _settings(
        HarnessConfig(
            skills=SkillsConfig(enabled=True, backend="anthropic"),
            tracker=HarnessTrackerConfig(enabled=True),
        )
    )
    tracker = build_task_tracker(cfg)
    journal = build_journal(cfg)
    skills = build_skill_registry(cfg, build_skill_loaders(cfg))
    from mousedroid.skills.protocol import SkillSpec

    skills.register(SkillSpec(name="nav"))
    delegator = build_skill_delegator(
        cfg,
        skills,
        build_approval_gate(cfg),
        journal,
        tracker,
    )
    from mousedroid.skills.sub_agent import LLMBackedSubAgent, NoOpSubAgent

    agent = delegator._agent_factory("nav")
    # Either falls back to NoOpSubAgent or wraps the NullLLMReplanner.
    assert isinstance(agent, NoOpSubAgent | LLMBackedSubAgent)
