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
