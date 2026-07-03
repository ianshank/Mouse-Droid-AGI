"""Regression pins for the NEXT_STEPS truth reconciliation (F-016, WS-1).

Pins the three load-bearing outcomes of the 2026-07-03 reconciliation:

* NEXT_STEPS.md stays inside the doc-hygiene budget (it was 37 KB / 72 ✅
  before the split — landed work now lives in CHANGELOG.md),
* the T3/arm contradiction is resolved as pause-at-T2 with an explicit
  unfreeze condition (CONFIRM-FIRST #1 decision),
* the "Phase 5" vocabulary collision between the root roadmap (physics sim,
  deferred) and the legacy v0.3.0 execution plan (LLM gateway, done) is
  disambiguated in both files.

Uses the same ``check_doc`` helper the advisory CLI uses — one budget
definition, two consumers.
"""

from __future__ import annotations

from pathlib import Path

from tools.doc_hygiene import _DEFAULT_MAX_BYTES, _DEFAULT_MAX_DONE_MARKS, check_doc

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NEXT_STEPS = _REPO_ROOT / "NEXT_STEPS.md"
_PLANNING_NEXT_STEPS = _REPO_ROOT / "docs" / "planning" / "NEXT_STEPS.md"
_CHANGELOG = _REPO_ROOT / "CHANGELOG.md"
_RUNBOOK = _REPO_ROOT / "docs" / "runbooks" / "claude-code-on-jetson.md"


def test_next_steps_within_hygiene_budget() -> None:
    warnings = check_doc(
        _NEXT_STEPS,
        max_bytes=_DEFAULT_MAX_BYTES,
        max_done_marks=_DEFAULT_MAX_DONE_MARKS,
    )
    assert warnings == [], f"NEXT_STEPS.md re-drifted: {warnings}"


def test_arm_arc_is_paused_at_t2_with_unfreeze_condition() -> None:
    text = _NEXT_STEPS.read_text(encoding="utf-8")
    assert "PAUSED at T2" in text, "the pause-at-T2 decision was removed"
    assert "Unfreeze condition" in text, "the pause must carry an explicit unfreeze condition"
    assert "Next-in-arc: T3" not in text, "the contradictory T3 next-in-arc line is back"


def test_phase_vocabulary_is_disambiguated() -> None:
    root = _NEXT_STEPS.read_text(encoding="utf-8")
    assert "Phase vocabulary" in root, "root doc must claim the Physical-AI phase numbering"

    planning = _PLANNING_NEXT_STEPS.read_text(encoding="utf-8")
    assert "Phase 5 (legacy v0.3.0 numbering)" in planning, (
        "docs/planning/NEXT_STEPS.md must qualify its Phase 5 as legacy "
        "numbering, distinct from the Physical-AI Phase 5 (physics sim)"
    )


def test_reconciliation_destinations_exist() -> None:
    assert "Historical record — reconciled from NEXT_STEPS.md" in _CHANGELOG.read_text(
        encoding="utf-8"
    ), "the CHANGELOG historical-record section is the split's destination"
    assert _RUNBOOK.is_file(), "the extracted Claude-Code-on-Jetson runbook must exist"


def test_frozen_skills_carry_status_frontmatter() -> None:
    for skill in ("robot-arm-trainer", "sim-test", "train-policy"):
        text = (_REPO_ROOT / ".claude" / "skills" / skill / "SKILL.md").read_text(encoding="utf-8")
        assert "status: frozen" in text, f"{skill} lost its frozen status"
        assert "unfreeze:" in text, f"{skill} must document its unfreeze condition"
