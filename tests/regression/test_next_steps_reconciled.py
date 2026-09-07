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

import re
from pathlib import Path

import yaml
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
    assert "CHARTER §5 **M6**" in root, "root doc must split CHARTER M6 from Phase 6 LoRA"
    assert "LoRA" in root

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


_CURRENT_HEADER = "## ⚡ Current Next Steps"
_CHARTER = _REPO_ROOT / "docs" / "CHARTER.md"


def _current_next_steps_section(text: str) -> str:
    """Return the Current Next Steps body (up to the next H2)."""
    start = text.index(_CURRENT_HEADER)
    rest = text[start + len(_CURRENT_HEADER) :]
    nxt = rest.find("\n## ")
    return rest if nxt < 0 else rest[:nxt]


def test_current_next_steps_contains_no_landed_token() -> None:
    """F-038: LANDED rows in Current Next Steps re-drift into a changelog.

    Operator leftovers use the words "operator leftover" / "Ops leftover",
    never the LANDED token.
    """
    section = _current_next_steps_section(_NEXT_STEPS.read_text(encoding="utf-8"))
    offenders = [line for line in section.splitlines() if "LANDED" in line]
    assert not offenders, f"Current Next Steps grew LANDED rows:\n{offenders}"


def test_done_catalog_ids_in_current_are_operator_leftovers() -> None:
    """A done F-id in Current Next Steps must be labelled leftover, not next work."""
    section = _current_next_steps_section(_NEXT_STEPS.read_text(encoding="utf-8"))
    catalog = yaml.safe_load((_REPO_ROOT / "features.yaml").read_text(encoding="utf-8"))
    done = {
        feat["id"]
        for feat in catalog["features"]
        if feat.get("status") == "done"
    }
    offenders: list[str] = []
    for line in section.splitlines():
        ids = re.findall(r"\bF-\d{3}\b", line)
        if not ids:
            continue
        if "leftover" in line.lower():
            continue
        leaked = [fid for fid in ids if fid in done]
        if leaked:
            offenders.append(f"{leaked}: {line.strip()}")
    assert not offenders, (
        "done catalog ids in Current Next Steps must sit on an "
        "'operator leftover' / 'Ops leftover' line:\n  " + "\n  ".join(offenders)
    )


def test_charter_points_at_root_next_steps_not_may16_plan() -> None:
    """F-038: CHARTER §5 living plan is root NEXT_STEPS.md, not May-16 IMPLEMENTATION_PLAN."""
    text = _CHARTER.read_text(encoding="utf-8")
    assert "docs/planning/IMPLEMENTATION_PLAN.md is authoritative" not in text
    assert "[`NEXT_STEPS.md`](../NEXT_STEPS.md)" in text
    assert "LoRA" in text
    assert "May-16" in text


def test_frozen_skills_carry_status_frontmatter() -> None:
    for skill in ("robot-arm-trainer", "sim-test", "train-policy"):
        text = (_REPO_ROOT / ".claude" / "skills" / skill / "SKILL.md").read_text(encoding="utf-8")
        assert "status: frozen" in text, f"{skill} lost its frozen status"
        assert "unfreeze:" in text, f"{skill} must document its unfreeze condition"
