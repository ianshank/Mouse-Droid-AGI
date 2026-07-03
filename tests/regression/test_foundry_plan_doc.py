# tests/regression/test_foundry_plan_doc.py
"""AQA: claude-code-foundry plan-document hygiene.

Locks the contracts the foundry plan promises its future executing session:

* Every backtick path in the *Mouse-Droid-AGI-scoped* sections (headings that
  name this repo) exists on disk — the plan's "prior art" and consumer-
  migration pointers must not rot when files move.
* No hardcoded IPv4 anywhere in the document (same environment-agnostic rule
  the skill-command validator enforces; the rule lives in exactly one place).
* Every ``[AUDIT-N]`` placeholder is registered in the "Prerequisites &
  unverified facts" section AND consumed by at least one other section — an
  orphaned tag means an unverified fact silently lost its verification task.

Foundry-repo-relative paths (``tools/validate.py`` etc.) are intentionally NOT
existence-checked here: they describe a repository that does not exist yet and
live outside the Mouse-Droid-AGI-scoped sections this test sweeps.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from tools.validate_skill_commands import find_hardcoded_hosts, referenced_repo_paths

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLANS_DIR = _REPO_ROOT / "docs" / "superpowers" / "plans"
# Discovered, not hardcoded: the dated filename may change on a re-issue of the
# plan; the slug is the stable contract.
_PLAN_GLOB = "*claude-code-foundry*.md"
_SECTION_RE = re.compile(r"^## ", re.MULTILINE)
_AUDIT_TAG_RE = re.compile(r"\[AUDIT-(\d+)\]")
_LOCAL_SECTION_MARKER = "Mouse-Droid-AGI"
_PREREQ_HEADING_MARKER = "Prerequisites"


def _plan_text() -> str:
    matches = sorted(_PLANS_DIR.glob(_PLAN_GLOB))
    assert len(matches) == 1, f"expected exactly one foundry plan doc, found: {matches}"
    return matches[0].read_text(encoding="utf-8")


def _sections(text: str) -> dict[str, str]:
    """Split the document into ``{heading_line: body}`` on ``## `` headings."""
    out: dict[str, str] = {}
    parts = _SECTION_RE.split(text)
    for part in parts[1:]:  # parts[0] is the preamble before the first section
        heading, _, body = part.partition("\n")
        out[heading.strip()] = body
    return out


def test_plan_doc_exists() -> None:
    assert sorted(_PLANS_DIR.glob(_PLAN_GLOB)), f"no foundry plan doc under {_PLANS_DIR}"


def test_no_hardcoded_hosts() -> None:
    hosts = find_hardcoded_hosts(_plan_text())
    assert hosts == [], f"hardcoded IPv4 literals in foundry plan doc: {hosts}"


def test_local_repo_path_references_exist() -> None:
    sections = _sections(_plan_text())
    local_bodies = [body for head, body in sections.items() if _LOCAL_SECTION_MARKER in head]
    assert local_bodies, "no Mouse-Droid-AGI-scoped sections found — heading convention changed?"
    missing = [
        ref
        for body in local_bodies
        for ref in referenced_repo_paths(body)
        if not (_REPO_ROOT / ref).exists()
    ]
    assert missing == [], f"foundry plan references non-existent local paths: {missing}"


def test_local_sections_reference_the_reused_validator() -> None:
    # The plan's WS-F2 generalizes this exact tool; if the pointer disappears
    # the executing session loses the prior-art trailhead.
    sections = _sections(_plan_text())
    local_text = "".join(body for head, body in sections.items() if _LOCAL_SECTION_MARKER in head)
    assert "tools/validate_skill_commands.py" in local_text


@pytest.mark.parametrize("tag_number", [1, 2, 3, 4])
def test_audit_tags_registered_and_consumed(tag_number: int) -> None:
    text = _plan_text()
    sections = _sections(text)
    prereq_bodies = [body for head, body in sections.items() if _PREREQ_HEADING_MARKER in head]
    assert len(prereq_bodies) == 1, "expected exactly one Prerequisites section"
    tag = f"[AUDIT-{tag_number}]"
    assert tag in prereq_bodies[0], f"{tag} not registered in the Prerequisites table"
    elsewhere = any(
        tag in body for head, body in sections.items() if _PREREQ_HEADING_MARKER not in head
    )
    assert elsewhere, f"{tag} registered but never consumed by any work stream / section"


def test_no_unregistered_audit_tags() -> None:
    # The inverse guard: a tag used in a WS but absent from the registry table
    # would be an unverified fact with no verification owner.
    text = _plan_text()
    used = {int(n) for n in _AUDIT_TAG_RE.findall(text)}
    assert used == {1, 2, 3, 4}, f"unexpected AUDIT tag set: {sorted(used)}"
