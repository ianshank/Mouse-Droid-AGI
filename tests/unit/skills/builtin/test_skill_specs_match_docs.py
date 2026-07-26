# tests/unit/skills/builtin/test_skill_specs_match_docs.py
"""Enforce the builtin SkillSpec <-> docs/openclaw_skills SKILL.md pairing.

The module docstring in ``src/mousedroid/skills/builtin/__init__.py`` promises
this test exists. It pins that every builtin spec has a publishable SKILL.md
whose leading H1 heading (``# <name>``) matches the spec name, so the two never
drift. These docs are plain Markdown and intentionally carry no YAML
front-matter.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from mousedroid.skills.builtin import all_builtin_specs
from mousedroid.skills.protocol import SkillSpec

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DOCS_ROOT = _REPO_ROOT / "docs" / "openclaw_skills"
# all_builtin_specs() is typed tuple[object, ...]; cast so spec.name is typed
# (no inline type:ignore — keeps WS4's suppression purge honest).
_SPECS = [cast(SkillSpec, s) for s in all_builtin_specs()]


@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: s.name)
def test_every_builtin_spec_has_matching_skill_doc(spec: SkillSpec) -> None:
    doc = _DOCS_ROOT / spec.name / "SKILL.md"
    assert doc.is_file(), f"missing publishable doc for builtin skill {spec.name!r}: {doc}"
    text = doc.read_text(encoding="utf-8")
    # The docs are plain Markdown whose leading H1 == the skill name (verified:
    # docs/openclaw_skills/mousedroid-navigate/SKILL.md:1 == "# mousedroid-navigate").
    # Do NOT assert YAML front-matter — these docs intentionally have none.
    # Match the FIRST non-blank line exactly (an H1), not a substring search:
    # a substring would false-positive on the name appearing in a code block or
    # a later heading, and miss a wrong/absent leading H1.
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    assert first_line.strip() == f"# {spec.name}", (
        f"SKILL.md leading H1 must be '# {spec.name}', got {first_line.strip()!r}"
    )


def test_no_orphan_skill_docs() -> None:
    """Every published SKILL.md dir maps to a registered builtin spec."""
    spec_names = {s.name for s in _SPECS}
    doc_dirs = {p.parent.name for p in _DOCS_ROOT.glob("*/SKILL.md")}
    assert doc_dirs == spec_names, f"doc/spec set mismatch: {doc_dirs ^ spec_names}"
