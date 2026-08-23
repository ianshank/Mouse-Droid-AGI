"""AQA: F-030 doc-reconciliation numeric claims stay tied to the source of truth.

A doc fix without a pin re-drifts -- the repeated lesson of this repo's own
history (the smoke tier ran in zero CI paths for months before
``test_ci_gate_wiring_aqa.py`` existed; "tests/security is the only coverage of
the pre-egress filter" was corrected five times before it was pinned). This
file exists so the two numeric claims fixed in F-030 -- the CI job count and
the coverage-gate percentage -- are asserted against the tree they describe,
not just corrected once in prose.

Deliberately narrow: this pins NUMBERS, not the doc prose itself
(``TestOrphanTierNarrativeAccuracy`` in ``test_ci_gate_wiring_aqa.py`` already
owns the "no live doc states a false claim unqualified" pattern for prose).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CI_YML = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# Live surfaces that state the CI job count in prose. Point-in-time records
# (CHANGELOG.md, openspec/changes/**) are deliberately excluded -- they
# describe what was true when written, not current truth.
_JOB_COUNT_DOCS = (
    _REPO_ROOT / "CLAUDE.md",
    _REPO_ROOT / "docs" / "claude" / "surfaces" / "ci-gates.md",
    _REPO_ROOT / "docs" / "claude" / "surfaces" / "README.md",
)

# Live surfaces that state the src/mousedroid line-coverage floor.
_SRC_COVERAGE_DOCS = (
    _REPO_ROOT / "HARNESS_SPEC.md",
    _REPO_ROOT / "tests" / "agent.md",
    _REPO_ROOT / "docs" / "architecture" / "c4-spec-harness.md",
)

_STALE_COVERAGE = re.compile(r"85\s*%")


def _real_ci_job_count() -> int:
    data = yaml.safe_load(_CI_YML.read_text(encoding="utf-8"))
    jobs = data.get("jobs")
    assert isinstance(jobs, dict), "ci.yml has no jobs mapping"
    return len(jobs)


def _real_src_coverage_floor() -> int:
    """The src/mousedroid fail_under from pyproject.toml, as an int percent."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"^fail_under\s*=\s*(\d+)", text, re.MULTILINE)
    assert match is not None, "pyproject.toml has no [tool.coverage.report] fail_under"
    return int(match.group(1))


# Only the two TOTAL-count phrasings this repo actually uses: a parenthesised
# heading ("(16 Jobs)") and a hyphenated compound adjective ("16-job"). A bare
# "N jobs" is deliberately NOT matched -- it also catches true statements like
# "5 jobs run *(advisory)*", which names the advisory subset, not the total.
_TOTAL_JOB_COUNT = re.compile(r"\((\d+)\s*[Jj]obs?\)|(\d+)-job\b")


def test_ci_job_count_docs_match_the_real_workflow() -> None:
    """Every live surface stating the TOTAL CI job count must state the real one.

    Regression target: CLAUDE.md/ci-gates.md/README.md all said "12 Jobs"
    while ci.yml defined 16 -- and six of the twelve job names CLAUDE.md
    listed (secret-scan, skills, test-fast, validate, regression, package)
    were not job names at all, only step names inside other jobs.
    """
    real_count = _real_ci_job_count()
    offenders = []
    for doc in _JOB_COUNT_DOCS:
        text = doc.read_text(encoding="utf-8")
        stale = {n for pair in _TOTAL_JOB_COUNT.findall(text) for n in pair if n}
        wrong = sorted(n for n in stale if int(n) != real_count)
        if wrong:
            offenders.append(f"{doc.relative_to(_REPO_ROOT)}: claims {wrong}, real is {real_count}")
    assert not offenders, "\n".join(offenders)


# A qualifying phrase near "85%" that means the mention is legitimately about
# the SEPARATE tools/claude_hooks gate, not a stale src/mousedroid claim.
_TOOLS_HOOKS_QUALIFIER = "claude_hooks"


def test_no_live_doc_claims_a_stale_src_coverage_floor() -> None:
    """No live surface may state 85% for the src/mousedroid gate -- it is 90%.

    85% is the CORRECT floor for tools/claude_hooks (a separate, narrower
    gate, pinned below) -- so this checks each LINE containing "85%", not the
    whole file, and only flags a line that does not also qualify itself as
    being about tools/claude_hooks.
    """
    real_floor = _real_src_coverage_floor()
    assert real_floor != 85, (
        "pyproject.toml's src/mousedroid fail_under is now 85 -- if that is "
        "intentional, this test (and its docs) needs updating, not deleting"
    )
    offenders = []
    for doc in _SRC_COVERAGE_DOCS:
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            if _STALE_COVERAGE.search(line) and _TOOLS_HOOKS_QUALIFIER not in line:
                offenders.append(f"{doc.relative_to(_REPO_ROOT)}:{lineno}")
    assert not offenders, (
        f"these lines still say 85% unqualified, where they mean the "
        f"src/mousedroid gate (now {real_floor}%): {offenders}"
    )


def test_tools_claude_hooks_coverage_floor_is_still_85() -> None:
    """The OTHER gate is genuinely 85% -- pin it so nobody 'fixes' it to 90%.

    The two-gate asymmetry (90% src/mousedroid, 85% tools/claude_hooks) is
    deliberate and stated in .claude/workforce.yaml; a future editor
    "correcting" tools/claude_hooks to match src/mousedroid would silently
    tighten a gate this test exists to keep intentional.
    """
    from tools.claude_hooks.config import load_config

    assert load_config().coverage.tools_line_min == 85
