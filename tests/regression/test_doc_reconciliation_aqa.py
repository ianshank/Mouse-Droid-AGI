"""AQA: F-030 doc-reconciliation claims stay tied to the source of truth.

A doc fix without a pin re-drifts -- the repeated lesson of this repo's own
history (the smoke tier ran in zero CI paths for months before
``test_ci_gate_wiring_aqa.py`` existed; "tests/security is the only coverage of
the pre-egress filter" was corrected five times before it was pinned). This
file exists so the claims fixed in F-030 -- the CI job count, the
coverage-gate percentage, and ``orchestrator/CLAUDE.md``'s symbol names -- are
asserted against the tree they describe, not just corrected once in prose.

Deliberately narrow on prose: this file pins NUMBERS and specific SYMBOL NAMES,
not free-form claim text (``TestOrphanTierNarrativeAccuracy`` in
``test_ci_gate_wiring_aqa.py`` owns the "no live doc states a false claim
unqualified" pattern for that). One claim from F-030's original scope is
deliberately NOT pinned here: see "growth/-wiring claim" in this bundle's
``tasks.md`` "Explicitly deferred" section for why a sentence-scoped sweep for
that specific claim was attempted and abandoned as too fragile.
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


# A qualifying phrase near a coverage claim that means the mention is
# legitimately about the SEPARATE tools/claude_hooks gate, not the
# src/mousedroid one.
_TOOLS_HOOKS_QUALIFIER = "claude_hooks"

# Any "<N>% coverage" (optionally "<N>% line coverage") phrase -- NOT anchored
# to a specific stale value, so a future change to the real floor that some
# doc misses is still caught, the same discipline _TOTAL_JOB_COUNT already
# applies to the CI job count. A bare "<N>%" is deliberately not enough: this
# repo also states unrelated per-file claims like "100% on spec.py"
# (docs/architecture/c4-spec-harness.md), which names no "coverage" word next
# to the percentage and must not be read as a src/mousedroid floor claim.
_COVERAGE_CLAIM = re.compile(r"(\d{1,3})\s*%\s*(?:line\s+)?coverage")


def test_no_live_doc_claims_a_stale_src_coverage_floor() -> None:
    """No live surface may state a src/mousedroid coverage floor but the real one.

    Extracts every "<N>% coverage" claim and compares it against
    _real_src_coverage_floor() directly, rather than searching for the one
    specific value ("85%") this test was first written against -- pinning a
    literal stale number only catches this one historical drift (85 -> 90),
    not the next one, whichever direction it goes.
    """
    real_floor = _real_src_coverage_floor()
    offenders = []
    for doc in _SRC_COVERAGE_DOCS:
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            if _TOOLS_HOOKS_QUALIFIER in line:
                continue
            for match in _COVERAGE_CLAIM.finditer(line):
                claimed = int(match.group(1))
                if claimed != real_floor:
                    relpath = doc.relative_to(_REPO_ROOT)
                    offenders.append(f"{relpath}:{lineno} claims {claimed}%, real is {real_floor}%")
    assert not offenders, (
        "these lines state a src/mousedroid coverage floor that does not "
        f"match pyproject.toml's real fail_under ({real_floor}%): {offenders}"
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


_ORCHESTRATOR_CLAUDE_MD = _REPO_ROOT / "src" / "mousedroid" / "orchestrator" / "CLAUDE.md"

# The four symbols this doc must name and their real homes. A small, explicit
# map is the right shape here (unlike a hardcoded skills/agents roster) --
# this doc discusses a specific, stable set of core orchestrator symbols, not
# an open-ended, organically-growing collection.
_ORCHESTRATOR_REAL_SYMBOLS = {
    "MouseDroidOrchestrator": (
        _REPO_ROOT / "src" / "mousedroid" / "orchestrator" / "orchestrator.py",
        "class MouseDroidOrchestrator",
    ),
    "AutonomousOrchestrator": (
        _REPO_ROOT / "src" / "mousedroid" / "orchestrator" / "autonomous.py",
        "class AutonomousOrchestrator",
    ),
    "MouseDroidSafetyMonitor": (
        _REPO_ROOT / "src" / "mousedroid" / "safety" / "monitor.py",
        "class MouseDroidSafetyMonitor",
    ),
    "build_orchestrator": (
        _REPO_ROOT / "src" / "mousedroid" / "factory.py",
        "def build_orchestrator",
    ),
}

# Regression target: this doc used to name these four symbols, none of which
# exist in any .py file in the tree.
_ORCHESTRATOR_PHANTOM_SYMBOLS = (
    "RobotOrchestrator",
    "ConstitutionalSafetyMonitor",
)


def test_orchestrator_claude_md_names_only_real_symbols() -> None:
    """orchestrator/CLAUDE.md -- the module map read before touching the
    orchestrator -- must never regress to naming symbols that don't exist.
    """
    text = _ORCHESTRATOR_CLAUDE_MD.read_text(encoding="utf-8")

    present_phantoms = [s for s in _ORCHESTRATOR_PHANTOM_SYMBOLS if s in text]
    assert not present_phantoms, (
        f"orchestrator/CLAUDE.md names {present_phantoms}, which exist in no "
        ".py file in the tree -- this doc is read before touching the "
        "orchestrator, so a phantom symbol here steers the next edit wrong"
    )

    missing_from_doc = []
    missing_from_source = []
    for symbol, (path, definition) in _ORCHESTRATOR_REAL_SYMBOLS.items():
        if symbol not in text:
            missing_from_doc.append(symbol)
            continue
        if definition not in path.read_text(encoding="utf-8"):
            missing_from_source.append(f"{symbol} (expected {definition!r} in {path})")

    assert not missing_from_doc, (
        f"orchestrator/CLAUDE.md no longer names {missing_from_doc} -- if "
        "these were renamed or removed, update this list, not just the doc"
    )
    assert not missing_from_source, (
        f"orchestrator/CLAUDE.md names symbols that no longer resolve: "
        f"{missing_from_source} -- either they moved (update the map above) "
        "or the doc is drifting again"
    )
