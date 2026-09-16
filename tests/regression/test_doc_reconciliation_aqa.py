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
import subprocess
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CI_YML = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_ARCHITECTURE_DIR = _REPO_ROOT / "docs" / "architecture"
_ADR_LOG = _ARCHITECTURE_DIR / "adr-log.md"

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

    The tools_hooks exemption is scoped to the SPECIFIC claimed value, not the
    whole line: HARNESS_SPEC.md:303 and tests/agent.md:6 both state the real
    90% src/mousedroid claim and the 85% tools_hooks claim on the same line.
    An earlier version of this test skipped any line containing "claude_hooks"
    at all, which -- caught by a Copilot review comment on this same PR --
    exempted the legitimate 90% claim right along with the 85% one, on
    exactly the two lines this test exists to check.
    """
    from tools.claude_hooks.config import load_config

    real_floor = _real_src_coverage_floor()
    tools_hooks_floor = load_config().coverage.tools_line_min
    offenders = []
    for doc in _SRC_COVERAGE_DOCS:
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            for match in _COVERAGE_CLAIM.finditer(line):
                claimed = int(match.group(1))
                if claimed == real_floor:
                    continue
                if claimed == tools_hooks_floor and _TOOLS_HOOKS_QUALIFIER in line:
                    continue
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
        _REPO_ROOT / "src" / "mousedroid" / "factory" / "orchestrator.py",
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


# ---------------------------------------------------------------------------
# Advisory-job count (the drift the bare-"N jobs" exclusion above made invisible)
# ---------------------------------------------------------------------------
#
# ``_TOTAL_JOB_COUNT`` deliberately does NOT match a bare "N jobs", because that
# phrasing also catches the true statement "5 jobs run *(advisory)*" -- which
# names the advisory SUBSET, not the total. Correct as far as it goes, but it
# left the subset count itself entirely unpinned, and it promptly drifted:
# ``docs/claude/surfaces/ci-gates.md`` claimed six advisory jobs while the root
# ``CLAUDE.md`` said five and ``ci.yml`` carried five, after the ``security``
# job was promoted advisory -> blocking on 2026-09-16. Two live surfaces
# contradicting each other, with the source of truth agreeing with neither by
# accident. This pins the subset the same way the total is pinned.

# Point-in-time records: they describe what was true when written, and several
# QUOTE a past count verbatim while explaining an earlier drift. Rewriting them
# to match today's tree would destroy the record. Same exclusion rationale as
# _JOB_COUNT_DOCS above, widened to the dated planning/progress surfaces --
# docs/planning/TECH_DEBT_REMEDIATION_PLAN.md states "6 advisory jobs" as its
# declared baseline (dddc16c), which was true at that baseline.
_POINT_IN_TIME_DOCS = (
    "CHANGELOG.md",
    "progress.md",
    "openspec/",
    "docs/planning/",
    "docs/analysis/",
)

# A line only states an advisory count if it also NAMES the advisory concept --
# otherwise "15 jobs total" reads as an advisory claim. Required on the line.
_ADVISORY_MARKER = re.compile(r"advisory|continue-on-error", re.IGNORECASE)

# The two phrasings this repo uses for the advisory SUBSET. Kept tight for the
# same reason _TOTAL_JOB_COUNT is: "(\d+)\s+jobs?" alone matches "15 jobs
# total, 5 advisory" (docs/analysis/...) and would read the TOTAL as the subset.
_ADVISORY_COUNT_CLAIMS = (
    re.compile(r"(\d+)\s+advisory\s+(?:job|stage)s?\b", re.IGNORECASE),
    re.compile(r"(\d+)\s+jobs?\s+(?:run|carry|are)\b", re.IGNORECASE),
)

_CONTINUE_ON_ERROR_LITERAL = "continue-on-error: true"


def _strip_comments(text: str) -> str:
    """Drop whole-line ``#`` comments, keeping ``#`` inside a value.

    Mirrors the helper of the same name in ``test_ci_gate_wiring_aqa.py``
    (duplicated rather than imported: that module is a sibling test, not a
    shared fixture, and importing it would drag its whole collection in).

    Load-bearing here, not decorative. ``grep -c 'continue-on-error: true'``
    over the raw ``ci.yml`` returns **6** while only **5** jobs carry it: the
    promotion-status comment above the ``onnx-world-model-extras`` job quotes
    the literal inside a ``#`` line. So a substring-counting derivation is
    satisfiable by a comment -- it would have "derived" the very wrong number
    this test exists to catch, and would keep on deriving it after a real
    advisory job was promoted, as long as the prose comment survived.
    """
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _real_advisory_job_count() -> int:
    """How many ``ci.yml`` jobs actually carry ``continue-on-error: true``.

    Derived from the parsed jobs mapping -- never hardcoded, so promoting or
    demoting a job updates this automatically and the docs are what break.
    """
    data = yaml.safe_load(_CI_YML.read_text(encoding="utf-8"))
    jobs = data.get("jobs")
    assert isinstance(jobs, dict), "ci.yml has no jobs mapping"
    return sum(1 for spec in jobs.values() if spec.get("continue-on-error") is True)


def _tracked_docs() -> list[Path]:
    """Every tracked ``*.md``, from ``git ls-files``.

    Discovery rather than a hardcoded roster, following ``_tracked_docs`` in
    ``test_ci_gate_wiring_aqa.py``: a new surface that states an advisory count
    is covered the moment it is committed, with no list to remember to extend.
    """
    result = subprocess.run(
        ["git", "ls-files", "--", "*.md"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [_REPO_ROOT / line for line in result.stdout.splitlines() if line.strip()]


def _is_point_in_time(relpath: str) -> bool:
    return any(relpath == prefix or relpath.startswith(prefix) for prefix in _POINT_IN_TIME_DOCS)


def test_advisory_job_count_derivation_is_not_comment_satisfiable() -> None:
    """The derivation and a comment-stripped literal count must agree.

    Cross-checks the two independent ways of asking the question. They diverge
    only if (a) a ``continue-on-error: true`` was added at STEP level rather
    than job level -- legitimate, but then :func:`_real_advisory_job_count`
    undercounts what a reader greps, and the docs need to say which they mean;
    or (b) the parse and the text genuinely disagree, which is a ci.yml bug.
    Either way it wants a human, not a silently-passing pin.
    """
    parsed = _real_advisory_job_count()
    stripped = _strip_comments(_CI_YML.read_text(encoding="utf-8"))
    literal = stripped.count(_CONTINUE_ON_ERROR_LITERAL)
    assert literal == parsed, (
        f"ci.yml's parsed job-level advisory count is {parsed} but the "
        f"comment-stripped text contains {literal} occurrences of "
        f"{_CONTINUE_ON_ERROR_LITERAL!r} -- a step-level continue-on-error, or "
        "a real ci.yml inconsistency. Resolve it rather than relaxing this."
    )
    raw = _CI_YML.read_text(encoding="utf-8").count(_CONTINUE_ON_ERROR_LITERAL)
    assert raw >= literal, "comment stripping added occurrences, which is impossible"


def test_live_docs_state_the_real_advisory_job_count() -> None:
    """No live surface may state an advisory-job count but the real one.

    Regression target: ``docs/claude/surfaces/ci-gates.md`` said "6 jobs carry
    ``continue-on-error: true``" after ``security`` was promoted to blocking,
    contradicting the root ``CLAUDE.md``'s "5 jobs run *(advisory)*" on the
    same fact. ``advisory_stages.yaml`` is machine-checked by
    ``scripts/check_advisory_promotions.py``; this prose was not, which is why
    the promotion could update the tracker and leave the narrative behind.
    """
    real_count = _real_advisory_job_count()
    offenders: list[str] = []
    for doc in _tracked_docs():
        relpath = doc.relative_to(_REPO_ROOT).as_posix()
        if _is_point_in_time(relpath):
            continue
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            if not _ADVISORY_MARKER.search(line):
                continue
            claimed = {n for pattern in _ADVISORY_COUNT_CLAIMS for n in pattern.findall(line)}
            wrong = sorted(n for n in claimed if int(n) != real_count)
            if wrong:
                offenders.append(f"{relpath}:{lineno} claims {wrong}")
    assert not offenders, (
        "these live lines state an advisory-job count that does not match "
        f"ci.yml's real count ({real_count} jobs carrying continue-on-error: "
        f"true): {offenders}"
    )


# ---------------------------------------------------------------------------
# ADR log <-> ADR files on disk (bidirectional)
# ---------------------------------------------------------------------------
#
# ADR-017 (God-Files Decomposition) shipped, is referenced by name from
# orchestrator/CLAUDE.md, telemetry/CLAUDE.md and ADR-018, and was absent from
# adr-log.md's table entirely -- the table ran 004..016 + l4t-container. The log
# is the index an engineer reads to find out what has already been decided, so a
# missing row is a decision that effectively does not exist.

# Markdown link targets inside adr-log.md that point at an ADR file. Matches the
# table's own `[017](ADR-017-god-files-decomposition.md)` link form; the
# numbering note's `[adr/TEMPLATE.md]` link does not match the ADR-* prefix.
_ADR_LOG_LINK = re.compile(r"\]\((ADR-[^)]+\.md)\)")


def _adr_files_on_disk() -> set[str]:
    """Every ``docs/architecture/ADR-*.md`` filename."""
    return {path.name for path in _ARCHITECTURE_DIR.glob("ADR-*.md")}


def _adr_files_linked_from_log() -> set[str]:
    return set(_ADR_LOG_LINK.findall(_ADR_LOG.read_text(encoding="utf-8")))


def test_every_adr_on_disk_has_a_row_in_the_adr_log() -> None:
    """An ADR nobody can find from the index is an undiscoverable decision."""
    missing = sorted(_adr_files_on_disk() - _adr_files_linked_from_log())
    assert not missing, (
        f"these ADRs exist in docs/architecture/ but adr-log.md links to none "
        f"of them: {missing} -- add a table row (ADR, Title, Status, Date, "
        "Area). The log is the index; an unlisted ADR is a decision a future "
        "engineer cannot find, so they will re-litigate or contradict it."
    )


def test_every_adr_the_log_links_to_exists_on_disk() -> None:
    """The other direction: no row may point at a file that is not there.

    A dead link in the index is worse than a missing row -- it looks like the
    decision is recorded and reachable right up until someone clicks it.
    """
    dangling = sorted(_adr_files_linked_from_log() - _adr_files_on_disk())
    assert not dangling, (
        f"adr-log.md links to ADR files that do not exist: {dangling} -- "
        "either the file was renamed (fix the link) or removed (an ADR is "
        "immutable once accepted; supersede it with a new one instead)"
    )
