# MouseDroidAGI — Rover Hardening + Roadmap Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Each workstream (WS) is an independently-shippable PR — execute in its own git worktree (`superpowers:using-git-worktrees`).

**Goal:** Bring every spec doc, skill validator, and code-health gate up to the *current* state of the application (through PR #117/#118), add real e2e + AQA/regression validation for the `.claude/commands` skills, fully purge lint/type suppressions, close the residual operator follow-ups, and activate the next roadmap features (MLflow logger + Phase 6 plan).

**Architecture:** Six independently-mergeable workstreams. WS1–WS4 are pure-hardening (docs, skill validation, code health) and parallelizable. WS5 closes residual operator gaps. WS6 activates forward features. All changes obey the repo's non-negotiable invariants: Protocol-based DI, factory-only wiring, schema-driven config (no hardcoded values), `structlog`, asyncio, `mypy --strict`, backwards-compatible defaults, reusable components.

**Tech Stack:** Python 3.10/3.11, Pydantic v2, pytest + pytest-asyncio + hypothesis, ruff 0.8.0, mypy --strict, structlog, MLflow (skinny), MkDocs-style markdown specs. Execution leverages git worktrees, subagent fan-out, TDD, and the Context7 MCP for library docs.

---

## Context — why this change is being made

PR #117 (full rover bring-up: unified dashboard + sensor-fusion summary) and PR #118 (operator Q&A + full backend telemetry) just merged. The codebase moved faster than three classes of supporting artifact:

1. **Spec drift.** `CLAUDE.md`, `AGENTS.md`, `agent.md`, and `SKILLS.md` document the system but lag the newest surfaces and contain at least one dangling promise: `src/mousedroid/skills/builtin/__init__.py:10-13` says the test `tests/unit/skills/builtin/test_skill_specs_match_docs.py` "enforces that pairing" — **that test does not exist**, so builtin specs and their `docs/openclaw_skills/*/SKILL.md` can silently diverge.
2. **Skills have no validation harness.** The `.claude/commands` skills (`robot-arm-trainer`, `sim-test`, `train-policy`) have no frontmatter linter, no referenced-path checker, and no e2e validation of what they produce. `train-policy.md:11` already points at a non-existent `configs/hanoi_3disk.yaml` — exactly the rot a validator catches.
3. **Suppression debt.** ~51 `# type: ignore` across 23 files and 32 `# noqa` across 17 files have accumulated (re-derive at execution — counts are approximate); some are load-bearing, some are stale. (NumPy itself is clean — no deprecated aliases, soft-pinned `!=2.0.0,!=2.0.1`, centralized in `common/math/numpy_ops.py`.)
4. **Open operator follow-ups.** `docs/planning/NEXT_STEPS.md` tracks F-006/F-009/F-010/F-013/F-014 and the multi-sprint Phase 6, plus a fully-written-but-unmerged MLflow plan.

**Intended outcome:** specs that match reality, skills that are linted *and* e2e-validated (with the contract pinned in AQA/regression), a zero-suppression-debt `mypy --strict` + `ruff` baseline, closed residual operator gaps, and the two next roadmap features activated — all backwards-compatible and reusable.

---

## Current-state findings (evidence)

| Area | Finding | Evidence |
|------|---------|----------|
| Specs | Root has `CLAUDE.md`, `AGENTS.md`, `agent.md` (separate lowercase file), `SKILLS.md`, `README.md`, `CHANGELOG.md`, `NEXT_STEPS.md`, `SMOKE_REPORT.md` | `ls *.md` |
| Skill validator gap | Referenced test missing | `src/mousedroid/skills/builtin/__init__.py:11`; `tests/unit/skills/builtin/` has no `test_skill_specs_match_docs.py` |
| Command-skill drift | `configs/hanoi_3disk.yaml` does not exist; real config is `config/robot_arm_training.yaml` | `.claude/commands/train-policy.md:11`, `:33` |
| Arm status | Robot-arm work is **deferred** baseline — code + tests exist but not active feature work | `docs/planning/NEXT_STEPS.md:217-220` |
| F-006 latency guard | **Already shipped** across all 3 gateways + counter | `llm_gateway/anthropic_gateway.py:360`, `gateway.py:231`, `openai_compatible.py:291`, `telemetry/metrics.py:1087` |
| F-013 deploy script | Missing | `scripts/deploy_config_jetson.sh` absent |
| F-014 boot log | No *boot-time* mock-hardware log in `start()` (note: `health_check()` already logs `mock_hardware`, so don't duplicate — add it at boot) | `orchestrator.py` |
| AQA pattern | Plain pytest functions importing the module + asserting contracts/no-hardcoded-host | `tests/regression/test_dashboard_aqa.py` |
| Test tiers | unit 323, integration 50, regression 30 (3 AQA), hardware 22, property 17, smoke 13, e2e 6, performance 5 | `find tests/*/ -name 'test_*.py'` |
| Suppressions | ~51 `type: ignore`/23 files; 32 `noqa`/17 files (re-derive at execution — treat as approximate). Hotspots: `hardware/display/expressions.py` (17 ignore), `validation/runtime.py` (10 noqa), `learning/offline_rl.py` (7 ignore) | grep counts |
| MLflow | Full 11-task plan written, unmerged | `docs/superpowers/plans/2026-06-05-mlflow-experiment-logger.md` |
| ruff config | `select` incl. `ANN, D, S, T20`; per-file ignores for tests/tools/scripts/harness/skills | `pyproject.toml:170-195` |
| mypy config | `strict = true`, `ignore_missing_imports = true`, module overrides | `pyproject.toml:197-242` |

---

## Execution methodology (agents · subagents · worktrees · skills · MCPs · TDD)

This plan is built to be executed with the full toolchain the request calls for. Apply these throughout:

- **Worktree isolation** — `superpowers:using-git-worktrees`. One worktree per workstream so WS1–WS4 can proceed in parallel without colliding. Branch names: `claude/ws1-spec-sync`, `claude/ws2-skill-validators`, etc.
- **Agent "teams"** — the user's "teams" maps to parallel agent fan-out across isolated worktrees: each independent WS runs as a small team (impl + test + reviewer subagents) concurrently via `superpowers:dispatching-parallel-agents`, synchronized only at the final integration gate. There is no separate "teams" product to invoke — it is this concurrency pattern.
- **Subagent-driven execution** — `superpowers:subagent-driven-development`: dispatch a fresh subagent per task, two-stage review (self-review → independent reviewer) between tasks. For independent tasks within a WS, fan out with `superpowers:dispatching-parallel-agents`.
- **TDD discipline** — `superpowers:test-driven-development`: every code/test task is RED (write failing test, run it, confirm the *expected* failure) → GREEN (minimal impl) → REFACTOR → commit. Never write impl before a failing test.
- **Subagent type mapping:**
  - Recon / "where is X": `Explore`.
  - Implementation: `python-pro` (or `general-purpose`).
  - Test authoring/running: `test-runner` / `testing-suite:test-engineer`.
  - Lint/type cleanup (WS4): `code-quality`.
  - Secret/credential review of `deploy_config_jetson.sh` + telemetry-token handling (WS5): `security-auditor`.
  - Docs (WS1, WS6 runbooks): `documentation-generator:technical-writer`.
  - Pre-merge review: `feature-dev:code-reviewer` + `coderabbit:code-review` + `superpowers:requesting-code-review`; respond with `superpowers:receiving-code-review`.
- **MCPs:** `Context7` for current MLflow + Anthropic-SDK docs (WS5/WS6) — *do not* rely on memory for library APIs. HuggingFace MCP for weight-repo facts when authoring the Phase 6 plan. (Linear MCP is available if the user wants issue mirroring — optional, ask first.)
- **Verification gate** — `superpowers:verification-before-completion`: before any "done"/PR claim, run the verification block for that WS and paste real output. Evidence before assertions.
- **Repo's own skills** — use the `/sim-test` command skill as the driver under test in WS3.

**Sequential reasoning per task:** state the contract being protected → write the test that fails without it → implement → prove green → confirm no invariant regressed (`ruff`, `mypy --strict`, backwards-compat).

---

## Workstream map

| WS | Title | Depends on | Parallel? | Primary agent | Lands as |
|----|-------|-----------|-----------|---------------|----------|
| WS0 | Bootstrap & plan persistence | — | first | general-purpose | (no PR — setup) |
| WS1 | Spec/doc synchronization + dangling-validator fix | WS0 | ✅ with WS2-4 | technical-writer + python-pro | PR: `docs: sync specs to PR#118 state` |
| WS2 | `.claude/commands` skill validators + AQA/regression | WS0 | ✅ | python-pro + test-engineer | PR: `test: skill-command validators + AQA` |
| WS3 | Skill e2e output validation | WS2 | after WS2 | test-engineer | PR: `test(e2e): skill-command output validation` |
| WS4 | Full lint/type/suppression purge | WS0 | ✅ | code-quality | PR: `chore: purge type:ignore/noqa debt` |
| WS5 | Operator follow-ups (F-006/09/10/13/14) | WS0 | ✅ | python-pro + security-auditor | PR: `feat(ops): close smoke-sprint follow-ups` |
| WS6 | Forward features: MLflow + Phase 6 plan | WS1–WS5 green | last | ai-engineer + technical-writer | PR: `feat(training): mlflow logger` + plan file |

---

## Global acceptance criteria (every WS must satisfy)

- [ ] `ruff check src/ tests/ tools/` and `ruff format --check src/ tests/` clean.
- [ ] `mypy --strict src/mousedroid/` clean (0 errors).
- [ ] `pytest` green with coverage ≥ 85% (`--cov-fail-under=85`), run `--import-mode=importlib`.
- [ ] **No hardcoded values** — every threshold/path/port comes from config, discovered at runtime, or a module-level named constant. New config fields are `Optional` with defaults (existing YAML loads unchanged).
- [ ] **Reusable** — shared logic lives in one function/module and is imported, not copy-pasted.
- [ ] **Backwards-compatible** — no behavior change unless explicitly gated by a default-off flag; regression test proves legacy path unchanged.
- [ ] Conventional-commit messages; CHANGELOG `## [Unreleased]` updated; relevant `docs/architecture` / `docs/runbooks` updated.

---

## WS0 — Bootstrap & plan persistence

**Files:**
- Create: `docs/superpowers/plans/2026-06-13-rover-hardening-and-roadmap.md` (canonical repo copy of this plan)

- [ ] **Step 1: Create the isolated workspace.** Use `superpowers:using-git-worktrees` to branch from the current HEAD. Create one base branch `claude/rover-hardening-2026-06-13`; each WS gets a child worktree/branch off it.

- [ ] **Step 2: Persist this plan into the repo (the "unique file").** Copy this plan's contents to `docs/superpowers/plans/2026-06-13-rover-hardening-and-roadmap.md` (the repo convention seen in git status; `2026-06-13` = today). This is the durable, version-controlled copy.

```bash
# from repo root, inside the worktree
mkdir -p docs/superpowers/plans
# author the file with the full plan body, then:
git add docs/superpowers/plans/2026-06-13-rover-hardening-and-roadmap.md
git commit -m "docs: add rover hardening + roadmap activation plan"
```

- [ ] **Step 3: Confirm baseline is green before touching anything.** Run `bash scripts/ci.sh` (or the staged commands) and record the starting state so every later "clean" claim is a delta, not an assumption.

---

## WS1 — Spec/doc synchronization + dangling-validator fix

Goal: make `CLAUDE.md`, `AGENTS.md`, `agent.md`, `SKILLS.md` reflect the current app (through PR #117/#118 + the new validation surfaces this plan adds), and resolve the dangling `test_skill_specs_match_docs.py` reference.

**Files:**
- Modify: `CLAUDE.md`, `AGENTS.md`, `agent.md`, `SKILLS.md`, `CHANGELOG.md`
- Create: `tests/unit/skills/builtin/test_skill_specs_match_docs.py`

### Task 1.1 — Resolve the dangling builtin-spec validator (closes a doc-accuracy bug)

- [ ] **Step 1: Write the failing test** that enforces the pairing the docstring already promises (builtin `SPEC` ↔ `docs/openclaw_skills/<name>/SKILL.md`).

```python
# tests/unit/skills/builtin/test_skill_specs_match_docs.py
"""Enforce the builtin SkillSpec <-> docs/openclaw_skills SKILL.md pairing.

The module docstring in ``src/mousedroid/skills/builtin/__init__.py`` promises
this test exists. It pins that every builtin spec has a publishable SKILL.md
whose front-matter name matches the spec name, so the two never drift.
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
    # The docs are plain Markdown whose H1 == the skill name (verified:
    # docs/openclaw_skills/mousedroid-navigate/SKILL.md:1 == "# mousedroid-navigate").
    # Do NOT assert YAML front-matter — these docs intentionally have none.
    assert f"# {spec.name}" in text, f"SKILL.md H1 must name the skill {spec.name!r}"


def test_no_orphan_skill_docs() -> None:
    """Every published SKILL.md dir maps to a registered builtin spec."""
    spec_names = {s.name for s in _SPECS}
    doc_dirs = {p.parent.name for p in _DOCS_ROOT.glob("*/SKILL.md")}
    assert doc_dirs == spec_names, f"doc/spec set mismatch: {doc_dirs ^ spec_names}"
```

- [ ] **Step 2: Run it; confirm GREEN.** `pytest tests/unit/skills/builtin/test_skill_specs_match_docs.py -v --import-mode=importlib`. All four docs are plain Markdown whose H1 equals the skill name (verified), so this passes today and will catch any future rename/removal of a doc or spec. It does NOT require adding YAML front-matter to the publishable docs (they intentionally have none).
- [ ] **Step 3: Commit.** `git commit -m "test(skills): add promised builtin-spec/doc pairing validator"`

### Task 1.2 — Sync the four spec docs to current state

- [ ] **Step 1: Update `CLAUDE.md`.** Append a short subsection under the existing PR-history sections documenting (a) PR #118 operator Q&A + backend telemetry, (b) the new skill-validation surface this plan adds (validators in `tools/`, AQA in `tests/regression/test_skill_commands_aqa.py`, e2e in `tests/e2e/test_skill_commands_e2e.py`), and (c) the test-tier table correction: add the **property** (17) and **performance** (5) tiers that currently exist but are absent from the PR #104 "Test surface mirror" table. Keep the established voice (contract-first, "non-negotiable").
- [ ] **Step 2: Update `SKILLS.md`.** Add a **"Skill validation"** section keyed by trigger ("validate skills", "skill drift") pointing at: the new `tools/validate_skill_commands.py`, the AQA test, the e2e test, and the builtin pairing test from Task 1.1. Document that `.claude/commands` skills are linted for frontmatter + referenced-path existence and e2e-validated for output.
- [ ] **Step 3: Update `AGENTS.md`.** Add a step to the "adding/maintaining a skill" contract: any new `.claude/commands/*.md` must (1) carry a non-empty `description`, (2) reference only paths that exist, (3) get an e2e output check if it produces an artifact. Cross-link the validator.
- [ ] **Step 4: Update `agent.md`.** Add the skill-validation invariant to the Key Invariants list (mirror the `assert`-under-`-O` discipline already there).
- [ ] **Step 5: CHANGELOG.** Add `## [Unreleased]` entries for the doc sync + new validator.
- [ ] **Step 6: Verify links.** Grep each edited doc for referenced paths and confirm they exist (reuse the WS2 validator logic if landed). Commit: `git commit -m "docs: sync CLAUDE/AGENTS/agent/SKILLS to PR#118 + validation surface"`

---

## WS2 — `.claude/commands` skill validators + AQA/regression

Goal: a reusable validator that lints every command skill (frontmatter + referenced-path existence + no-hardcoded-values) plus an AQA regression test that pins the contract. This is the heart of "update skill validators … include in aqa/regression." (Note: no `.claude/commands` validator exists today — this *creates* the first; WS1.1 separately resolves the one builtin validator that is referenced-but-missing.)

**Files:**
- Create: `tools/validate_skill_commands.py` (reusable library + CLI)
- Create: `tests/regression/test_skill_commands_aqa.py`
- Modify: `.claude/commands/train-policy.md` (fix the `configs/hanoi_3disk.yaml` drift the validator catches)
- Modify: `scripts/ci.sh` (wire the validator into the lint stage)

### Task 2.1 — Reusable command-skill validator

- [ ] **Step 1: Write the failing test** for the validator library (TDD on the tool itself).

```python
# tests/unit/tools/test_validate_skill_commands.py
"""Unit tests for the reusable command-skill validator."""
from __future__ import annotations

from pathlib import Path

from tools.validate_skill_commands import (
    SkillCommandIssue,
    referenced_repo_paths,
    validate_command_skill,
)


def _write(p: Path, body: str) -> Path:
    p.write_text(body, encoding="utf-8")
    return p


def test_valid_skill_has_no_issues(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / "src").mkdir()
    real = _write(repo / "src" / "thing.py", "x = 1\n")
    skill = _write(
        repo / "ok.md",
        "---\ndescription: Does a thing\n---\n\nUses `src/thing.py`.\n",
    )
    assert validate_command_skill(skill, repo_root=repo) == []
    assert real.exists()


def test_missing_description_is_flagged(tmp_path: Path) -> None:
    skill = _write(tmp_path / "bad.md", "---\nname: x\n---\nbody\n")
    issues = validate_command_skill(skill, repo_root=tmp_path)
    assert any(i.code == "missing-description" for i in issues)


def test_missing_referenced_path_is_flagged(tmp_path: Path) -> None:
    skill = _write(
        tmp_path / "bad.md",
        "---\ndescription: d\n---\nUses `config/nope.yaml`.\n",
    )
    issues = validate_command_skill(skill, repo_root=tmp_path)
    assert any(i.code == "missing-path" and "config/nope.yaml" in i.detail for i in issues)


def test_glob_and_format_tokens_are_ignored(tmp_path: Path) -> None:
    # Pattern paths are NOT real files and must not be flagged.
    skill = _write(
        tmp_path / "ok.md",
        "---\ndescription: d\n---\nOutput `weights/arm/{task}_{stage}_final.pt`.\n",
    )
    assert referenced_repo_paths(skill.read_text(encoding="utf-8")) == []
```

- [ ] **Step 2: Run; confirm import failure (RED).** `pytest tests/unit/tools/test_validate_skill_commands.py -v --import-mode=importlib` → fails (module not found).

- [ ] **Step 3: Implement the validator** — reusable, config-free, no hardcoded skill names (discovers files), no hardcoded paths (extracts from body).

```python
# tools/validate_skill_commands.py
"""Validate ``.claude/commands/*.md`` skill files.

Reusable library + CLI. Checks, per skill file:
  * YAML front-matter parses and carries a non-empty ``description``.
  * Every backtick-wrapped repo path it references actually exists.
  * It contains no hardcoded host/IP (skills must stay environment-agnostic).

Paths are *discovered* from the body, never enumerated here, so the tool keeps
working as skills evolve. Format/glob tokens ({}, *, $, <>) are excluded so
illustrative patterns like ``weights/arm/{task}_final.pt`` are not false flags.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# Capture any backtick-delimited token, then decide IN CODE whether it looks
# like a repo-relative file reference. Filtering in Python (not the regex) keeps
# the format/glob exclusion reachable + testable and catches partially-braced
# tokens a stricter regex would silently miss.
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_PATH_EXT_RE = re.compile(r".+\.(?:py|ya?ml|md|sh|pt|onnx|json|urdf|usd)$")
_FORBIDDEN_IN_PATH = set("{}*$<> ")
_HARDCODED_HOST_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


@dataclass(frozen=True)
class SkillCommandIssue:
    """A single validation problem found in a skill file."""

    path: Path
    code: str
    detail: str


def referenced_repo_paths(text: str) -> list[str]:
    """Return backtick-wrapped repo-relative file references, excluding patterns.

    A token counts when it ends in a known source/config extension, contains a
    ``/`` (repo-relative, not a bare word), and carries no format/glob
    metacharacters — so illustrative patterns like ``weights/arm/{task}.pt`` are
    correctly skipped while real refs like ``config/foo.yaml`` are validated.
    """
    out: list[str] = []
    for m in _BACKTICK_RE.finditer(text):
        token = m.group(1).strip()
        if "/" not in token:
            continue
        if not _PATH_EXT_RE.match(token):
            continue
        if any(c in _FORBIDDEN_IN_PATH for c in token):
            continue
        out.append(token)
    return out


def _split_front_matter(text: str) -> tuple[dict[str, object] | None, str]:
    if not text.startswith("---"):
        return None, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None, text
    return (meta if isinstance(meta, dict) else None), parts[2]


def validate_command_skill(path: Path, *, repo_root: Path) -> list[SkillCommandIssue]:
    """Validate one skill file; return a list of issues (empty == valid)."""
    text = path.read_text(encoding="utf-8")
    issues: list[SkillCommandIssue] = []

    meta, body = _split_front_matter(text)
    if meta is None:
        issues.append(SkillCommandIssue(path, "bad-front-matter", "missing/invalid YAML front-matter"))
        meta, body = {}, text

    description = str(meta.get("description", "")).strip()
    if not description:
        issues.append(SkillCommandIssue(path, "missing-description", "front-matter 'description' is empty"))

    for ref in referenced_repo_paths(body):
        if not (repo_root / ref).exists():
            issues.append(SkillCommandIssue(path, "missing-path", ref))

    for host in _HARDCODED_HOST_RE.findall(body):
        issues.append(SkillCommandIssue(path, "hardcoded-host", host))

    return issues


def validate_all(commands_dir: Path, *, repo_root: Path) -> list[SkillCommandIssue]:
    """Validate every ``*.md`` skill in ``commands_dir``."""
    issues: list[SkillCommandIssue] = []
    for md in sorted(commands_dir.glob("*.md")):
        issues.extend(validate_command_skill(md, repo_root=repo_root))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--commands-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    repo_root: Path = args.repo_root.resolve()
    commands_dir: Path = (args.commands_dir or repo_root / ".claude" / "commands").resolve()

    issues = validate_all(commands_dir, repo_root=repo_root)
    # print in tools/ is already exempt from T20 via pyproject per-file-ignores
    # (pyproject.toml: "tools/**/*.py" = [..., "T20"]) — no inline noqa needed.
    for i in issues:
        print(f"{i.path}: [{i.code}] {i.detail}")
    if issues:
        print(f"FAIL: {len(issues)} skill-command issue(s)")
        return 1
    print("OK: all skill commands valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run unit tests (GREEN).** `pytest tests/unit/tools/test_validate_skill_commands.py -v --import-mode=importlib` → pass.
- [ ] **Step 5: Run the CLI against the real repo; expect it to FAIL on the known drift.** `python tools/validate_skill_commands.py` → reports `train-policy.md: [missing-path] configs/hanoi_3disk.yaml`.
- [ ] **Step 6: Fix the drift.** Edit `.claude/commands/train-policy.md:11` default arg from `configs/hanoi_3disk.yaml` to a real config (`config/robot_arm_training.yaml`, consistent with line 33). Re-run CLI → `OK`.
- [ ] **Step 7: Commit.** `git commit -m "feat(tools): reusable .claude/commands skill validator + fix train-policy drift"`

### Task 2.2 — AQA/regression pin

- [ ] **Step 1: Write the AQA test** (mirrors `tests/regression/test_dashboard_aqa.py` style — import + assert contracts).

```python
# tests/regression/test_skill_commands_aqa.py
"""AQA: .claude/commands skill-file hygiene.

Locks the contract a careless edit could break: every command skill carries a
non-empty description, references only paths that exist, and bakes in no host/IP.
Reuses the shared validator so the rule lives in exactly one place.
"""
from __future__ import annotations

from pathlib import Path

from tools.validate_skill_commands import validate_all

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMANDS = _REPO_ROOT / ".claude" / "commands"


def test_all_command_skills_are_valid() -> None:
    issues = validate_all(_COMMANDS, repo_root=_REPO_ROOT)
    assert issues == [], "skill-command issues:\n" + "\n".join(
        f"  {i.path.name}: [{i.code}] {i.detail}" for i in issues
    )


def test_commands_dir_is_non_empty() -> None:
    # Guards against the validator silently passing on an empty/renamed dir.
    assert list(_COMMANDS.glob("*.md")), "no .claude/commands/*.md skills found"
```

- [ ] **Step 2: Run (GREEN, since Task 2.1 fixed the drift).** `pytest tests/regression/test_skill_commands_aqa.py -v --import-mode=importlib`.
- [ ] **Step 3: Wire into CI.** The actual PR gate is the WS2 AQA pytest (it runs in the regression/test stage of `.github/workflows/ci.yml`). For a fast standalone signal, also add `python tools/validate_skill_commands.py` to `scripts/ci.sh` **and** extend that script's `ruff check` / `ruff format --check` invocations to include `tools/` — `ci.sh` currently lints `src/ tests/` only (`ci.sh:32,38`), so the new tool file is otherwise ungated locally. (Don't overstate this as a CI-lint-stage gate; the AQA test is what blocks the PR.)
- [ ] **Step 4: Commit.** `git commit -m "test(regression): AQA pin for skill-command hygiene + CI wire"`

---

## WS3 — Skill e2e output validation

Goal: "include e2e validations on output of skills." For `.claude/commands`, the *output* is the artifact/result the documented workflow produces. Validate the executable ones, dep-gated so CI hosts without `[arm]` extras skip cleanly (arm is the deferred baseline — `docs/planning/NEXT_STEPS.md:217`).

**Files:**
- Create: `tests/e2e/test_skill_commands_e2e.py`

### Task 3.1 — e2e for `sim-test` (its output is a passing arm test run)

- [ ] **Step 1: Write the e2e test** — invoke the exact command `sim-test.md` documents and assert it produces a successful run, gated on `mujoco`.

```python
# tests/e2e/test_skill_commands_e2e.py
"""E2E validation of .claude/commands skill *outputs*.

Each test runs the workflow a skill documents and asserts on what it produces.
Arm-dependent skills are gated on the optional [arm] extras so CI hosts without
MuJoCo skip cleanly (robot-arm is the deferred baseline per NEXT_STEPS.md).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.e2e
def test_sim_test_skill_runs_arm_suite() -> None:
    pytest.importorskip("mujoco")  # arm extras absent -> skip, not fail
    # The command documented in .claude/commands/sim-test.md (smallest scope).
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/arm/", "-k", "env",
         "-q", "--import-mode=importlib", "--no-cov"],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"sim-test 'env' scope failed:\n{result.stdout}\n{result.stderr}"
```

- [ ] **Step 2: Run.** On a host without MuJoCo → SKIPPED (correct). On an `[arm]` host → PASS. `pytest tests/e2e/test_skill_commands_e2e.py -v --import-mode=importlib`.
- [ ] **Step 3: Commit.** `git commit -m "test(e2e): validate sim-test skill output"`

### Task 3.2 — e2e for `train-policy` (its output is a checkpoint artifact)

This is a **smoke** e2e: it proves the documented workflow wiring *produces a checkpoint*, NOT that training converges. Real RL-to-convergence is too slow and non-deterministic for a test tier (and `scripts/ci.sh` runs `tests/e2e/` unconditionally, so it would execute on every `[arm]` host).

- [ ] **Step 1: Write the smoke e2e** — load the real arm YAML via the repo's config loader (NOT bare `Settings()`, which has no YAML source and leaves `cfg.arm*` = `None`), build env + SAC+HER agent via the arm factory, run a few `update_step`s, `save()` to `tmp_path`, and assert *a* checkpoint file appears (discover it, don't hardcode a name — see Step 4).

```python
@pytest.mark.slow  # registered marker; e2e selection is by tests/e2e/ directory
def test_train_policy_skill_emits_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("stable_baselines3")
    from mousedroid.config.loader import load_runtime_settings  # repo's YAML loader

    cfg = load_runtime_settings(["config/robot_arm_training.yaml"])
    assert cfg.arm is not None, "arm overlay must load from YAML, not env"
    # Build env + agent via the real arm factory (do not reimplement a loop).
    # env = build_arm_env(cfg); agent = build_sac_her_agent(cfg, env)
    for _ in range(5):        # test-local smoke budget — a handful of steps
        agent.update_step()   # exact API per src/mousedroid/arm/control/sac_agent.py
    agent.save(tmp_path)
    assert any(p.is_file() for p in tmp_path.rglob("*")), "save() wrote no checkpoint"
```

> Executor notes: (1) confirm the real loader symbol (`load_runtime_settings` / `resolve_runtime_config_paths`) and the real factory builders the arm module exposes — wire to what exists; do not invent. (2) The 5-step loop is a **test-local smoke constant** (idiomatic in a smoke test), *not* a production config value — so NO new schema field is required and the no-hardcoded-config rule is not violated. (3) Discover the artifact via `rglob` rather than asserting `sac_her_checkpoint` vs `.pt`; that doc drift is fixed in Step 4.

- [ ] **Step 2: Run dep-gated; confirm SKIP on a no-MuJoCo host, PASS on an `[arm]` host.**
- [ ] **Step 3: Keep heavy training out of the default CI run** — add a `-m "not slow"` deselect to the e2e CI invocation (or place this test under `tests/hardware/`), so a SAC smoke doesn't run on every CI pass.
- [ ] **Step 4: Fix the doc drift the e2e exposes.** `train-policy.md:37` documents `weights/arm/{task}_{stage}_final.pt`, but `sac_agent.save()` writes a Stable-Baselines3 checkpoint with a different name/extension. Reconcile the doc to the real artifact (executor confirms the actual filename from `sac_agent.py`).
- [ ] **Step 5: Commit.** `git commit -m "test(e2e): train-policy checkpoint smoke + fix output-path doc drift"`

### Task 3.3 — e2e smoke for `robot-arm-trainer` (full-cycle milestone 1 smoke)

- [ ] **Step 1:** Add a dep-gated (`pytest.importorskip("mujoco")`) test asserting the milestone-1 deliverable the skill names (MuJoCo scene + Gymnasium wrapper `reset()`/`step()` produce valid observations) by importing the arm env and exercising one reset/step. Reuse existing arm test fixtures. No `@pytest.mark.e2e` (unregistered marker — rely on the `tests/e2e/` directory).
- [ ] **Step 2/3:** Run (skip/pass), commit `test(e2e): robot-arm-trainer milestone-1 smoke`.

### Task 3.4 — e2e for a LIVE builtin skill output (added per peer review)

The three `.claude/commands` skills all target the *deferred* arm subsystem, so Tasks 3.1–3.3 SKIP on every non-`[arm]` CI host — leaving "e2e on skill output" unproven in normal CI. Add ONE output-e2e for a *live* builtin OpenClaw skill so the ask actually executes on any host. This **augments** (does not replace) the `.claude/commands` surface choice.

- [ ] **Step 1: Write the e2e** — drive the read-only `mousedroid-sensor-report` skill through the `SkillDelegator` and assert the result validates against its `schema_out`.

```python
# tests/e2e/test_builtin_skill_output_e2e.py
"""E2E: a live builtin skill produces schema_out-conformant output via the delegator."""
from __future__ import annotations

import pytest

from mousedroid.skills.builtin import SENSOR_REPORT_SPEC


@pytest.mark.asyncio
async def test_sensor_report_output_conforms_to_schema_out() -> None:
    spec = SENSOR_REPORT_SPEC
    assert spec.schema_in is not None and spec.schema_out is not None
    payload = {"include_lidar": True, "include_imu": True, "include_battery": True}
    spec.schema_in.model_validate(payload)  # input contract holds
    # Build a delegator + stub sub-agent (reuse tests/unit/skills/test_delegator.py
    # fixtures) that returns the skill's real output dict, then:
    # result = await delegator.delegate(spec.name, payload)
    # assert result.status == "ok"
    # assert spec.schema_out.model_validate(result.output) is not None
```

> Executor: reuse the delegator + stub-sub-agent fixtures already in `tests/unit/skills/test_delegator.py`. This proves the *output* contract end-to-end with no arm deps, so it runs in every CI pass.

- [ ] **Step 2/3:** Run (no dep-gate needed); commit `test(e2e): builtin sensor-report output schema validation`.

---

## WS4 — Full lint/type/suppression purge

Goal (user-selected "Full suppression purge"): remove every `# type: ignore` and `# noqa` in `src/` that is **stale or unjustified**. For the genuinely-required ones (untyped third-party boundaries like `luma.oled`/`tensorrt`, Pydantic field-name shadows `A00x`), KEEP but justify — relocate to `[[tool.mypy.overrides]]` / `per-file-ignores` or add a one-line reason comment. **Target = zero stale/unjustified suppressions, not literal zero** — forcing inline ignores to absolute zero can *reduce* type safety by replacing a precise `[import-untyped]` with a whole-module override. NumPy is already clean (audited: no deprecated aliases found) — verify and lock it.

**Pattern (applied per file, smallest-diff TDD):**
1. Remove one suppression.
2. Run the narrow gate it suppressed: `mypy --strict src/mousedroid/<module>.py` or `ruff check src/mousedroid/<module>.py`.
3. If now-clean → keep removed. If a real error surfaces → fix it properly (annotate, narrow a type, add a stub to `[dev]`, or add a scoped `[[tool.mypy.overrides]]` module entry with a comment) rather than re-adding the inline ignore.
4. Re-run full `mypy --strict src/mousedroid/` + `ruff check src/ tests/` to confirm no regression.
5. Commit per logical group of files.

**Representative example (the pattern, not every file):**

```python
# BEFORE  src/mousedroid/learning/offline_rl.py
loss = critic(obs, act)  # type: ignore[no-any-return]

# AFTER — annotate the boundary so the ignore is unnecessary
loss: torch.Tensor = critic(obs, act)
```

```toml
# When an ignore is load-bearing (untyped 3rd-party), justify it centrally
# in pyproject.toml instead of inline:
[[tool.mypy.overrides]]
# luma.oled ships no stubs; expressions.py drives the OLED face panel.
module = ["mousedroid.hardware.display.expressions"]
disallow_untyped_calls = false
```

**Files (grouped by suppression density — full list discovered via grep, not enumerated here):**
- `# type: ignore` hotspots: `hardware/display/expressions.py` (17), `learning/offline_rl.py` (7), `telemetry/{server,serialization,auth}.py`, `comms/wifi_driver.py`, `efficiency/tensorrt.py`, +14 more files.
- `# noqa` hotspots: `validation/runtime.py` (10), `comms/wifi_driver.py` (4), `config/schema.py` (3), `health/watchdog.py` (2), +13 more files.

### Tasks
- [ ] **Task 4.1:** Audit. Generate the live inventory: `python -m ruff check src/ tests/`, `python -m mypy --strict src/mousedroid/`, plus the grep counts. Record the baseline (expected 0 ruff/mypy *errors*, but ~51 `type: ignore` + 32 `noqa` *suppressions* — re-derive exactly, don't trust these headline numbers). Dispatch `code-quality` subagent.
- [ ] **Task 4.2:** Purge `# type: ignore` group-by-group following the pattern above. After each group: full `mypy --strict` clean. Commit per group.
- [ ] **Task 4.3:** Purge `# noqa` group-by-group; for legitimately-needed ones (e.g. CLI `T201` prints in `tools/`), confirm they're covered by per-file-ignores in `pyproject.toml` and drop the inline `noqa`. Commit per group.
- [ ] **Task 4.4:** NumPy lock — add/confirm a regression test asserting no deprecated aliases reappear.

```python
# tests/regression/test_numpy_hygiene.py
"""Regression: no deprecated NumPy aliases creep back into src/."""
from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "mousedroid"
_BANNED = re.compile(r"\bnp\.(float|int|bool|object|str|NaN|complex)\b")


def test_no_deprecated_numpy_aliases() -> None:
    offenders = [
        f"{p}:{i}" for p in _SRC.rglob("*.py")
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if _BANNED.search(line)
    ]
    assert offenders == [], "deprecated numpy aliases:\n" + "\n".join(offenders)
```

- [ ] **Task 4.5:** Add a suppression-budget regression test that pins the *new* (lower) counts so debt can't silently grow again.

```python
# tests/regression/test_suppression_budget.py
"""Regression: cap inline type:ignore / noqa debt in src/.

Update the budgets DOWN as the purge lands; never up without justification.
"""
from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "mousedroid"
# Budgets = the MEASURED post-purge residual of load-bearing suppressions
# (untyped 3rd-party boundaries, Pydantic A00x field shadows). Expected
# NON-ZERO. The executor REPLACES these with the real counts after Tasks
# 4.2/4.3 and may only ratchet them DOWN, never up without a documented reason.
_MAX_TYPE_IGNORE = -1  # REPLACE with measured residual (e.g. ~8) after Task 4.2
_MAX_NOQA = -1         # REPLACE with measured residual after Task 4.3


def _count(token: str) -> int:
    return sum(
        line.count(token)
        for p in _SRC.rglob("*.py")
        for line in p.read_text(encoding="utf-8").splitlines()
    )


def test_type_ignore_within_budget() -> None:
    assert _count("type: ignore") <= _MAX_TYPE_IGNORE


def test_noqa_within_budget() -> None:
    assert _count("noqa") <= _MAX_NOQA
```

- [ ] **Task 4.6:** Full-suite verification + commit `chore(quality): purge type:ignore/noqa debt; lock numpy + suppression budgets`.

---

## WS5 — Operator follow-ups (F-006 / F-009 / F-010 / F-013 / F-014)

> **EXECUTION DECISION (2026-06-13): ADOPT & RECONCILE — do NOT reimplement.** This work already exists, unmerged, on `claude/ops-hardening-pr-a2` (F-006 n_gpu_layers env override, F-009 `test_build_tensorrt_compiler`, F-014 compose `env_file` + `loader.py`), plus `claude/f006-remote-llm` (remote-LLM overlay + injection-filter tests) and `claude/f006-verify` (`tools/llm_latency_probe.py` + SMOKE_REPORT addendum). WS5 = **review → rebase onto current HEAD → resolve conflicts → verify → merge** those branches, not greenfield coding. Use the task list below only as the acceptance checklist to confirm each branch delivers it; fill any genuine gap.

Goal: close the residual smoke-sprint gaps. **Verify current state first** — PR #115 already shipped the latency guard + counter, and the compose mock-hardware handling was partly refactored. Implement only what's actually missing.

**Files:**
- Modify: `config/jetson_production.yaml` (F-006 config)
- Create: `scripts/deploy_config_jetson.sh` (F-013)
- Modify: `docker-compose.jetson.yml` (F-014 — verify, then close residual)
- Modify: `src/mousedroid/orchestrator/orchestrator.py` (F-014 boot log)
- Modify: `src/mousedroid/efficiency/tensorrt.py` (F-009 INFO log)
- Modify: `CHANGELOG.md` (F-010 doc), `docs/runbooks/`
- Tests: `tests/regression/test_llm_latency_guard_all_backends.py`, `tests/unit/orchestrator/test_mock_hardware_boot_log.py`, `tests/unit/efficiency/test_tensorrt_backend_log.py`

### Task 5.1 — F-006 (config + cross-backend regression; guard already exists)
- [ ] **Step 1:** Set `llm.n_gpu_layers: -1` in `config/jetson_production.yaml`. (No code — the guard already fires.)
- [ ] **Step 2: Write a regression test** asserting *every* backend emits a budget signal when `elapsed > latency_target_ms`, so a future refactor can't drop one path. Cover `anthropic_gateway` (`anthropic_gateway_slow` + `inc_llm_latency_budget_exceeded`), `gateway` (llama_cpp), `openai_compatible`. Use the existing test fakes; monkeypatch the clock to force `elapsed_ms` over budget.
- [ ] **Step 3:** Confirm green; commit `feat(ops): F-006 enable CUDA offload + pin cross-backend latency-guard regression`.

### Task 5.2 — F-013 deploy-config script (genuinely missing)
- [ ] **Step 1:** Author `scripts/deploy_config_jetson.sh` — `scp config/*.yaml jetson:/etc/mousedroid/` + checksum verify + orchestrator restart. **No hardcoded host/token** — read `JETSON_HTTP`/host + paths from env with documented defaults (mirror `tools/dashboard_proxy.py`'s env-or-arg pattern). Use `set -euo pipefail`. Have `security-auditor` review it (no secret echo).
- [ ] **Step 2:** Add a shellcheck/`bash -n` smoke + a unit test asserting the script exists, is executable, and references no literal IP. Document in `docs/runbooks/jetson-rover-smoke.md`.
- [ ] **Step 3:** Commit `feat(ops): F-013 deploy_config_jetson.sh config-sync path`.

### Task 5.3 — F-014 compose + boot log
- [ ] **Step 1: Verify** the current `docker-compose.jetson.yml` mock-hardware handling (the file already has comments about honoring `docker.env`). Only flip the default to `:-false` / add the telemetry-token env block if still absent.
- [ ] **Step 2:** Add `self._log.info("mock_hardware_resolved", value=cfg.mock_hardware)` at orchestrator boot (confirmed missing). TDD: unit test capturing the structlog event via the repo's existing log-capture fixture.
- [ ] **Step 3:** Commit `feat(ops): F-014 mock_hardware boot log + compose default verify`.

### Task 5.4 — F-009 TensorRT backend visibility
- [ ] **Step 1:** Promote `tensorrt_compiler_built` to INFO with a `backend: real|mock` field in `efficiency/tensorrt.py`. TDD: unit test asserting the field is present for both branches.
- [ ] **Step 2:** Commit `feat(observability): F-009 TensorRT real/mock backend log`.

### Task 5.5 — F-010 VLM-mock documentation
- [ ] **Step 1:** Add a CHANGELOG `## [Unreleased]` note that the Tier C2.3 replanner evaluates wiring against `MockVLMProgress`, not a real VLM, until Tier C3. Docs-only.
- [ ] **Step 2:** Commit `docs: F-010 document VLM-progress mock status`.

---

## WS6 — Forward features: MLflow logger + Phase 6 plan

> **EXECUTION DECISION (2026-06-13): ADOPT & RECONCILE Task 6.1 — do NOT reimplement.** The MLflow logger is fully built (unmerged) on `feat/mlflow-experiment-logger` (24 commits: `training/observability/` modules, `test_mlflow_logger.py` 448 lines + `test_noop_logger.py`, C4/README/CHANGELOG/NEXT_STEPS docs, regression tests). Task 6.1 = **review → rebase onto current HEAD → resolve conflicts → verify → merge** that branch. Task 6.2 (author the Phase 6 plan) remains greenfield.

Goal: activate the next two roadmap items. MLflow has a complete written plan; Phase 6 needs its plan authored (the user asked to "scope Phase 6").

### Task 6.1 — Execute the MLflow experiment-logger plan
- [ ] **Step 1:** Execute `docs/superpowers/plans/2026-06-05-mlflow-experiment-logger.md` task-by-task via `superpowers:subagent-driven-development` in its own worktree. Do **not** duplicate it here — that plan already specifies the protocol-DI abstraction (`ExperimentLoggerProtocol`, `NoOpExperimentLogger` default-off, `MlflowExperimentLogger`), `build_experiment_logger()` factory, schema config (`ObservabilityConfig`), and 8 test modules.
- [ ] **Step 2:** Use the `Context7` MCP to confirm current `mlflow-skinny` API before implementing the logger (don't code MLflow from memory).
- [ ] **Step 3:** Honor global acceptance criteria — default OFF, byte-identical legacy behavior, `mypy --strict` clean, ≥85% coverage. Land as its own PR.

### Task 6.2 — Author the Phase 6 plan (scope only — implementation is 3–4 sprints)
- [ ] **Step 1:** Using `superpowers:writing-plans`, author `docs/superpowers/plans/2026-06-13-phase6-on-device-incremental-learning.md` from the scope in `docs/planning/NEXT_STEPS.md:75-102`: online update paths in `learning/ewc.py` + `learning/progressive_net.py` (SHA-256-gated per ADR-010, separate weight slot), replay-triggered update step via `harness/replay_buffer.py`, safety-regression gate emitting a new `mousedroid_on_device_learning_reverted_total{reason}` counter, and A/B vs cloud weights.
- [ ] **Step 2:** Reference HF weight repos (`ianshank/mousedroid-weights`) via the HuggingFace MCP for accurate artifact paths. Keep it a *plan*, not an implementation, in this PR.
- [ ] **Step 3:** Commit `docs(plan): author Phase 6 on-device incremental-learning plan`.

---

## Roadmap catalog (beyond this plan — prioritized, each its own future plan)

These were surfaced in recon and are intentionally **not** implemented here (each is its own plan/PR). Listed so the next session has the map:

| Priority | Item | Source | Type |
|----------|------|--------|------|
| HIGH | Phase 6 on-device incremental learning (impl) | `NEXT_STEPS.md:75` + plan authored in Task 6.2 | Code, 3–4 sprints |
| MED | Isaac Lab training backend (6 phases) | `docs/planning/ISAAC_LAB_ROVER_RESEARCH.md`, ADR-009 | Code, research |
| MED | Tier D unified dashboard + Grafana (D1–D6) | `docs/superpowers/plans/2026-05-15-full-stack-...md` | Code |
| MED | Cloud training Phase 2/3 (Vertex AI, GKE sim) | `NEXT_STEPS.md:559-575` | Code |
| LOW | L4T container docs (M6); ROS 2 bridge; Whisper STT | `NEXT_STEPS.md:578-588`, `PLANNING.md` | Docs/Code |

---

## Verification (end-to-end)

Run from repo root after each WS, and a full pass before the final integration:

```bash
# 1. Lint + format (Stage 1)
ruff check src/ tests/ tools/
ruff format --check src/ tests/

# 2. Skill-command validator (WS2)
python tools/validate_skill_commands.py

# 3. Type check (Stage 2)
mypy --strict src/mousedroid/

# 4. Targeted new tests (WS1-WS5)
pytest tests/unit/skills/builtin/test_skill_specs_match_docs.py \
       tests/unit/tools/test_validate_skill_commands.py \
       tests/regression/test_skill_commands_aqa.py \
       tests/regression/test_numpy_hygiene.py \
       tests/regression/test_suppression_budget.py \
       tests/regression/test_llm_latency_guard_all_backends.py \
       -v --import-mode=importlib

# 5. e2e skills — arm-gated tests skip without [arm]; the builtin one always runs
pytest tests/e2e/test_skill_commands_e2e.py tests/e2e/test_builtin_skill_output_e2e.py \
       -v --import-mode=importlib

# 6. Full suite + coverage gate (Stage 3)
pytest --import-mode=importlib --cov=src/mousedroid --cov-fail-under=85

# 7. Full local CI mirror
bash scripts/ci.sh
```

**Manual / MCP verification:**
- WS5 F-006: on an `[arm]`-free dev box, run `python scripts/translate_mission.py "turn left slowly"` (dry-run, no motors) and confirm the latency-budget log path; on the rover, confirm `n_gpu_layers: -1` brings the round-trip under budget.
- WS6 MLflow: run a tiny training with the logger enabled, then `mlflow ui --backend-store-uri ./mlruns` and confirm parent/child runs render.

**Success = every command above exits 0, coverage ≥ 85%, and the suppression-budget test pins the new lower counts.**

---

## Self-review checklist (run after drafting, before execution)

- [ ] **Spec coverage:** every user ask maps to a WS — next-steps roadmap (WS6 + catalog), docs sync of CLAUDE/AGENTS/agent/SKILLS (WS1), skill validators (WS2), e2e on skill output (WS3), AQA/regression inclusion (WS2/WS4), ruff/lint/mypy/numpy (WS4), backwards-compat + no-hardcoded + reusable (global criteria), unique plan file (WS0 Step 2).
- [ ] **No placeholders:** new-code tasks (WS1.1, WS2, WS3.1, WS4.4/4.5) carry complete code; pattern-based tasks (WS4 purge, WS6 MLflow) reference exact existing plans/files rather than vague "handle it."
- [ ] **Type consistency:** `validate_command_skill` / `referenced_repo_paths` / `validate_all` / `SkillCommandIssue` names match across `tools/validate_skill_commands.py`, its unit test, and the AQA test.
- [ ] **Verify-before-build honored:** F-006 downgraded to config+test after confirming the guard exists; F-014 framed as verify-then-close.

---

## Peer-review revisions (applied 2026-06-13)

Two independent adversarial reviewers + direct file verification found and this plan now fixes:

- **[blocker] WS1.1** asserted YAML front-matter (`name:`) that no `SKILL.md` has → switched to the H1 assertion (`# <name>`), which matches all four docs today. No doc-format change needed.
- **[blocker] WS3.2** used bare `Settings()` (no YAML source → `cfg.arm = None`) and asserted a `.pt` artifact the code never writes → now loads the arm YAML via the repo loader, runs a *smoke* (few steps), discovers the real checkpoint via `rglob`, and adds a step to fix the `train-policy.md` output-path drift.
- **[major] WS3.2** ran real SAC training as a test tier → reframed as a wiring smoke + CI `not slow` deselect; removed the smuggled-in schema-field change (step budget is a test-local constant).
- **[major] WS4** budget literals `0` contradicted the "keep load-bearing" prose and fought WS2's added `# noqa` → budgets now `-1` placeholders the executor sets to the *measured non-zero residual*; target reframed to "zero stale/unjustified," not literal zero; removed the redundant `# noqa: T201` from the validator.
- **[minor]** validator regex had unreachable dead-code filter → redesigned to capture-then-filter (now reachable + catches partially-braced paths); `@pytest.mark.e2e` (unregistered) dropped in favor of directory-based selection; CI-wiring wording corrected (AQA pytest is the gate; extend `ruff` to `tools/`); suppression counts corrected to ~51/23 (approximate, re-derive); F-014 evidence reworded; "teams" explicitly mapped; **added Task 3.4** (a live builtin-skill output e2e) so "e2e on skill output" actually runs in CI rather than always skipping on the deferred arm surface.

Reviewer-confirmed-correct (no change): path-depth math (`parents[4]`/`parents[2]`), `SkillSpec.name`, the `train-policy.md → configs/hanoi_3disk.yaml` drift, AQA-in-`tests/regression/` placement, NumPy cleanliness.

## Execution handoff

Plan saved (WS0 persists it to `docs/superpowers/plans/2026-06-13-rover-hardening-and-roadmap.md`). Recommended execution:

1. **Subagent-Driven (recommended)** — `superpowers:subagent-driven-development`: WS0 first; then WS1/WS2/WS4/WS5 in parallel worktrees (independent), WS3 after WS2, WS6 last after the suite is green. Fresh subagent per task + two-stage review.
2. **Inline Execution** — `superpowers:executing-plans`: batch per-WS with checkpoints.

The plan file is the only artifact created during planning; all other changes happen at execution time after approval.
