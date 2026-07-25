# Peer Review — `mouse-droid-claude-workforce` (original draft)

- **reviewed_artifact**: user-supplied OpenSpec change bundle `mouse-droid-claude-workforce`
  (draft; never merged, never previously materialized in this tree)
- **basis_commit**: `c9ac439` (clean working tree)
- **review_date**: 2026-07-24
- **method**: every factual claim checked against the working tree (evidence cited by
  symbol/path); Claude Code platform behaviour checked against the current official docs
  (sub-agents, hooks, skills, settings, MCP, memory pages)
- **confidence vocabulary**: `[Certain]` — verified against the tree or platform docs;
  `[Likely]` — strong inference (house style per
  `docs/superpowers/plans/2026-07-03-validation-first-rev-b.md`)
- **outcome**: rewrite required and delivered as rev. B in this directory
  (`proposal.md`, `design.md`, `tasks.md`, `specs/`). The draft's gap diagnosis is largely
  correct; its factual base, two of its mechanisms, and several of its enforcement claims
  are not.

## Verdict table

All verdicts `[Certain]` unless tagged otherwise.

| # | Original claim / proposal | Verdict | Evidence |
|---|---|---|---|
| 1 | "One (`sim-test`) already carries the correct pattern — `status: frozen` + explicit `unfreeze:` condition — the other two do not" | **REFUTED — inverted** | All three `.claude/skills/*/SKILL.md` carry `status: frozen` + a byte-identical `unfreeze:` string, frozen together in one commit (`a12c95b`, PR #151) and pinned by `tests/regression/test_next_steps_reconciled.py::test_frozen_skills_carry_status_frontmatter`. The draft's skill-refresh task is a no-op or a test-breaker |
| 2 | Add a `legacy` lifecycle status and a `superseded_by:` key; mark `robot-arm-trainer` superseded by `train-policy` | **REFUTED — fails an existing gate** | `tools/validate_skill_commands.py::_ALLOWED_SKILL_STATUSES` is exactly `{"active", "frozen", "deferred"}`; an out-of-set status is an `invalid-status` finding → CLI exit 1 → `tests/regression/test_skill_commands_aqa.py::test_all_skills_are_valid` fails. The triple is also pinned by `tests/unit/tools/test_validate_skill_commands.py::test_valid_lifecycle_status_passes`. `superseded_by` appears nowhere in the repo. Wrong on the merits too: `train-policy` (RL policy training) does not supersede `robot-arm-trainer` (full-cycle platform skill) |
| 3 | Recreate `.claude/commands/` as thin slash-command wrappers | **REFUTED — fails an existing gate** | `tests/regression/test_skill_commands_aqa.py::test_legacy_commands_dir_stays_deleted` asserts the directory must not exist (foundry plan WS-F7a; restated in `CLAUDE.md`, `AGENTS.md`, `agent.md`, and `features.yaml` F-004). Redundant regardless: current Claude Code invokes `.claude/skills/<name>/SKILL.md` directly as `/<name>` |
| 4 | "WS-0 (secret-scan gate) … exist[s] only as prose in plans — nothing mechanical enforces [it]" | **REFUTED** | Shipped as F-015 (`status: "done"`): ci.yml `gitleaks` job (advisory `continue-on-error: true`, image pinned `docker://zricethezav/gitleaks:v8.24.3`, full-history `fetch-depth: 0`), `.gitleaks.toml` (regex-only allowlist — "NEVER by path" invariant in its header), `scripts/ci.sh` advisory stage, `tests/regression/test_secret_scan_gate.py`, `docs/runbooks/secret-scanning.md`. The real residual gap is narrower: no edit-time/pre-commit enforcement (WS-0.4's "pre-commit hook entry" never shipped — no `.pre-commit-config.yaml`, empty `.git/hooks/`) and the CI gate is still advisory |
| 5 | rev-B workstream items (WS-0, WS-1 "doc reconciliation", WS-5 "dashboard close-out") treated as open work | **REFUTED — stale-checkbox trap** | The rev-B plan's checkboxes are unchecked, but F-015–F-020 (WS-0/1/3/4.3/5/8) are all `done` in `features.yaml` with `implemented_in: a12c95b…`. F-008 is the only `todo` in the catalog. Also title drift: WS-1 is "Truth reconciliation", WS-5 is "Observability". The freeze rule itself is verified verbatim: "hardware readiness preempts all in-flight software streams" (rev-B lines 24-25, 59-61) |
| 6 | "rev-B finding #5: `check_branch_coverage.py` measures line only" | **PARTIALLY TRUE** | That is one of three sub-claims in finding #5 (the others: `test_smoke_harness.py` style, untested `write_summary()`). The substance is confirmed: branch coverage is measured nowhere (`[tool.coverage.run]` has no `branch` key; `--cov-branch` appears only in rev-B's own prose; the script gates changed-**line** coverage — "branch" in its name means git branch). Bonus live defect the draft missed: **README.md:255 falsely claims "85% branch coverage"** — exactly the "don't silently claim" failure rev-B's doctrine forbids |
| 7 | Materialization gated on `openspec validate … --strict`; archive-on-merge workflow | **REFUTED — cannot run here** | `openspec/project.md`: the tree is "documentation-only: no OpenSpec CLI/tooling is installed in this repository, and nothing in CI validates or consumes this tree." The authoritative mechanism is `features.yaml` (schema + `scripts/validate.py` + `.github/workflows/harness.yml`) + `docs/superpowers/` + ADRs; new bundles must extend the registry table in `openspec/project.md` and tie to an F-number |
| 8 | Exemplar agent frontmatter `tools: Read, Grep, Glob, Bash(git diff*), …` | **REFUTED — unsupported syntax** | Current sub-agent docs: the `tools:` field takes bare tool names only. Permission-style patterns belong in `settings.json` permissions or hook matchers. As written, the exemplar is invalid for all seven proposed agents |
| 9 | I-2: "Ruff `PLR2004` applies to all new Python" | **REFUTED** | No `PL*` rule family in `[tool.ruff.lint] select`. The repo's magic-number mechanism is the bespoke `scripts/check_no_hardcoded_values.py`, which gates changed lines under `src/mousedroid` only — new `tools/` code is outside it |
| 10 | I-4: "repo coverage gate (≥85% line) applies" to the new hook/linter code | **REFUTED as written** | Every `--cov` target and `[tool.coverage.run] source` is `src/mousedroid` — tests for `tools/claude_hooks/` would execute with zero measured coverage, invisible to the gate in both directions. Adjacent gaps the draft also missed: mypy runs on `src/` only, and GitHub ci.yml lints `src/ tests/` + `scripts/` but **not** `tools/` (only local `scripts/ci.sh` covers `tools/`) |
| 11 | D-7: `.mcp.json` = GitHub MCP now; Grafana/HF evaluate-first; nothing else | **PARTIALLY TRUE — major miss** | The selection discipline is sound, but the repo ships its **own** MCP server (`src/mousedroid/mcp/`, optional extra `mcp = ["mcp>=1.0", "anyio>=4.0"]`) whose canonical `.mcp.json` stanza is already specified in `docs/MCP_OPERATOR_GUIDE.md` ("Claude Code" section) and tracked as an open checkbox at `docs/MCP_NEXT_STEPS.md:51`. A `.mcp.json` that omits it contradicts shipped docs. `${VAR}` / `${VAR:-default}` expansion in `.mcp.json` is confirmed supported, so the secretless check-in works |
| 12 | hw-evidence-auditor: every hardware/perf claim in "README, BENCHMARKS, or features.yaml" must trace to a committed artifact under `reports/`/`smoke-reports/` | **PARTIALLY TRUE — conflicts with deliberate policy** | BENCHMARKS.md does not exist. `reports/` and `smoke-reports/` exist and are only partially tracked: several families are deliberately gitignored as "Local-only artefacts — never checked in" (`reports/trunk_sync/**`, `jetson_full_validation/**`, `jetson_smoke/**`, `dead_code/`, …). An auditor that flags every such artifact indicts policy, not drift. The underlying finding is real: the 2026-07-12 on-device run is referenced 11× (CHANGELOG.md:99, a regression-test docstring, two plan docs) with zero committed artifact |
| 13 | `jetson-smoke` skill "documents the RUN-MOTION consent token … (mirrors current practice)" | **PARTIALLY TRUE** | RUN-MOTION exists only as a prose consent phrase in the `smoke-reports/` pair; no code reads it. The mechanical motion gate is `MOUSEDROID_SMOKE_ALLOW_MOTION` / `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION` (`scripts/jetson_smoke_test.sh`, `src/mousedroid/config/schema.py`). Formalizing the phrase is a genuine gap; presenting it as the current enforced gate is not accurate |
| 14 | Census and bookkeeping: "three skills"; new epic `Dev Tooling & Governance`; feature ID via the rev-B policy; CLAUDE.md 848 / AGENTS.md 333 lines, "partially duplicates" | **PARTIALLY TRUE** | A **fourth** skill exists outside the validated layout: `.github/skills/jetson-hardware-debug/SKILL.md` (381 lines, two hardcoded IPv4 literals that would fail the validator's `hardcoded-host` rule if swept). The proposed epic does not exist (12 epics in use; `epic` is a schema-free string — `Hygiene` is the conventional fit). Next free ID is F-024 (F-009–F-014 permanently burned, ADR-013). Line counts verified exact. "Partially duplicates" overstates AGENTS.md: it self-declares a complementary axis (project vs worker surface), cross-references CLAUDE.md 4×, ~40–60 genuinely redundant lines of 333. The "~100 agent files across ~44 repos" figure is unverifiable from this repo `[Likely]` — carried as context, not fact |

One collision the draft never addresses: **WS-F7b** (foundry adoption, unstarted, gated on an
external repo milestone — `docs/superpowers/plans/2026-07-03-claude-code-foundry.md`) already
claims future `.claude/settings.json` edits (marketplace/plugin keys) and the eventual
retirement of `tools/validate_skill_commands.py`. Rev. B declares coexistence explicitly
(`proposal.md` → "Coexistence with WS-F7b").

## Corrected-design map (original → rev. B)

| Original | Rev. B correction |
|---|---|
| D-1 directory convention incl. `commands/` | D-1 without `commands/` (gate-pinned deleted; skills are slash-invocable directly) |
| D-2 `workforce.yaml` + Pydantic | Kept; keys extended (evidence policy, worktree prefix); honest note that `check_settings_identity.py` is unrelated to `.claude/settings.json` |
| D-3 hooks trio | Kept; secret scan wraps the repo's own gitleaks + `.gitleaks.toml`; explicit absent-binary posture (warn+allow default, `strict` flips to deny); freeze gate denies while F-008 ≠ `done`, fail-closed on unreadable catalog; PostToolUse stays report-only (platform cannot block there) |
| D-4 roster with pattern-restricted `tools:` | Same seven mandates; bare tool names; Bash restraint moved to prompt body + permissions; evidence auditor reconciled with the local-only artifact policy; BENCHMARKS reference dropped |
| D-5 lifecycle `active\|frozen\|legacy` + `superseded_by` | Vocabulary unchanged (`active\|frozen\|deferred`); the three frozen skills are **verify-only, byte-untouched**; five new skills |
| D-6 worktrees | Kept; cites existing precedent (`scripts/check_config_compat.py::worktree_at_sha`, `.dockerignore` "Claude worktrees" entry) |
| D-7 GitHub-only MCP | mousedroid server (per `docs/MCP_OPERATOR_GUIDE.md`, closes `docs/MCP_NEXT_STEPS.md:51`) + GitHub; Grafana/HF stay evaluate-first |
| D-8 CLAUDE.md → pointer stubs in `docs/claude/surfaces/` | Hybrid: trimmed root core + **nested per-directory CLAUDE.md** (documented auto-load) for subsystem contracts + `docs/claude/surfaces/` index only for cross-cutting surfaces |
| D-9 new `workforce-lint` GitHub Actions job | House pattern instead: `tests/regression/test_claude_workforce_aqa.py` (runs in the existing ci.yml matrix) + a `scripts/ci.sh` stage; plus a dedicated `--cov=tools/claude_hooks` invocation and the ci.yml `tools/` ruff-scope fix |
| Materialization gate `openspec validate … --strict` | Repo-native validation: skill validator + regression tier + harness fast tier + grep gates |

## What survives review unchanged

The draft's gap diagnosis is mostly right, and rev. B keeps its spine: there are no
subagents, no hooks anywhere in the repo, and no `.mcp.json`; `.claude/settings.json` is a
14-entry permissions allowlist only; CLAUDE.md is 848 lines of which 15 PR-stamped surface
sections are 82%, and its "CI Pipeline (5 stages)" section is stale against the 12-job
ci.yml; the freeze rule is enforced by prose plus one substring test, not at edit time.
The seven-agent roster shape, the three-hook concept, worktree-per-change isolation, the
secretless-MCP posture, config-driven thresholds, docs-last task ordering, and the explicit
deferred list all survive into rev. B.

## Load-bearing pins any implementation must satisfy

- `tests/regression/test_skill_commands_aqa.py` — skills valid, sweep clean, skills dir
  non-empty, `.claude/commands/` stays deleted.
- `tests/regression/test_next_steps_reconciled.py::test_frozen_skills_carry_status_frontmatter`
  — the three named skills keep `status: frozen` + `unfreeze:`.
- `tools/validate_skill_commands.py::_ALLOWED_SKILL_STATUSES` — `active|frozen|deferred`
  only; `description` required; backtick paths must exist; no IPv4 literals.
- `tests/unit/tools/test_validate_skill_commands.py` — pins the status triple and validator
  behaviour.
- `tests/regression/test_portfolio_reframe_aqa.py` — headline/forward docs must not carry
  the banned reframe tokens; any doc restructure keeps those files clean (and the rev. B
  AQA extends the sweep to moved content, allowlisting the literal repo slug).
- `tests/unit/test_scripts.py::TestCiSh` — presence-substring pins on `scripts/ci.sh`;
  new stages are additions, not rewrites.
- `features.schema.json` + `scripts/validate.py` + `.github/workflows/harness.yml` — any
  `features.yaml` edit (F-024) must validate; `done` requires `validation_command` +
  `implemented_in`.
- `.github/advisory_stages.yaml` — any `continue-on-error: true` job needs an entry
  (rev. B adds no such job).

## Appendix — refuted excerpts (verbatim, quoted for the record)

REFUTED — original "Why" §1 (skills asymmetry):

```text
One (`sim-test`) already carries the correct pattern — `status: frozen` + explicit
`unfreeze:` condition — the other two do not.
```

REFUTED — original D-5 lifecycle contract (out-of-vocabulary status + unknown key):

```yaml
status: active | frozen | legacy
superseded_by: <skill-name>                 # REQUIRED when status: legacy
```

REFUTED — original D-4 exemplar frontmatter (unsupported pattern syntax):

```yaml
tools: Read, Grep, Glob, Bash(git diff*), Bash(git log*), Bash(pytest *)
```

REFUTED — original materialization instruction (no such CLI in this repo):

```text
openspec validate mouse-droid-claude-workforce --strict
```
