# OpenSpec Change: Claude Code Workforce Modernization (agents, skills, hooks, worktrees, MCP) — rev. B

> **Archival note.** Rev. B of a user-supplied draft, rewritten after the objective peer
> review in `peer-review.md` (all factual corrections and refuted mechanisms are recorded
> there). This tree is documentation-only; the repo-native artifacts named in
> `openspec/project.md` are authoritative. When this document disagrees with the tree, the
> tree wins.

- **change_id**: `mouse-droid-claude-workforce`
- **project**: MouseDroid (repository `ianshank/Mouse-Droid-AGI`)
- **status**: proposed
- **feature_id**: F-024 (expected — next free per ADR-013 F-number namespaces; F-009–F-014
  are permanently burned; reserved in `features.yaml` at task 0.2, never hardcoded before
  then)
- **epic**: `Hygiene` (existing epic, used 3×; `features.schema.json` leaves `epic`
  unconstrained, so this is conventional reuse over minting a 13th one-off)
- **owner**: TBD
- **created**: 2026-07-24
- **basis_commit**: `c9ac439`

## Why

Each premise below is verified against the tree at the basis commit (`peer-review.md` holds
the evidence; the draft this replaces got several of these wrong):

1. **No workforce surface exists.** There are no `.claude/agents/`, no Claude Code hooks
   anywhere in the repo, and no `.mcp.json`. `.claude/settings.json` is a 14-entry
   permissions allowlist and nothing else.
2. **CLAUDE.md is 848 lines of accretion.** Nine evergreen sections plus fifteen PR-stamped
   surface sections (82% of the file); its "CI Pipeline (5 stages)" section is stale against
   the 12-job `ci.yml`. AGENTS.md (333 lines) is a deliberately complementary worker
   contract with ~40–60 genuinely redundant lines.
3. **Secret scanning stops at CI, and CI is advisory.** The F-015 gitleaks gate is real
   (pinned image, full-history, regex-only allowlist) but `continue-on-error: true`, and
   WS-0.4's "pre-commit hook entry" never shipped — nothing scans at edit time. A PreToolUse
   hook closes exactly that gap with the same scanner and config.
4. **The freeze rule is prose.** Rev-B's "hardware readiness preempts all in-flight software
   streams" and the three skills' F-008 unfreeze condition are enforced only by plan text
   plus one substring regression test; nothing blocks a capability edit at edit time while
   F-008 (`status: "todo"`, the catalog's only todo) is open.
5. **Branch coverage is claimed but not measured.** No `branch =` key, no `--cov-branch`
   anywhere; `scripts/check_branch_coverage.py` gates changed-line coverage despite its
   name — and README.md:255 falsely claims "85% branch coverage" today.
6. **A fourth skill sits outside the validated layout.** `.github/skills/jetson-hardware-debug/SKILL.md`
   (381 lines) is never swept by `tools/validate_skill_commands.py` and carries two
   hardcoded IPv4 literals that would fail its `hardcoded-host` rule.
7. **The `.mcp.json` template is a tracked open item.** `docs/MCP_NEXT_STEPS.md:51` has the
   open checkbox; `docs/MCP_OPERATOR_GUIDE.md` already specifies the canonical stanza for
   the repo's own MCP server (`src/mousedroid/mcp/`) that no checked-in file realizes.
8. **The evidence discipline is ad-hoc.** The 2026-07-12 on-device validation run is
   referenced 11× (CHANGELOG, a regression-test docstring, two plan docs) with zero
   committed artifact — partly by deliberate `.gitignore` policy ("Local-only artefacts —
   never checked in"). The discipline needs a declared rule that distinguishes tracked
   evidence from declared local-only evidence chains, instead of prose convention.

## What Changes

- **ADD** `.claude/agents/` — seven subagents (`openspec-author`, `test-engineer`,
  `peer-reviewer`, `security-scanner`, `doc-reconciler`, `hw-evidence-auditor`,
  `config-guardian`) that collaborate by default (I-6), frontmatter restricted to
  platform-supported keys with bare tool names.
- **ADD** `tools/claude_hooks/` — three tested, config-driven hooks (`secret_scan.py`
  PreToolUse, `freeze_gate.py` PreToolUse, `post_edit_check.py` PostToolUse report-only) +
  `config.py` (`WorkforceConfig`, Pydantic v2, `extra="forbid"`), wired additively into
  `.claude/settings.json`.
- **ADD** `.claude/workforce.yaml` — the single config source for every workforce
  threshold, gate key, glob, and budget.
- **ADD** five skills — `openspec-change`, `coverage-gate`, `evidence-commit`,
  `worktree-flow` (active), `jetson-smoke` (Tier-3 section frozen behind the typed
  RUN-MOTION consent phrase, alongside the mechanical `MOUSEDROID_SMOKE_ALLOW_MOTION`
  gate it documents).
- **ADD** `.mcp.json` — checked-in, secretless (`${VAR}` expansion): the repo's own
  `mousedroid` MCP server exactly per `docs/MCP_OPERATOR_GUIDE.md` (closing
  `docs/MCP_NEXT_STEPS.md:51`) plus GitHub MCP. Grafana/HF remain evaluate-first notes,
  not servers.
- **ADD** `docs/runbooks/worktrees.md` + the `worktree-flow` skill — one worktree per
  change-id for parallel agent isolation.
- **ADD** `tests/regression/test_claude_workforce_aqa.py` + a `scripts/ci.sh` stage — the
  house AQA pattern (reusing `validate_skill_commands` helpers), NOT a new GitHub Actions
  job; plus a dedicated `--cov=tools/claude_hooks` coverage invocation (line-gated,
  branch measured and reported advisory-first).
- **VERIFY (not modify)** the three existing skills — byte-untouched; lifecycle vocabulary
  stays `active|frozen|deferred`.
- **RESTRUCTURE** CLAUDE.md docs-last (hybrid: trimmed root core + nested per-directory
  CLAUDE.md files + `docs/claude/surfaces/` index) and dedupe AGENTS.md on its declared
  worker-contract axis.
- **FIX** README.md:255 (branch-coverage falsehood → "85% line" until branch measurement
  lands); **FIX** the ci.yml ruff scope to include `tools/` (closing the ci.sh↔ci.yml
  divergence).
- **NO** capability code. **NO** `.claude/commands/` (gate-pinned deleted). **NO** new
  lifecycle statuses. **NO** new GitHub Actions job.

## Invariants (binding on every task)

| # | Invariant | Enforcement |
|---|---|---|
| I-1 | Additive/backwards-compatible: existing skill frontmatter, permission entries, and YAML files load unchanged; moved doc content stays resolvable | AQA test + PR review |
| I-2 | No hardcoded values: thresholds, gate keys, globs, budgets live in `.claude/workforce.yaml` via `WorkforceConfig` (`extra="forbid"`). Honest note: the repo's magic-number gates are `scripts/check_no_hardcoded_values.py` (src-scoped) plus review — `PLR2004` is not enabled and is not claimed | schema tests + review |
| I-3 | Portability: nothing under `.claude/` carries an absolute path or IPv4 literal; repo access via `$CLAUDE_PROJECT_DIR` and config | AQA reusing `find_hardcoded_hosts` / `referenced_repo_paths` |
| I-4 | Tested tooling: hooks + config + AQA module covered by a dedicated `--cov=tools/claude_hooks` invocation ≥ `coverage.tools_line_min`; branch coverage measured and reported advisory-first (rev-B doctrine: never claim an unmeasured metric) | ci.sh stage |
| I-5 | Freeze compliance: capability-touching skills/agents carry `status: frozen` + `unfreeze:` referencing F-008, matching the existing three | AQA + validator |
| I-6 | Collaboration default: the CLAUDE.md orchestration directive instructs proactive delegation to the roster unless the user opts out | doc + review |

## Impact

- **Affected specs**: `claude-workforce` (new), `dev-governance` (new).
- **Affected code** (at implementation, per `tasks.md`): `.claude/**` (additive),
  `tools/claude_hooks/**` (new), `.mcp.json` (new), `tests/regression/test_claude_workforce_aqa.py`
  (new), `tests/unit/tools/**` (new tests), `scripts/ci.sh` (+ `TestCiSh` pin additions),
  `.github/workflows/ci.yml` (one ruff-scope line), `CLAUDE.md`/`AGENTS.md`/`README.md:255`,
  nested `CLAUDE.md` files, `docs/claude/**` (new), `docs/runbooks/worktrees.md` (new),
  `features.yaml` (F-024 row), `openspec/project.md` (registry row). This bundle itself is
  docs-only.
- **Risk**: Low–Medium. Everything is additive; hooks default to safe postures
  (secret-scan warn+allow when the scanner is absent, freeze-gate scoped to configured
  globs); zero changes to the robot runtime or the 30 Hz loop.
- **Breaking changes**: none.

## Coexistence with WS-F7b (foundry adoption)

WS-F7b (unstarted, gated on the external foundry repo's milestone) already claims future
`.claude/settings.json` edits and the eventual retirement of
`tools/validate_skill_commands.py`. This change stays out of its way: settings edits here
are additive (a `hooks` block only; marketplace/plugin keys untouched), and the new AQA
module **reuses** the legacy validator's public helpers (`find_hardcoded_hosts`,
`referenced_repo_paths` — the `test_foundry_plan_doc.py` precedent) instead of growing the
validator. At WS-F7b retirement time the two pure helpers migrate to a neutral module and
importers re-point; that note lives in `design.md` D-9.

## Non-goals / Deferred (separate changes, do not fold in)

History purge + repo rename; committing/backfilling the 2026-07-12 on-device validation
evidence; on-device measurement of the 30 Hz loop-rate target; branch-coverage threshold
promotion (advisory → blocking); gitleaks CI job promotion (advisory → blocking; the
7-green-run tracker exists in `.github/advisory_stages.yaml`); Grafana/HF MCP adoption;
any F-02x capability work (gated by the freeze-gate hook this change ships); WS-F7b itself.

## Spec Deltas

- `specs/claude-workforce/spec.md` — 6 requirements (lifecycle, portability, agent tool
  declarations, freeze gate, edit-time secret scanning, MCP configuration).
- `specs/dev-governance/spec.md` — 5 requirements (evidence-backed claims, single config
  source, tested governance tooling, truthful coverage claims, additive compatibility).

## Tasks

`tasks.md` — seven phases (0 ground truth → 6 docs-last), each landing green before the
next; deferred list explicit.

## Validation

Repo-native (no OpenSpec CLI exists here — `openspec/project.md`):

- `python tools/validate_skill_commands.py` → `OK` (skill surface undisturbed).
- `python -m pytest tests/regression -m "not hardware" --import-mode=importlib` → green.
- `python scripts/validate.py --tier fast` → green (harness catalog remains valid).
- Grep gates: no out-of-vocabulary lifecycle tokens or unknown frontmatter keys outside
  the labeled REFUTED quotes in `peer-review.md`; `.claude/commands/` remains absent.
