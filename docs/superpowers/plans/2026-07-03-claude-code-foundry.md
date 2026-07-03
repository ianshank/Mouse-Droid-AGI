# claude-code-foundry — Implementation Plan

> **For agentic workers:** This plan targets a **NEW repository `ianshank/claude-code-foundry`
> — NOT this repo.** It was authored on 2026-07-03 from a session whose GitHub access was
> policy-scoped to `ianshank/mouse-droid-agi` only, so every cross-repo claim in it is tagged
> `[AUDIT-*]` and MUST be verified by the work stream that names it (WS-F0 for cross-repo
> facts, WS-F1 for official-docs facts) before any dependent task starts. Never treat an
> `[AUDIT-*]` claim as fact; never invent repo names or file contents to fill a placeholder.
> Execute work streams in order; **CONFIRM-FIRST gates are hard stops** — do not proceed past
> one without an explicit user decision. Steps use checkbox (`- [ ]`) syntax.
>
> Date: 2026-07-03 · Author: Claude (Mouse-Droid-AGI session) · Status: **Draft — pending
> WS-F0 audit** · Authored on branch: `claude/foundry-implementation-plan-gaaijr` (this repo)

**Goal:** Build a single-source-of-truth Claude Code **plugin marketplace repository** that
consolidates the ~100 agent files and ~35 skill files currently duplicated across ~44 repos
`[AUDIT-2]` (three incompatible layout conventions: `.claude/skills/`, `.agent/skills/`,
`.agents/skills/`) into versioned, tested, installable plugins. The repo doubles as the
marketplace: a generated root `.claude-plugin/marketplace.json` indexes plugins living under
`plugins/<name>/`, consumers pin versions (tag or sha) in `.claude/settings.json` instead of
vendoring files, and a Python validation harness gates every change through seven
mechanically verifiable stages.

**Tech Stack:** Python 3.11+, Pydantic v2, `jsonschema`, pytest (+`pytest-cov`, ≥90% on
`tools/`), structlog (JSON to stderr), ruff + mypy strict, Claude Agent SDK (headless plugin
smoke), GitHub Actions CI.

---

## Invariants (non-negotiable)

Verbatim from the source spec (§1.3 P3):

1. **Every release is additive or major-bumped — never silently breaking.**
2. **`marketplace.json` is generated from plugin manifests, never hand-edited.**
3. **No plugin merges without passing evals.**
4. **Hooks are deterministic and side-effect-scoped to the workspace** — no network calls
   from hooks, no secrets in any asset.
5. **No hardcoded values anywhere** — `${CLAUDE_PLUGIN_ROOT}` for all plugin-internal paths;
   every tunable in schema-validated `config/*.json`; no magic numbers in hook scripts.
   *(House addition: promoted to invariant from spec §2.1 hard constraints, matching
   Mouse-Droid-AGI's no-hardcoded-values standard.)*

State that must stay synchronized: `plugin.json` version ↔ `marketplace.json` entry ↔ git
tag. The version is the update cache key — a mismatch means "my fix never reached consumers".

---

## Autonomy boundaries

Verbatim from the source spec (§3.2):

- **AUTONOMOUS:** file creation within scope, dev-dependency installs, running tests/evals,
  refactoring for consistency, generating `marketplace.json` from manifests.
- **CONFIRM FIRST:** choosing which divergent variant of a duplicated skill becomes
  canonical; any change to public plugin naming/namespacing (renames break consumer slash
  commands); adding a new plugin (vs a new skill inside an existing plugin).
- **PROHIBITED:** hand-editing generated `marketplace.json`; committing to `main`;
  publishing releases; adding hooks that modify files outside the consumer's workspace.

Resource budgets (spec §3.3): max **5 iterations** on a failing eval before stopping and
reporting analysis; **one plugin ported end-to-end at a time** — no batch porting; ≤10 min
doc research before asking; structural facts come from official docs
(docs.claude.com / code.claude.com) only, never community blogs.

---

## Verification loop (seven stages, every PR)

Run in order; each stage must exit 0 before the next:

- [ ] 1. `ruff check tools/ tests/ && ruff format --check tools/ tests/`
- [ ] 2. `mypy tools/`
- [ ] 3. `python -m tools.validate` — JSON Schema validation of every `plugin.json`,
  `hooks.json`, config file, and skill/agent frontmatter; regenerates `marketplace.json`
  and **fails if drift detected**.
- [ ] 4. `pytest tests/ -v` — hook-script unit tests (fixture stdin payloads + temp
  `${CLAUDE_PLUGIN_ROOT}`), packaging tests, version-consistency tests
  (`plugin.json` ↔ `marketplace.json` ↔ latest tag).
- [ ] 5. `python -m tools.sdk_smoke` — Agent SDK headless load of each plugin via
  `plugins: [{type: "local", path: ...}]`; asserts each namespaced skill appears in
  `slash_commands` (verified 2026-07-03; re-verify via `[AUDIT-4]`).
- [ ] 6. Skill evals via the skill-creator loop per changed skill — `evals/evals.json`,
  subagent-isolated runs, grading all-pass; with-skill pass rate ≥ without-skill.
- [ ] 7. `python -m tools.token_budget` — parses the plugin-details output (exact CLI
  spelling is `[AUDIT-4]`); fails if any plugin's always-on cost exceeds its ceiling in
  `budget.json`.

Error-handling protocol (spec §4.2): **on eval regression** (passed before edit, fails
after) revert the edit and report diff analysis — never weaken the eval to pass. **On
version skew** regenerate `marketplace.json`; if the skew came from a manual edit, flag an
invariant-2 violation in the report. **On hook test failure over path assumptions** treat it
as a hardcoded-value bug — fix the script and add a regression test with a randomized temp
root.

---

## Global acceptance criteria (every WS / PR)

- [ ] All seven verification stages above green.
- [ ] **No hardcoded values** — hook scripts use `${CLAUDE_PLUGIN_ROOT}`; tunables live in
  schema-validated `config/*.json`; no literal hosts/IPs/paths anywhere in assets.
- [ ] **≥90% coverage on `tools/`** (`pytest --cov=tools --cov-fail-under=90`).
- [ ] structlog everywhere in Python tooling and hook scripts — JSON to stderr, event-style
  keys; **never `print()`**.
- [ ] CHANGELOG entry + semver bump with rationale (additive → minor; breaking → major with
  migration note); foundry `CLAUDE.md` decisions log updated.
- [ ] Known limitations documented (e.g., skills whose behavioral evals aren't written yet).

---

## Execution methodology

- Work streams run **strictly in order** WS-F0 → WS-F6 (WS-F7a/b execute in
  Mouse-Droid-AGI, see below). Within a WS, checkboxes are ordered.
- Two standing **CONFIRM-FIRST checkpoints**: (1) the WS-F3 canonical-variant decision —
  the six-variant diff from WS-F0 is a required attachment to the confirmation request;
  (2) any new plugin or any public plugin/skill rename, at any point.
- Milestone exits are hard gates: **M0 (WS-F1+F2) must be green on the empty marketplace
  before any asset is ported; the M2 forced-migration exit (WS-F4) blocks all of WS-F5.**
- Per-WS cadence: implement → run the seven stages → self-review against the global
  acceptance criteria → update foundry `CLAUDE.md` decisions log → PR (never to `main`
  directly).
- Eval budget: after 5 failed iterations on one eval, stop and report analysis instead of
  iterating further (spec §3.3).

---

## Non-goals (v0.x)

Verbatim from the source spec:

- **No runtime compatibility shim** for the three legacy directory conventions — consumers
  migrate, foundry doesn't adapt. Dead conventions get deleted, not adapted.
- **No MCP server implementations** in this repo — plugins may *reference* `.mcp.json`
  configs but server code lives elsewhere.
- **No submission to the official Anthropic marketplace** before ≥3 internal consumers are
  stable.
- **No plugin that hasn't been vendored/duplicated at least twice** — this repo consolidates
  proven assets, it doesn't speculate new ones.

---

## Context persistence (foundry CLAUDE.md contract)

The foundry repo maintains its own `CLAUDE.md` (authored in WS-F1, updated every WS) that
carries, per spec §5:

- The seven verification-stage commands (the loop above).
- An **architecture-decisions log** (date, decision, rationale). First three entries, seeded
  at WS-F1: *"skills/ not commands/ (commands/ is the legacy plugin format)"*,
  *"marketplace.json is generated — never hand-edited"*, *"one-plugin-at-a-time porting"*.
- **Canonicalization decisions**: for each consolidated duplicate, which repo's variant won
  and why (appended at every WS-F3/WS-F5 port).
- **Known consumer pins**: which repos consume which plugin at which version — this drives
  breaking-change impact analysis and feeds the M4 pin registry (WS-F6).

---

## Prior art in this repo (Mouse-Droid-AGI)

**(a) `tools/validate_skill_commands.py` — the validator core to generalize (WS-F2).**
Lift: the YAML front-matter parse with required non-empty `description`; backtick-path
*discovery* from the body (never enumerated) with an extension allowlist and exclusion of
format/glob tokens (`{}`, `*`, `$`, `<>`) so illustrative patterns aren't false-flagged; the
deliberate IPv4-literal-only hardcoded-host check (narrow by design to avoid false positives
on prose); typed issue codes + exit-1-on-any-issue CLI shape; defensive handling of
absolute/`..`-traversal refs and unreadable files. Deltas the foundry needs: `SKILL.md`
directory layout instead of flat command files, `${CLAUDE_PLUGIN_ROOT}` enforcement in hook
scripts (a literal plugin-internal path is an error), and a JSON Schema layer on top of the
frontmatter lint.

**(b) Legacy-layout debt — Mouse-Droid is itself a migration case.** This repo uses the
legacy `.claude/commands/` layout (`robot-arm-trainer.md`, `sim-test.md`,
`train-policy.md`). Per the non-goals, foundry ships **no shim** — this repo migrates itself
(WS-F7a). Those three skills are Mouse-Droid-specific and (unless the WS-F0 audit finds them
duplicated ≥2×) they **stay local** — they do not become foundry plugins.

**(c) Consumer registry entry.** Mouse-Droid-AGI is a named future consumer: pre-draft its
row in the M4 consumer-pin registry as `planned` — candidate adoption is the quality-gate
plugin post-v0.1.0 (WS-F7b), plus retiring the local validator in favor of the generalized
foundry one.

---

## Prerequisites & unverified facts

No WS task may treat a tagged claim below as fact. The executing session replaces each tag
with evidence (file paths, diff output, fetched-doc citations) and only then unblocks the
dependent work streams. Every official-docs claim in this plan carries
"(verified 2026-07-03; re-verify via `[AUDIT-4]`)".

| Tag | Claim (from spec, unverifiable from the authoring session) | Verified by | Blocks |
|---|---|---|---|
| `[AUDIT-1]` | Six divergent quality-gate skill variants exist across repos `<TBD: repo list from WS-F0>` | WS-F0 | WS-F3 |
| `[AUDIT-2]` | ~100 agent files / ~35 skill files / ~44 repos duplication counts | WS-F0 | WS-F5 priority order |
| `[AUDIT-3]` | `ianshank/Agents` CI shape + vendored quality-gate paths | WS-F0 | WS-F4 |
| `[AUDIT-4]` | Current `plugin.json`/`marketplace.json` schemas; formal `${CLAUDE_PLUGIN_ROOT}` semantics (appears in official hook examples but is not formally specified as of 2026-07-03); exact CLI spelling for plugin details / always-on token cost (`claude plugin details <name>` unconfirmed as of 2026-07-03) | WS-F1 doc-fetch into `docs/reference/` (dated) | WS-F1 schemas, WS-F2 `sdk_smoke`/`token_budget` |

Facts already confirmed against official docs (verified 2026-07-03; re-verify via
`[AUDIT-4]`): `skills/` is preferred and `commands/` is the legacy plugin format; a repo
doubles as a marketplace via root `.claude-plugin/marketplace.json` with plugins in
subdirectories via the entry's `source` path; consumer settings keys are
`extraKnownMarketplaces` (tag/sha pinning lives here, on the `source` config — `ref`/sha
fields) and `enabledPlugins` (maps `plugin-name@marketplace-name` to a boolean — enable/
disable only, no pinning); installed
plugin skills namespace as `/plugin-name:skill-name`; the Agent SDK loads plugins via
`plugins: [{type: "local", path: ...}]` and namespaced skills appear in `slash_commands`;
plugin hooks merge with user/project hooks automatically on enable.

---

## WS-F0 — Cross-repo audit (prerequisite; read-only; no foundry code)

**Files (foundry-repo-relative):** `docs/audit/duplication-matrix.md`,
`docs/audit/quality-gate-variants/` (six variant snapshots + unified diff),
`docs/audit/consumer-agents-ci.md`.

- [ ] Enumerate the candidate repos (~44 `[AUDIT-2]`) and produce
  `docs/audit/duplication-matrix.md`: rows = agent/skill asset (by content hash + name),
  columns = repo, cell = path + layout convention (`.claude/skills/` / `.agent/skills/` /
  `.agents/skills/` / `.claude/commands/`). Record exact counts, replacing the `~100/~35/~44`
  estimates.
- [ ] Locate all quality-gate variants `[AUDIT-1]`; snapshot each verbatim under
  `docs/audit/quality-gate-variants/<repo>/` and produce a unified six-way diff + a table of
  behavioral differences (triggers, thresholds, tool permissions, test assertions).
- [ ] Capture `ianshank/Agents` CI `[AUDIT-3]`: workflow files, the jobs that exercise the
  vendored quality-gate copy, and the exact vendored paths to delete in WS-F4.
- [ ] Verify the ≥2×-duplication bar for every plugin candidate; drop any that fail it
  (non-goal 4).

**Acceptance:** matrix + variant snapshots + diff + CI capture committed; every `[AUDIT-1/2/3]`
tag in this plan replaceable with a file reference; duplication counts recorded with evidence.

---

## WS-F1 — M0 skeleton: layout, schemas, CLAUDE.md, C4, CI

**Files (foundry-repo-relative):** `.claude-plugin/marketplace.json` (generated),
`plugins/` (empty), `schemas/*.json`, `budget.json`, `pyproject.toml`, `CLAUDE.md`,
`CHANGELOG.md`, `docs/reference/` (dated doc snapshots), `docs/architecture/c4-foundry.md`,
`docs/architecture/architecture.yaml`, `.github/workflows/verify.yml`.

- [ ] Fetch and cache the official plugin + marketplace + skills reference docs into
  `docs/reference/` with fetch dates; resolve every `[AUDIT-4]` item (marketplace/plugin
  schema fields, `${CLAUDE_PLUGIN_ROOT}` semantics, the plugin-details CLI spelling for
  stage 7) from these snapshots — structural facts come from the snapshots, never memory.
- [ ] Scaffold the repo layout per spec §6.2: `.claude-plugin/`, `plugins/`, `schemas/`,
  `tools/`, `tests/`, `docs/{architecture,reference,migration}/`, `budget.json`,
  `CLAUDE.md`, `CHANGELOG.md`, `pyproject.toml`.
- [ ] Author JSON Schemas in `schemas/`: plugin manifest, marketplace, hooks, skill/agent
  frontmatter, per-plugin config, `budget.json`, `evals.json`. Schemas are the single source
  of truth — this plan deliberately does not restate field lists (risk R1).
- [ ] Author foundry `CLAUDE.md` per the context-persistence contract above, seeding the
  three initial decisions-log entries.
- [ ] Author `docs/architecture/c4-foundry.md` with the spec §6.3 Level-1 (system context)
  and Level-2 (container) Mermaid diagrams, and `architecture.yaml` declaring the intended
  `tools/` component edges (validator, generator, eval adapter, SDK smoke, budget checker,
  version checker) for the WS-F2 import-graph assertion.
- [ ] Wire `.github/workflows/verify.yml` running all seven stages on every PR; stages 3/5/7
  are no-ops-that-still-execute on the empty marketplace (they must exit 0, not be skipped).

**Acceptance:** all seven stages exit 0 on the empty marketplace (**M0 exit gate**);
`docs/reference/` snapshots dated; `[AUDIT-4]` resolved with citations.

---

## WS-F2 — M0 tooling: the validation harness

**Files (foundry-repo-relative):** `tools/validate.py`, `tools/generate_marketplace.py`,
`tools/sdk_smoke.py`, `tools/token_budget.py`, `tools/versioning.py`, `tools/eval_runner.py`,
`tests/tools/`, `reports/` (gitignored).

- [ ] `tools/generate_marketplace.py`: derive `marketplace.json` deterministically from all
  `plugins/*/.claude-plugin/plugin.json` manifests (sorted keys, stable ordering).
- [ ] `tools/validate.py`: schema-validate every manifest, `hooks.json`, config file, and
  skill/agent frontmatter; then regenerate the marketplace **in-memory and diff against the
  committed file — any drift is a hard failure** (invariant 2). Emit machine-readable
  `reports/validation.json` alongside human output; CI archives it.
- [ ] Generalize `tools/validate_skill_commands.py` from Mouse-Droid-AGI (prior-art §a):
  port the frontmatter parse, backtick-path-existence discovery with the glob-token
  exclusion set, and the IPv4-literal host check; extend with `${CLAUDE_PLUGIN_ROOT}`
  enforcement — any literal plugin-internal path in a hook script or `hooks.json` fails.
- [ ] `tools/versioning.py`: the triple-check `plugin.json` ↔ `marketplace.json` ↔ latest
  git tag; runs inside stage 4 tests and as a tag-time CI job.
- [ ] `tools/sdk_smoke.py`: Agent SDK headless load of each plugin
  (`plugins: [{type: "local", path: ...}]`, verified 2026-07-03; re-verify via `[AUDIT-4]`);
  assert every skill registers as `/plugin-name:skill-name` in `slash_commands`.
- [ ] `tools/token_budget.py`: parse the plugin-details output (spelling from `[AUDIT-4]`);
  compare each plugin's always-on cost against `budget.json`; fail over-ceiling; record the
  cost in each plugin's README.
- [ ] `tools/eval_runner.py`: adapter for the skill-creator eval loop — run each changed
  skill's `evals/evals.json` in subagent isolation, require grading all-pass, and compare
  with-skill vs without-skill pass rates. Define the **minimal-viable eval contract** here,
  once (risk R2): skill loads; triggers on its canonical prompt; output-shape assertion.
  Depth beyond that is M4 work, not a porting gate.
- [ ] CI assertion that the actual `tools/` import graph matches
  `docs/architecture/architecture.yaml` declared edges (grimp — same pattern as the
  architecture-drift skill, which itself becomes a foundry plugin in WS-F5).
- [ ] Tests for all of the above in `tests/tools/`, including: a test that hand-editing
  `marketplace.json` makes `tools.validate` exit non-zero; hook-path regression tests with a
  randomized temp `${CLAUDE_PLUGIN_ROOT}`.
- [ ] `FOUNDRY_DEBUG=1` support in all tools: raises log level and dumps resolved config
  with `${CLAUDE_PLUGIN_ROOT}` expansion shown.

**Acceptance:** `pytest --cov=tools --cov-fail-under=90` green; all seven stages still exit
0 on the empty marketplace; the hand-edit-detection test proves invariant 2 is mechanical.

---

## WS-F3 — M1: the quality-gate plugin

**Files (foundry-repo-relative):** `plugins/quality-gate/.claude-plugin/plugin.json`,
`plugins/quality-gate/skills/quality-gate/{SKILL.md,evals/evals.json}`,
`plugins/quality-gate/agents/gate-reviewer.md`, `plugins/quality-gate/hooks/hooks.json`,
`plugins/quality-gate/bin/`, `plugins/quality-gate/config/`, `plugins/quality-gate/README.md`.

- [ ] **CONFIRM FIRST — hard stop.** Present the WS-F0 six-variant diff and behavioral-
  difference table to the user; obtain an explicit canonical-variant decision (pick or
  merge). Do not write any plugin file before this decision. Record the decision + rationale
  in foundry `CLAUDE.md` (canonicalization log).
- [ ] Port the canonical skill to `skills/quality-gate/SKILL.md` — frontmatter with `name`,
  `description` (≤~500 words always-on; heavy content in the body), and explicit
  `allowed-tools` where tool permissions are needed. Keep the **union of test assertions**
  from all six variants when merging.
- [ ] Port/author `agents/gate-reviewer.md` and the PreToolUse enforcement hook:
  `hooks/hooks.json` referencing `bin/` scripts via `${CLAUDE_PLUGIN_ROOT}`; Python for
  anything with logic, bash only for one-liners; thresholds/tunables in schema-validated
  `config/*.json`, no magic numbers.
- [ ] Hook unit tests: fixture stdin payloads, randomized temp `${CLAUDE_PLUGIN_ROOT}`,
  assertions that the hook is deterministic, makes **no network calls**, and touches
  **nothing outside the workspace** (invariant 4). Hooks log their full received stdin
  payload at debug level.
- [ ] Author `evals/evals.json` per the WS-F2 minimal-viable contract; run the eval loop to
  all-pass (max 5 iterations, then stop and report).
- [ ] Regenerate marketplace; record the always-on token cost in the plugin README; set the
  `budget.json` ceiling.
- [ ] Manual smoke on a scratch project: `/plugin marketplace add <local path>` →
  `/plugin install quality-gate@claude-code-foundry` → invoke `/quality-gate:quality-gate`.

**Acceptance:** all seven stages green with one plugin present; evals all-pass; SDK smoke
shows the namespaced skill; manual local install + invoke works (**M1 exit gate**).

---

## WS-F4 — M2: v0.1.0 release + forced migration

**Files:** foundry `CHANGELOG.md`, release PR; migration PR in `ianshank/Agents`
(separate repo, authored per-repo as its own PR).

- [ ] Prepare the v0.1.0 release PR: CHANGELOG entry, version stamped consistently
  (`plugin.json` = `marketplace.json` entry = tag-to-be), versioning triple-check green.
  **Publishing the release/tag is a human action — the agent prepares, the maintainer
  publishes (PROHIBITED list).**
- [ ] Author the migration PR to `ianshank/Agents` `[AUDIT-3]`: delete the vendored
  quality-gate files; add `extraKnownMarketplaces` (**sha-pinned via its `source`
  config** for reproducible installs) + `enabledPlugins` (boolean enable by plugin ID) to
  `.claude/settings.json`; write
  `docs/migration/agents.md` in foundry documenting the steps as the template for later
  consumers. Keep the vendored-copy deletion a **single revert-ready commit** (risk R3).
- [ ] Drive consumer CI green. If it cannot go green after the WS's iteration budget,
  escalate to the user with diagnosis — **never weaken the gate** (risk R3).
- [ ] Record `ianshank/Agents` in the foundry `CLAUDE.md` consumer-pins registry.

**Acceptance:** consumer PR merged with its CI green and the vendored copy deleted. **This
is the M2 exit gate and it blocks ALL of WS-F5.**

---

## WS-F5 — M3: incremental porting (one plugin at a time)

**Files:** `plugins/<name>/…` per plugin; `docs/migration/<consumer>.md` per migration.

Priority order comes from the WS-F0 duplication matrix — highest duplication count first.
Seed table (populate from WS-F0; do not invent entries):

| Priority | Plugin | Duplication count | Source repos |
|---|---|---|---|
| 1 | `<TBD: from WS-F0 matrix>` | `<TBD>` | `<TBD>` |
| 2 | architecture-drift skill (named in spec §6.3 as an M3 port) | `<TBD>` | `<TBD>` |

Per-plugin checklist template (repeat verbatim for each port — do not batch):

- [ ] **CONFIRM FIRST:** plugin name/namespace + canonical-variant choice (with WS-F0 diff
  attached) approved by the user.
- [ ] Port assets (rewrite, don't just copy — converge divergent copies keeping the union
  of test assertions); `${CLAUDE_PLUGIN_ROOT}` throughout; tunables to `config/*.json`.
- [ ] Author `evals/evals.json`; eval loop all-pass (≤5 iterations).
- [ ] Hook tests (if any hooks): fixture stdin, randomized temp root, no-network +
  workspace-scope assertions.
- [ ] Regenerate marketplace; token cost recorded + `budget.json` ceiling set.
- [ ] All seven stages green; minor-version bump + CHANGELOG (additive) — major only with a
  migration note.
- [ ] One consumer migration PR (vendored copy deleted, sha-pinned) merged with CI green;
  consumer-pins registry updated.
- [ ] Canonicalization decision logged in foundry `CLAUDE.md`.

Also in WS-F5 scope:

- [ ] `docs/migration/convention-adapter.md`: migration guides (docs, **not** runtime
  shims — non-goal 1) for consumers on the dead `.agent/skills/` and `.agents/skills/`
  layouts.

**Acceptance:** per-plugin — template fully checked. Global — every marketplace entry has
≥1 consumer pin recorded; no plugin in the catalog below the 2×-duplication bar.

---

## WS-F6 — M4: hardening

**Files (foundry-repo-relative):** `.github/workflows/release.yml`,
`tools/token_budget.py` (trend mode), `docs/consumers.md`,
`docs/architecture/c4-foundry.md` (refresh).

- [ ] Auto-release workflow: on human-pushed tag → run all seven stages → publish the
  release; ship a dry-run mode first and keep publishing human-triggered.
- [ ] Token-budget trend report: persist per-plugin always-on cost per release; fail CI on
  regression beyond the recorded ceiling; surface the trend in the report artifact.
- [ ] `docs/consumers.md` pin registry (generated from the `CLAUDE.md` consumer-pins data or
  vice versa — one source of truth); `tools.validate` checks registry entries reference
  existing plugins + real versions.
- [ ] Refresh the C4 doc (add the Level-3 component diagram now that `tools/` is stable) and
  re-assert `architecture.yaml` edges.

**Acceptance:** release dry-run green end-to-end; pin registry validated by stage 3; trend
report produced in CI artifacts.

---

## WS-F7a — Mouse-Droid-AGI local layout migration (this repo; ungated)

**Files (this repo):** `.claude/commands/{robot-arm-trainer,sim-test,train-policy}.md` →
`.claude/skills/<name>/SKILL.md`; `tools/validate_skill_commands.py`;
`tests/regression/test_skill_commands_aqa.py`; `scripts/ci.sh`.

No foundry dependency — schedulable any time. The three skills are Mouse-Droid-specific and
stay local (below the 2×-duplication bar unless WS-F0 finds otherwise).

- [ ] Migrate the three command files to the `skills/` layout, preserving frontmatter
  `description` and body paths.
- [ ] Update `tools/validate_skill_commands.py` (or its invocation) + the pinning regression
  test + `scripts/ci.sh` for the new location.

**Acceptance:** `bash scripts/ci.sh` green; `tests/regression/test_skill_commands_aqa.py`
green against the new layout; slash commands still invocable.

---

## WS-F7b — Mouse-Droid-AGI foundry adoption (this repo; gated on M2 v0.1.0)

**Files (this repo):** `.claude/settings.json`; `tools/validate_skill_commands.py`
(retirement); `scripts/ci.sh`; `tests/regression/test_skill_commands_aqa.py`.

- [ ] Add `extraKnownMarketplaces` (claude-code-foundry, sha-pinned) + `enabledPlugins`
  (`quality-gate@claude-code-foundry`) to `.claude/settings.json`.
- [ ] Once the generalized foundry validator covers this repo's checks, retire the local
  `tools/validate_skill_commands.py` in favor of it (keep the regression test pinning the
  *behavior*, pointed at the new tool).
- [ ] Register Mouse-Droid-AGI in the foundry consumer-pins registry (flip the `planned`
  row to active).

**Acceptance:** `bash scripts/ci.sh` green with the foundry plugin enabled; consumer-pins
registry updated in foundry.

---

## Milestone map

| Milestone | Work streams | Exit criterion | Gate |
|---|---|---|---|
| M0 — Skeleton + harness | WS-F1, WS-F2 | All seven stages exit 0 on the **empty** marketplace | Blocks any asset porting |
| M1 — quality-gate plugin | WS-F3 (after WS-F0) | Evals pass, SDK smoke passes, local install + invoke works | Blocks M2 |
| M2 — v0.1.0 + forced migration | WS-F4 | `ianshank/Agents` CI green with vendored copy **deleted** | **Blocks ALL of WS-F5 (M3)** |
| M3 — Incremental porting | WS-F5 | Per-plugin template fully checked; every entry has ≥1 consumer pin | Per-plugin, sequential |
| M4 — Hardening | WS-F6 | Release dry-run green; budget trend + pin registry in CI | — |
| (Consumer) | WS-F7a (ungated), WS-F7b (needs M2) | This repo's CI green post-migration/adoption | — |

---

## Risk register

| # | Risk | Mitigation | Trip-wire |
|---|---|---|---|
| R1 | Spec/doc drift — plugin/marketplace formats evolve, or a from-memory "fact" in this plan is stale | `[AUDIT-4]` doc-fetch in WS-F1 with dated snapshots; `schemas/` is the single source of truth; this plan never restates JSON field lists; every docs claim carries an as-of stamp | `tools.validate` failing on a freshly generated marketplace |
| R2 | Eval-authoring cost stalls porting ("no merge without evals" + expensive evals) | Minimal-viable eval contract defined once in WS-F2; depth deferred to M4; 5-iteration budget then escalate | A single port exceeding its iteration budget |
| R3 | M2 forced-migration gate blocks everything (consumer CI red, vendored copy load-bearing) | WS-F0 captures consumer CI up front; single revert-ready deletion commit; escalate-don't-weaken fallback written into WS-F4 | Consumer CI not green within the WS iteration budget |
| R4 | Version skew across plugin.json / marketplace.json / tags once multiple plugins release | `tools/versioning.py` triple-check in stage 4 + tag-time CI; releases human-published until the M4 dry-run-verified workflow | Triple-check failure on any tag |
| R5 | Canonical-variant choice made unilaterally by an executing agent | CONFIRM-FIRST as the *first checkbox* of WS-F3 and of every WS-F5 port; WS-F0 diff is a required attachment; restated in the header callout | Any plugin file written before a logged canonicalization decision |

---

## Logging & debugging standards

- All Python tooling and hook scripts: **structlog, JSON to stderr**, event-style keys
  (`hook_fired`, `gate_blocked`, `schema_violation`); never `print()`.
- **`FOUNDRY_DEBUG=1`** raises log level and dumps resolved config with
  `${CLAUDE_PLUGIN_ROOT}` expansion shown — the primary path-debugging aid on foreign
  machines.
- Hook scripts log their **full received stdin payload at debug level** — the #1 debugging
  need for PreToolUse hooks.
- The validator emits machine-readable **`reports/validation.json`** in addition to human
  output; CI archives it.
