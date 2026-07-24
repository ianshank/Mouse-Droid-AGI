# Design — Claude Code Workforce Modernization (rev. B)

Corrections from `peer-review.md` are baked in throughout; where a decision reverses the
original draft, the reversal is called out inline.

## D-1. Directory convention (the reusable shape)

```
.claude/
  settings.json          # permissions (existing) + hooks wiring (additive)
  workforce.yaml         # SINGLE config source: thresholds, gate keys, globs, budgets
  agents/                # seven subagents (markdown + YAML frontmatter)
  skills/                # existing three (byte-untouched) + five new
.mcp.json                # project-scope MCP servers (checked in, secretless)
tools/claude_hooks/      # tested Python hook implementations + Pydantic config
  config.py              #   WorkforceConfig (Pydantic v2, extra="forbid")
  paths.py               #   repo-root resolution + separator-respecting globs
  logging_setup.py       #   stderr-only structured logging (structlog or fallback)
  hookio.py              #   hook stdin/stdout protocol
  portability.py         #   absolute-path rule for the AQA sweep
  secret_scan.py         #   PreToolUse  (blocking)
  freeze_gate.py         #   PreToolUse  (blocking)
  post_edit_check.py     #   PostToolUse (report-only)
tests/regression/test_claude_workforce_aqa.py
tests/unit/tools/claude_hooks/   # one test module per hook module
docs/claude/surfaces/    # cross-cutting surface docs + index (D-8)
docs/runbooks/claude-workforce-hooks.md
docs/runbooks/worktrees.md
```

The four primitives (`paths`, `logging_setup`, `hookio`, `portability`) exist so
the three hooks carry policy only: each is independently unit-tested and reused
by the AQA gate, rather than re-implemented per hook.

**No `.claude/commands/`** — reversal of the original D-1/T-4.4. The directory is
gate-pinned deleted (`test_skill_commands_aqa.py::test_legacy_commands_dir_stays_deleted`,
WS-F7a), and current Claude Code invokes `.claude/skills/<name>/SKILL.md` directly as
`/<name>`, so wrappers would be dead weight even without the gate.

Everything portable lives under `.claude/`; everything executable-and-tested lives under
`tools/`; the only repo-specific facts live in `workforce.yaml`. This keeps the shape
foundry-extractable without violating WS-F7b's claim on the settings/marketplace surface.

## D-2. Configuration (I-2)

`.claude/workforce.yaml`, validated by `tools/claude_hooks/config.py::WorkforceConfig`
(Pydantic v2, `extra="forbid"`, range-validated fields). Sole source for:

```yaml
freeze:
  feature_key: F-008              # the gate feature
  features_file: features.yaml    # repo-relative
  frozen_paths:                   # capability globs the gate protects (deliberate,
    - src/mousedroid/arm/**       #  reviewable config choice — not hook code)
  override_env: MOUSEDROID_WORKFORCE_ALLOW_FROZEN
secret_scan:
  command: gitleaks               # resolved via shutil.which at runtime
  config_file: .gitleaks.toml     # the repo's regex-only allowlist
  timeout_s: 20
  strict: false                   # absent binary: warn+allow; true flips to deny
coverage:
  tools_line_min: 85
  tools_branch_min_advisory: 0    # reported, never blocking (rev-B doctrine)
docs:
  core_max_lines: 250             # root CLAUDE.md budget after D-8
worktree:
  prefix: mdcw-                   # ../<prefix><change-id>
evidence:
  tracked_roots: [reports, smoke-reports]
  stale_after_days: 90
  local_only_declared:            # gitignored-by-policy families (see D-4 auditor)
    - reports/trunk_sync
    - reports/jetson_full_validation
    - reports/jetson_smoke
    - reports/dead_code
```

Hooks and the AQA module read **only** this file. Tests exercise unknown-key rejection and
range validation. Naming note: `scripts/check_settings_identity.py` validates Python module
import identity, not `.claude/settings.json` — the name collision is documented here so
nobody "fixes" the wrong thing.

## D-3. Hooks (mechanical governance)

| Hook | Event / matcher | Behavior |
|---|---|---|
| `secret_scan.py` | PreToolUse, `Write\|Edit\|MultiEdit\|NotebookEdit` | Reads the pending `tool_input.content` / `tool_input.new_string` from stdin JSON, scans it with the configured scanner (`gitleaks` + `.gitleaks.toml` — the same regex-only allowlist as CI, honored by regex, never by path). Findings → deny. Scanner binary absent → **warn+allow** by default (mirrors the shipped advisory posture); `secret_scan.strict: true` flips to deny. Closes WS-0.4's never-shipped edit-time gap |
| `freeze_gate.py` | PreToolUse, `Write\|Edit\|MultiEdit\|NotebookEdit` | Loads `freeze.features_file`, reads the `freeze.feature_key` entry. Target path matching `freeze.frozen_paths` is **denied while status ≠ `done`**, with the rev-B preemption rule quoted ("hardware readiness preempts all in-flight software streams"). `freeze.override_env` set → permitted-but-logged (the RUN-MOTION consent pattern). F-008 flipping to `done` self-disables the gate with zero code change. Catalog missing/unreadable/malformed → **fail-closed** (a broken governance input is itself a red flag) |
| `post_edit_check.py` | PostToolUse, `Write\|Edit\|MultiEdit` | `ruff check` + `mypy` on the touched file; report-only. Platform constraint, not a choice: PostToolUse fires after execution and cannot block |

Deny mechanics per platform docs: JSON `hookSpecificOutput.permissionDecision:
"deny"` carrying a reason. **Allow is silent** — emitting an explicit `allow` would
suppress the user's normal permission prompt, so a hook with no objection exits 0
writing nothing to stdout. Hooks **merge additively** across user/project scopes, so
adding this first project `hooks` block cannot shadow anyone's personal hooks.

Hook commands are `cd "$CLAUDE_PROJECT_DIR" && python3 -m tools.claude_hooks.<module>`,
with a per-hook `timeout`. Both halves are load-bearing and were verified by hand:
running the module *file* by path leaves the repository root off `sys.path`, so the
package import fails with `ModuleNotFoundError` on every edit. The AQA test pins the
invocation shape so a future "simplification" back to a bare path fails loudly.

All hook logging goes to **stderr** (`logging_setup.py`), because stdout is the
decision channel — a stray log line there would corrupt the payload. The module binds
its own structlog logger via `wrap_logger` rather than `structlog.configure()`, so
importing a hook never mutates the process-global structlog config the test suite pins.

Implementation hygiene: `tools/**` per-file-ignores are `["D", "ANN", "T20"]` — the `S`
family still applies, so subprocess calls use `shutil.which`-resolved absolute executables
and list args (S603/S607-clean), no shell. All three hooks are ordinary Python modules with
unit tests under `tests/unit/tools/claude_hooks/` (temp-dir fixtures; fake
`features.yaml` states: todo / in_progress / done / missing-key / malformed). The scanner
and the post-edit checkers are driven through synthesised stub executables rather than the
real binaries, so the exit-code contracts are covered deterministically on any host.

Exit-code contract:

| Exit | stdout | Meaning |
|---|---|---|
| 0 | *(empty)* | allow — no objection; the normal permission flow proceeds |
| 0 | deny payload | deny, with `permissionDecisionReason` carried in the JSON |
| non-zero | — | hook crashed (e.g. dependencies absent). Claude Code reports it and continues, so a broken hook never wedges the session; CI is the backstop |

## D-4. Subagent roster

Seven agents under `.claude/agents/`, each ≤60 lines, frontmatter limited to
platform-supported keys (`name`, `description`, `tools`, optional `model`) with **bare
tool names only** — reversal of the original exemplar's `Bash(...)` patterns, which the
platform does not support in agent frontmatter. Bash restraint is expressed in the prompt
body and enforced by permissions/hooks, not frontmatter. Collaboration default (I-6) lives
in CLAUDE.md's orchestration directive.

| Agent | Mandate |
|---|---|
| `openspec-author` | Scaffold/author change bundles in the house formats; extend the `openspec/project.md` registry; reserve F-numbers per ADR-013; run the repo-native validation checklist (no OpenSpec CLI exists here) |
| `test-engineer` | Tests-first; coverage delta ≥ gate on touched files; forbids mock/patch on hardware-tier test paths |
| `peer-reviewer` | Adversarial review; `[Certain]/[Likely]/[Guessing]` tags; cite by symbol, never line number; verify negative claims by search; severity-ordered findings; explicitly list what survives |
| `security-scanner` | Secret/credential sweep; `.gitleaks.toml` drift; permissions-allowlist review |
| `doc-reconciler` | CLAUDE.md / AGENTS.md / README / nested-CLAUDE.md drift detection; surfaces-index maintenance |
| `hw-evidence-auditor` | Hardware/perf claims must trace to a tracked artifact under `evidence.tracked_roots` **or** a declared local-only evidence chain (gitignored-by-policy family + CHANGELOG/plan-doc reference). Claims with neither are findings (the 2026-07-12 run is the canonical example). Staleness from `evidence.stale_after_days`. No BENCHMARKS.md reference — the file does not exist |
| `config-guardian` | Hunts hardcoded values and schema-bypass patterns; proposes `workforce.yaml`/Pydantic homes |

### Exemplar agent file (corrected; template for all seven)

```markdown
---
name: peer-reviewer
description: >
  Adversarial reviewer. Invoke proactively on any diff > 50 lines, any plan/spec
  document, and before any PR is opened. Verifies claims against the tree,
  tags confidence, and cites by symbol.
tools: Read, Grep, Glob, Bash
---

You are the adversarial peer reviewer for this repository.

Bash discipline: read-only invocations only (git diff, git log, pytest with
report-only flags). Never write, stage, commit, or mutate state.

Rules:
1. Verify before trusting: every factual claim in the artifact under review is
   checked against the working tree. Negative claims ("X does not exist") are
   verified by search, not assumed.
2. Confidence tags on every finding: [Certain] (verified), [Likely] (strong
   inference), [Guessing] (gap-filling). If most findings are [Guessing], say so
   first and stop reviewing — ask for the missing context instead.
3. Cite by symbol (`module.py::SYMBOL_NAME`), never by line number.
4. Severity-order findings; lead with the one the author least wants to hear.
5. Explicitly list what survives review unchanged — a review that only lists
   defects is not calibrated.
6. Never soften: no praise openers, no hedged verdicts.
```

## D-5. Skill lifecycle (unchanged vocabulary; verify, then extend)

Reversal of the original D-5: **no new lifecycle status, no supersession key** (the
draft's proposed additions are quoted and refuted in `peer-review.md`). The
vocabulary stays exactly `active|frozen|deferred`
(`tools/validate_skill_commands.py::_ALLOWED_SKILL_STATUSES`, pinned by unit + regression
tests). The existing three skills are already uniformly frozen with identical `unfreeze:`
conditions — they are **verify-only and byte-untouched** in this change
(`test_next_steps_reconciled.py` pins them).

New skills (frontmatter passes the existing validator at authoring time — every backtick
path must exist):

- `openspec-change` (active) — authoring workflow: scaffold the change dir, house delta
  format, registry-row + F-number reservation, repo-native validation checklist.
- `coverage-gate` (active) — runs the dedicated `--cov=tools/claude_hooks` invocation
  (line gate from `coverage.tools_line_min`) plus the advisory branch report; owns the
  README coverage-claim truth-fix and emits the PR delta table.
- `evidence-commit` (active) — places commit-safe artifacts under
  `reports/<surface>/<date>/`, links them from `features.yaml` notes; for
  gitignored-by-policy families it records the declared local-only chain instead; refuses
  to close a claim with neither.
- `worktree-flow` (active) — see D-6.
- `jetson-smoke` — tiered smoke discipline; documents **both** the RUN-MOTION typed
  consent phrase (operator practice, `smoke-reports/` precedent) **and** the mechanical
  `MOUSEDROID_SMOKE_ALLOW_MOTION` / `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION` gates.
  The Tier-3 (live actuation) section is `frozen` with
  `unfreeze: "typed RUN-MOTION consent + chassis raised + arm clear"`.

`.github/skills/jetson-hardware-debug/SKILL.md` (the fourth skill, currently unswept, two
IPv4 literals): handled as an explicit task (4.7). Recommended disposition:
placeholder-swap both IPs (pointing at `docs/runbooks/claude-code-on-jetson.md` for the
real values), then widen the AQA sweep to that directory. Fallback: declared exclusion
with the reason recorded in the AQA module docstring.

## D-6. Worktrees (parallel agent isolation)

One worktree per change-id: `git worktree add ../<worktree.prefix><change-id> -b <change-id>`.
Rules: agents never share a worktree with a human's dirty tree; merge back via PR only;
`git worktree list` is the session-preamble audit. Precedent already in-tree:
`scripts/check_config_compat.py::worktree_at_sha` (ephemeral detached worktrees) and the
`.dockerignore` "Claude worktrees + agent workspaces" entry. Documented in
`docs/runbooks/worktrees.md`; scaffolded by the `worktree-flow` skill.

## D-7. MCP selection (earn a place, not "use all")

Checked-in `.mcp.json` (project scope, secretless — the platform expands `${VAR}` /
`${VAR:-default}` in `command`/`args`/`env`/`url`/`headers`):

| Server | Status | Justification |
|---|---|---|
| `mousedroid` | **adopt** | The repo's own MCP server; stanza exactly per `docs/MCP_OPERATOR_GUIDE.md` ("Claude Code" section), plus `"MOUSEDROID_MOCK_HARDWARE": "${MOUSEDROID_MOCK_HARDWARE:-true}"` so a workstation session defaults motion-safe. Closes the open checkbox at `docs/MCP_NEXT_STEPS.md:51` |
| GitHub MCP | **adopt** | PR triage, issue ops, review comments; token via env expansion, never a literal |
| Grafana MCP | evaluate-first | Only if the observability workflow benefits; requires a reachable Grafana |
| Hugging Face MCP | evaluate-first | Model/dataset pulls; relevant only post-unfreeze |

Explicitly rejected for now: filesystem/memory MCPs (native tools suffice), browser MCPs
(no workflow needs one), everything else. Each server added is attack surface and context
cost. Evaluate-first notes are committed under `docs/claude/surfaces/`; the servers are not.

## D-8. Docs restructure (hybrid; docs-last)

Reversal of the original pointer-stub-only D-8, using a documented platform mechanism the
draft missed: **nested per-directory CLAUDE.md files auto-load when Claude works in those
directories.**

- Root `CLAUDE.md` keeps the nine evergreen sections, a corrected CI-pipeline section
  (the current one says "5 stages" against a 12-job ci.yml), the orchestration directive
  (I-6), the freeze rule, and a surface map — budget `docs.core_max_lines`.
- Subsystem-scoped surface sections move to nested CLAUDE.md files next to the code they
  govern (e.g. `src/mousedroid/telemetry/`, `src/mousedroid/learning/`,
  `src/mousedroid/growth/`, `src/mousedroid/world_model/`) — auto-loaded exactly when an
  agent works there, which pointer stubs cannot do.
- Cross-cutting surfaces (deployment/CI-gate contracts, full-validation flow, USB-C smoke)
  move to `docs/claude/surfaces/` behind an index; the root keeps one-line pointers.
- AGENTS.md is deduped on its declared axis (worker contract), keeping its recipes and
  delegating project facts to CLAUDE.md.

Constraint inventory (verified): only **negative** pins exist on these files —
`test_portfolio_reframe_aqa.py` requires the headline/forward docs to exist and to carry
none of the banned reframe tokens; no test pins any CLAUDE.md section or heading, and no
test reads AGENTS.md. The rev. B AQA extends the banned-token sweep to nested CLAUDE.md
files and `docs/claude/surfaces/**`, allowlisting the literal repository slug (its hyphen
boundary would otherwise false-positive the bare-token regex).

## D-9. Testing & CI (I-4)

House pattern — reversal of the original's new `workforce-lint` GitHub Actions job:

- `tests/regression/test_claude_workforce_aqa.py` — the PR gate, reusing the public
  helpers from `tools.validate_skill_commands` (`find_hardcoded_hosts`,
  `referenced_repo_paths`; precedent `test_foundry_plan_doc.py`). Asserts: workforce.yaml
  schema-valid; agent frontmatter contract (bare tool names — reject `(` / `*` tokens);
  portability (no IPv4/absolute paths under `.claude/`); `.mcp.json` parses, is
  secretless, and its `mousedroid` entry matches the operator guide; lifecycle contract
  on all skills; banned-token sweep over moved docs. Runs in the existing ci.yml `test`
  job matrix (3.10/3.11/3.12) for free — no new workflow, no `advisory_stages.yaml`
  entry, no `${{ }}`-in-`run:` risk.
- `scripts/ci.sh` gains stage `=== Claude Workforce Validation ===` (local mirror, same
  module) and a dedicated coverage invocation for the hook code:
  `pytest tests/unit/tools -m "not hardware" --cov=tools/claude_hooks --cov-branch
  --cov-fail-under=<coverage.tools_line_min-mirrored>` — the existing `src/mousedroid`
  gate cannot see `tools/` (verified), so this is a NEW invocation; branch numbers are
  reported, advisory-first. `tests/unit/test_scripts.py::TestCiSh` gets presence-substring
  pin **additions** for the new stage.
- `.github/workflows/ci.yml` lint step gains `tools/` (today only local `scripts/ci.sh`
  lints it — divergence closed).
- WS-F7b note: when the foundry validator replaces `validate_skill_commands.py`, the two
  pure helpers migrate to a neutral module (e.g. `tools/claude_hooks/textchecks.py`) and
  both importing tests re-point; behaviour pins stay.
