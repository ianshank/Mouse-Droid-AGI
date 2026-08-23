# Proposal — Documentation reconciliation across live governance surfaces

- change_id: mouse-droid-doc-reconciliation
- project: mouse-droid
- status: in progress
- feature_id: F-030
- epic: Quality Gates
- owner: ianshank
- created: 2026-08-23
- basis_commit: 74a4c11
- rev: A

## Why

`CLAUDE.md`, `src/mousedroid/orchestrator/CLAUDE.md`, `HARNESS_SPEC.md`, and the other
surfaces this change touches are not background reading — they are the ground truth an
autonomous Claude Code workforce reads before every tool call (CLAUDE.md's own
Collaboration Directive tells agents to "act proactively as lead developer, architect, and
SQE"). A stale claim in one of them does not just mislead a human skimming docs; it steers
the next agent's tool calls toward a class, file, or job that does not exist. Six such
drifts were verified against the tree:

1. `CLAUDE.md`, `docs/claude/surfaces/ci-gates.md`, and `docs/claude/surfaces/README.md`
   all stated a "12-job" CI pipeline naming six jobs — `secret-scan`, `skills`,
   `test-fast`, `validate`, `regression`, `package` — that are not job names in
   `.github/workflows/ci.yml` at all, only step names nested inside other jobs. The live
   file defines 16 real top-level jobs (`actionlint`, `lint`, `config-validate`,
   `usbc-config-gate`, `typecheck`, `test`, `performance`, `test-windows`, `local-gates`,
   `prometheus-check`, `vla-extras`, `onnx-world-model-extras`, `gitleaks`,
   `vulture-audit`, `security`, `docker`), 5 of which carry `continue-on-error: true` and
   are tracked in `.github/advisory_stages.yaml`.
2. `src/mousedroid/orchestrator/CLAUDE.md` described a `RobotOrchestrator` class, a
   `state.py` file, a private `factory.py:_build_orchestrator` helper, and a
   `ConstitutionalSafetyMonitor` — none exist in any `.py` file in the tree. The real
   production path is `src/mousedroid/orchestrator/orchestrator.py::MouseDroidOrchestrator`,
   wired by the public `src/mousedroid/factory.py::build_orchestrator`, with
   `src/mousedroid/safety/monitor.py::MouseDroidSafetyMonitor` as the safety filter and
   `self._esp32.emergency_stop()` as the real e-stop call.
   `src/mousedroid/orchestrator/autonomous.py::AutonomousOrchestrator` exists but its
   builder, `build_autonomous_orchestrator`, has zero callers anywhere under `src/`,
   `scripts/`, or `tools/` — a live alternate loop, not the production one.
3. `HARNESS_SPEC.md` (three sites), `tests/agent.md`, and
   `docs/architecture/c4-spec-harness.md` stated an 85% coverage floor for
   `src/mousedroid`; `pyproject.toml`'s `[tool.coverage.report] fail_under` is 90. 85% is
   correct, but only for the separate, narrower `tools/claude_hooks` gate
   (`tools/claude_hooks/config.py`'s `CoverageConfig.tools_line_min` defaults to 85) — the
   two sites that mentioned 85% at all left that scoping unstated on the line.
4. `docs/CHARTER.md` §5 said `growth/` was "not yet instantiated by `factory.py`" in the
   same breath as `meta/` and `scaling/`. `src/mousedroid/factory.py:4308` calls
   `build_growth_coordinator(cfg, metrics=metrics_registry, vla_policy=vla_policy,
   world_model=wm)` — default-OFF behind `Settings.growth: GrowthConfig | None` and wired
   to the shared metrics registry, the same posture already correctly documented for M6.
   `meta/` and `scaling/` genuinely have no `build_meta_*`/`build_scaling_*` call site
   (verified — none), so only the `growth/` clause was false. **The same stale claim turned
   out to be live in four more places**, found by applying this bundle's own
   `narrative-correction-sweep` procedure to its own §5 fix rather than trusting it
   generalized: `docs/CHARTER.md` §1 and §3 both still called `growth/` "not yet wired",
   contradicting the just-corrected §5 in the same file; `README.md`'s cognitive-stack table
   had `growth/` in the "not yet wired" bucket outright; `docs/architecture.md`'s Level 3d
   section cited it as a same-shape precedent for genuinely-unwired GCP components; and
   `NEXT_STEPS.md` item 0b repeated that same citation. All five were corrected together —
   `README.md` gained a third table bucket ("Factory-instantiated, default-OFF pending a soak
   decision") rather than forcing `growth/` into either the fully-wired or the
   not-yet-wired bucket, neither of which was accurate.
5. Root `NEXT_STEPS.md` called Phase 5 (the MuJoCo physics simulator) "(stretch) ...
   deferred until Phase-3b 30-day soak completes", while `docs/CHARTER.md`'s own M5 entry
   already read "✅" before this sprint touched either file, and
   `src/mousedroid/sim/mujoco_rover_env.py::RoverMuJoCoEnv` is a real,
   `RoverEnvProtocol`-conformant implementation that replaced the NumPy kinematic sim. The
   two docs disagreed with each other and with the tree.
6. `SKILLS.md`'s "Subagent skills" table named seven plugin-namespaced subagents
   (`security-auditor`, `feature-dev:code-reviewer`, ...) matching none of the seven real
   `.claude/agents/*.md` files, and 12 of 17 real `.claude/skills/` directories were
   entirely absent from `SKILLS.md`'s index.

Fixing the prose once is not durable — this exact class of drift (a claim corrected in the
file an engineer remembers, left standing in the ones they don't) is why
`.claude/skills/narrative-correction-sweep/SKILL.md` exists, and why this change adds two
catalog-wide pins rather than six one-off edits.

## What Changes

- `CLAUDE.md`'s CI/CD Pipeline section rewritten from the fictional 12-job/6-fictional-name
  breakdown to the real 16-job breakdown sourced from `.github/workflows/ci.yml`'s own
  `# Stage N` comments; its `docs/claude/surfaces/ci-gates.md` cross-reference updated
  "12-job" → "16-job".
- `docs/claude/surfaces/ci-gates.md` fully rewritten: all 16 real jobs listed by real stage
  number, plus the 5 real advisory jobs with their real `since`/`promote_after_days` sourced
  from `.github/advisory_stages.yaml`.
- `docs/claude/surfaces/README.md`'s cross-reference line: "12-job" → "16-job".
- `src/mousedroid/orchestrator/CLAUDE.md` fully rewritten to name only symbols that exist:
  `src/mousedroid/orchestrator/orchestrator.py::MouseDroidOrchestrator`,
  `src/mousedroid/factory.py::build_orchestrator`,
  `src/mousedroid/orchestrator/autonomous.py::AutonomousOrchestrator` (explicitly flagged
  zero production callers), and `src/mousedroid/safety/monitor.py::MouseDroidSafetyMonitor`.
- `HARNESS_SPEC.md` (3 sites), `tests/agent.md`, `docs/architecture/c4-spec-harness.md`:
  coverage floor corrected to 90% for `src/mousedroid`, with the two sites that mention 85%
  at all now scoping it explicitly to `tools/claude_hooks/`.
- `docs/CHARTER.md` §1, §3, and §5: every `growth/` clause split out of the `meta/`/`scaling/`
  "not yet wired" wording into its own, correctly-cited statement (three sections, found by
  re-sweeping the file after the initial §5 fix rather than trusting it generalized).
- `README.md`'s cognitive-stack table gains a third bucket, "Factory-instantiated, default-OFF
  pending a soak decision", holding `growth/` alone; the "Implemented and unit-tested — not
  yet wired into the loop" bucket now holds only `meta/` and `scaling/`.
- `docs/architecture.md`'s Level 3d component diagram: the "same shape as unwired
  `meta`/`growth`/`scaling`" comparison for the three unwired GCP components corrected to
  `meta`/`scaling` only, with a note that `growth` has since gained a factory builder.
- `NEXT_STEPS.md`: the Phase 5 line corrected from "(stretch) ... deferred" to "✅ landed",
  citing `RoverMuJoCoEnv` and cross-referencing `docs/CHARTER.md`'s M5, with a one-sentence
  note on why the stale wording existed (written before the simulator landed, never
  revisited); item 0b's separate `growth/` same-shape citation corrected to match the
  Level 3d fix above.
- `SKILLS.md` gains a "## Workforce skills (invocable, not covered above)" section (15 real
  `.claude/skills/` directories not already covered by an existing `###` entry) and its
  "## Subagent skills (delegation-facing)" table is rewritten from 7 fictional names to the
  real 7 `.claude/agents/` names.
- `AGENTS.md`'s subagent-dispatch line corrected to name real agents, and its `.gitignore`
  negation rule clarified (the directory-level negations on `.claude/skills/` and
  `.claude/agents/` already cascade to new files inside them; only genuinely new top-level
  asset categories need a new negation line).
- Two new pins in `tests/regression/test_claude_workforce_aqa.py`:
  `test_every_skill_directory_is_mentioned_in_the_index` and
  `test_every_agent_is_listed_in_the_subagent_skills_table`, both discovery-based (glob the
  real directories/files) rather than a hardcoded expected-list.
- New `tests/regression/test_doc_reconciliation_aqa.py`: pins the CI-job-count,
  coverage-floor, and orchestrator-symbol claims against their source of truth
  (`.github/workflows/ci.yml`'s parsed `jobs:` mapping, `pyproject.toml`'s `fail_under`,
  `tools/claude_hooks/config.py`'s `tools_line_min`, and the four real
  `orchestrator`/`safety`/`factory` symbols `src/mousedroid/orchestrator/CLAUDE.md` names)
  rather than against prose alone. The coverage-floor check compares an extracted claim
  against `_real_src_coverage_floor()` directly rather than searching for the one stale
  value ("85%") it was first written against, so a future floor change some doc misses is
  still caught. The sixth original drift (the `growth/`-wiring claim) is **not** pinned
  here -- a sentence-scoped sweep was attempted and produced false positives on legitimate
  text; see `tasks.md`'s "Explicitly deferred" section for the full account, not silently
  omitted from this description.
- New `scripts/validations/F-030.sh`, the feature's `validation_command`.
- Riding the same sprint: `tests/regression/test_ci_gate_wiring_aqa.py`'s `_DOC_GLOBS` (a
  5-pattern directory roster that silently missed 15 git-tracked `.md` files, including
  `tests/agent.md`) replaced with a `git ls-files -- "*.md"`-sourced `_tracked_docs()`. New
  skill `.claude/skills/narrative-correction-sweep/SKILL.md` documents the whole-file,
  whitespace-normalized sweep procedure this defect class needs, and is registered in
  `SKILLS.md`'s Workforce-skills table.

## Impact

No production runtime behavior changes. Zero `.py` files under `src/mousedroid/` change as
part of this bundle; the one `src/mousedroid/` path it touches,
`src/mousedroid/orchestrator/CLAUDE.md`, is a documentation surface, not source code. (The
working tree also carries unrelated, in-flight diffs to
`src/mousedroid/orchestrator/orchestrator.py`, `src/mousedroid/comms/_utils.py`,
`src/mousedroid/config/schema/misc.py`, and `src/mousedroid/validation/runtime/_storage.py`
from other work on this same branch; none of them are part of this change.)

## Spec Deltas

`openspec/changes/mouse-droid-doc-reconciliation/specs/docs-governance/spec.md` — two ADDED
requirements.

## Tasks

See `tasks.md`.

## Validation

`bash scripts/validations/F-030.sh` — runs `tests/regression/test_doc_reconciliation_aqa.py`
plus the two new `tests/regression/test_claude_workforce_aqa.py` node IDs
(`test_every_skill_directory_is_mentioned_in_the_index`,
`test_every_agent_is_listed_in_the_subagent_skills_table`); exits 0 with 5 tests passing.
Also `python scripts/validate.py --check F-030` and `python scripts/validate.py --tier fast`.
