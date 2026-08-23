# Peer review — Documentation reconciliation across live governance surfaces

## Verdict table

| Claim | Verdict |
|---|---|
| `CLAUDE.md` / `docs/claude/surfaces/ci-gates.md` / `docs/claude/surfaces/README.md` claimed a "12-job" pipeline naming six jobs that don't exist (`secret-scan`, `skills`, `test-fast`, `validate`, `regression`, `package`) | **CONFIRMED** — `git diff HEAD -- CLAUDE.md` shows the pre-fix heading literally read `## CI/CD Pipeline (12 Jobs)` and named exactly those six step-not-job names across its old Stage 0-4 breakdown |
| The real pipeline defines 16 top-level jobs | **CONFIRMED** — `grep -nE "^    [a-zA-Z0-9_-]+:$" .github/workflows/ci.yml` under `jobs:` lists exactly 16 keys (`actionlint`, `lint`, `config-validate`, `usbc-config-gate`, `typecheck`, `test`, `performance`, `test-windows`, `local-gates`, `prometheus-check`, `vla-extras`, `onnx-world-model-extras`, `gitleaks`, `vulture-audit`, `security`, `docker`); `test_ci_job_count_docs_match_the_real_workflow`'s `_real_ci_job_count()` counts the same `jobs:` mapping via `yaml.safe_load` |
| `docs/claude/surfaces/ci-gates.md`'s 5 advisory jobs and their `since`/`promote_after_days` are real | **CONFIRMED** — all 5 (`performance`, `test-windows`, `onnx-world-model-extras`, `vulture-audit`, `security`) and every stated date/window matches `.github/advisory_stages.yaml` verbatim |
| `src/mousedroid/orchestrator/CLAUDE.md`'s old symbols (`RobotOrchestrator`, `state.py`, `factory.py:_build_orchestrator`, `ConstitutionalSafetyMonitor`) don't exist | **CONFIRMED** — zero matches for any of the four across every `.py` file in the tree |
| `src/mousedroid/orchestrator/CLAUDE.md`'s new symbols exist | **CONFIRMED** — `MouseDroidOrchestrator` (`src/mousedroid/orchestrator/orchestrator.py:84`), `build_orchestrator` (`src/mousedroid/factory.py:4016`), `AutonomousOrchestrator` (`src/mousedroid/orchestrator/autonomous.py:27`), `MouseDroidSafetyMonitor` (`src/mousedroid/safety/monitor.py:25`), `self._esp32.emergency_stop()` (multiple call sites, e.g. `src/mousedroid/orchestrator/orchestrator.py:640`) |
| `AutonomousOrchestrator` has zero production callers | **CONFIRMED** — `build_autonomous_orchestrator` is defined at `src/mousedroid/factory.py:4850` and called nowhere else under `src/`, `scripts/`, or `tools/` |
| `HARNESS_SPEC.md`/`tests/agent.md`/`docs/architecture/c4-spec-harness.md`'s coverage floor is now 90%, with 85% correctly scoped to `tools/claude_hooks` where it still appears | **CONFIRMED** — `pyproject.toml`'s `[tool.coverage.report] fail_under = 90`; `tools/claude_hooks/config.py`'s `CoverageConfig.tools_line_min` defaults to 85; all 5 site-hits across the 3 docs now say 90% for `src/mousedroid`, and the 2 of those 5 that also mention 85% (`HARNESS_SPEC.md:303`, `tests/agent.md:6`) qualify it as `tools/claude_hooks/` on the same line |
| `docs/CHARTER.md`'s `growth/` correction matches `src/mousedroid/factory.py` | **CONFIRMED** — `src/mousedroid/factory.py:4308` calls `build_growth_coordinator(cfg, metrics=metrics_registry, vla_policy=vla_policy, world_model=wm)`; `meta/`/`scaling/` genuinely have no `build_meta_*`/`build_scaling_*` call site (grep: none) |
| The `growth/` correction is complete across every live doc that repeated the claim, not just `docs/CHARTER.md` §5 | **CONFIRMED** — `git diff HEAD -- docs/CHARTER.md` shows §1 and §3 also corrected (three sections total, not one); `git diff HEAD -- README.md` shows a new "Factory-instantiated, default-OFF pending a soak decision" bucket holding `growth/` alone, with the "not yet wired" bucket's `tests/unit/{meta,growth,scaling}/` citation narrowed to `tests/unit/{meta,scaling}/`; `git diff HEAD -- docs/architecture.md` shows the Level 3d same-shape comparison narrowed to `meta`/`scaling` with an explanatory clause; `git diff HEAD -- NEXT_STEPS.md` shows item 0b's repeat citation corrected identically. Five files, one conceptual claim, verified individually |
| `NEXT_STEPS.md`'s Phase 5 correction matches the tree | **CONFIRMED** — `src/mousedroid/sim/mujoco_rover_env.py::RoverMuJoCoEnv` exists and self-documents as "Conforms structurally to ... `RoverEnvProtocol`"; `docs/CHARTER.md`'s M5 line already read "✅" before this sprint touched either file, so the two docs previously disagreed with each other, not just with the tree |
| `SKILLS.md`'s skill-directory index is now complete | **CONFIRMED** — 19 real `.claude/skills/*/` directories; 4 already carry `###` entries (`test-tier-mirror`, `prove-pin-fails`, `regression-pair-scaffold`, `feature-closeout`) + 15 in the new Workforce-skills table = 19/19 accounted for |
| `SKILLS.md`'s Subagent skills table names only real agents, 1:1 | **CONFIRMED** — table names exactly `config-guardian`, `doc-reconciler`, `hw-evidence-auditor`, `openspec-author`, `peer-reviewer`, `security-scanner`, `test-engineer` — the 7 real `.claude/agents/*.md` stems, no extras, no omissions |
| `test_every_skill_directory_is_mentioned_in_the_index` and `test_every_agent_is_listed_in_the_subagent_skills_table` exist and pass | **CONFIRMED** — both present in `tests/regression/test_claude_workforce_aqa.py`; `bash scripts/validations/F-030.sh` runs both and exits 0 |
| `tests/regression/test_doc_reconciliation_aqa.py`'s three pins exist and pass | **CONFIRMED** — `tests/regression/test_doc_reconciliation_aqa.py` collected 3 tests, all passing as part of the 5-passed run below |
| `_DOC_GLOBS` → `_tracked_docs()` closes a real 15-file gap | **CONFIRMED** — `git diff HEAD -- tests/regression/test_ci_gate_wiring_aqa.py` shows the literal removal of the 5-pattern glob tuple, replaced with a `git ls-files -- "*.md"` subprocess call; the new docstring names `tests/agent.md` as one of the 15 previously-missed files, and that file is independently confirmed as one of this bundle's own Phase 1 targets |
| `scripts/validations/F-030.sh` exists, is executable, and passes | **CONFIRMED** — `bash scripts/validations/F-030.sh` → "F-030 OK: CI job count, coverage floors, and the skills/agents index all match the tree" (5 passed in 0.46s); `python scripts/validate.py --check F-030` → "F-030: OK" |
| `features.yaml`'s F-030 entry parses and validates | **CONFIRMED** — `yaml.safe_load` succeeds and `jsonschema.validate(...)` against `features.schema.json` raises no error; `python scripts/validate.py --tier fast` → "OK: 22 done; ran 20 for tier(s) ['fast'], skipped 2 (other tiers)" |

## Corrected-design map

| Original intent | Corrected |
|---|---|
| "Reuse the existing `dev-governance` capability for the spec delta" | Rejected — `dev-governance` (from `mouse-droid-claude-workforce`) is scoped to the workforce tooling's own claims (evidence chains, `.claude/workforce.yaml`, `tools/claude_hooks` coverage); F-030's drifts span CI job identity, orchestrator symbols, and a cognitive-pillar wiring narrative that are not workforce-tooling claims. A new `docs-governance` capability keeps the two verification surfaces distinct (D-1). |
| "Close the doc-inventory gap by adding `tests/**/*.md` to `_DOC_GLOBS`" | Rejected — a sixth hardcoded pattern only pushes the next gap out to whichever directory is missed next, which is exactly how `tests/agent.md` was missed the first time. Replaced the roster with `git ls-files`-sourced discovery (D-4). |
| "Assert the CI job count with a bare `\d+ jobs?` regex" | Rejected — matches true statements like "5 jobs run *(advisory)*" (the advisory subset, not the total). Narrowed to the two phrasings this repo actually uses for a TOTAL: `"(N Jobs)"` and `"N-job"` (D-3). |
| "Search the whole file for '85%' to catch the stale coverage claim" | Rejected — would also flag this sprint's own correctly-qualified "85% for `tools/claude_hooks/`" text. Narrowed to a per-line check with a same-line qualifier (D-3). |
| "The deferred Hailo mock-shapes item sits under a `hardware/` directory prefix that `scripts/check_no_hardcoded_values.py` exempts" | **REFUTED as originally framed, while authoring this bundle** — reading `scripts/check_no_hardcoded_values.py` directly shows `ALLOWED_DIR_PREFIXES` names exactly four prefixes (`src/mousedroid/config/schema/`, `src/mousedroid/telemetry/metrics/`, `src/mousedroid/telemetry/server/`, `src/mousedroid/validation/runtime/`); no `hardware/` entry exists. The deferral still stands, but on the correct ground: the gate only inspects changed/added lines in a diff, and `src/mousedroid/hardware/accelerator/hailo_runtime.py` is untouched by any current diff (confirmed clean via `git status`/`git diff` against HEAD), not exempt by directory. Corrected in `tasks.md`'s deferred-items entry rather than left standing. |
| "The deferred `scripts/prove_pin_fails.sh` item has 6 passing tests" | **REFUTED as originally framed, while authoring this bundle** — `python -m pytest tests/unit/scripts/test_prove_pin_fails.py -q` collects and passes 10 tests, not 6. The deferral still stands (dev/CI-only tool, clear-error failure mode); only the test count in the stated rationale was wrong. Corrected in `tasks.md`'s deferred-items entry. |

## What survives review unchanged

- All six drift fixes stay pure documentation edits — zero `.py` files under
  `src/mousedroid/` touched by this change (confirmed via `git diff --stat`; the four
  `src/mousedroid/*.py` files showing modified in the working tree —
  `src/mousedroid/orchestrator/orchestrator.py`, `src/mousedroid/comms/_utils.py`,
  `src/mousedroid/config/schema/misc.py`, `src/mousedroid/validation/runtime/_storage.py`
  — belong to unrelated in-flight work on this same branch, e.g.
  `src/mousedroid/orchestrator/orchestrator.py`'s single-line diff threads an unrelated
  `vision_feature_max_samples` telemetry field).
- Reuse of the existing `tests/regression/test_claude_workforce_aqa.py` and
  `tests/regression/test_ci_gate_wiring_aqa.py` modules for the catalog pins and the
  doc-inventory fix, rather than scattering new files across the regression tier for
  closely-related concerns.
- Discovery-based assertions throughout (glob the real directories/agents, `git ls-files`
  for docs) over hardcoded expected-lists, so the next skill, agent, or doc ships governed
  rather than silently exempt.
- The `growth/` fix (task 1.6) turning out to be incomplete on first pass, and getting caught
  *within this same bundle* (task 3.5, D-6) rather than in a later round, is treated here as
  evidence the change's own thesis holds under its own weight — not as a defect to paper over.
  The original scope statement (`features.yaml`'s F-030 description) names only the
  `docs/CHARTER.md` §5 site; this bundle's prose was updated to the fuller, five-file scope
  once found, rather than left matching the narrower original framing.

## Load-bearing pins any implementation must satisfy

1. `CLAUDE.md`, `docs/claude/surfaces/ci-gates.md`, and `docs/claude/surfaces/README.md`
   state the total CI job count as 16, never 12.
2. No live doc states an unqualified 85% floor for the `src/mousedroid` coverage gate — 90%
   is the real gate; 85% is real only for `tools/claude_hooks`, and that separate floor
   stays pinned at 85% so a future edit cannot silently tighten it to match.
3. Every real `.claude/skills/*/` directory is mentioned somewhere in `SKILLS.md`; every
   real `.claude/agents/*.md` stem appears in its Subagent skills table, and no others do.
4. `bash scripts/validations/F-030.sh` exits 0.

## Appendix — observed during verification, out of scope for this bundle

`AGENTS.md`'s "Red flags" section (around line 164) still reads "the security-auditor
subagent will flag it". This is generic prose invoking a security-review role, not an
enumerated roster claim — unlike the old Subagent skills table, which explicitly listed all
seven names and got every one wrong, this single mention does not present itself as an
exhaustive or authoritative list. It is not pinned by either new AQA test, and `SKILLS.md`'s
own Subagent skills table (the artifact F-030's verification list actually names) is
independently confirmed correct above. Noted here per this session's "verify before
narrating" discipline rather than passed over silently; left uncorrected because it sits
outside F-030's stated verification scope, and editing prose beyond that scope was out of
bounds for this bundle-authoring pass.

## Appendix — deferred items

Three items are recorded, with full stated rationale, in `tasks.md`'s "Explicitly deferred"
section: the Hailo mock output-shape constants
(`src/mousedroid/hardware/accelerator/hailo_runtime.py`), `scripts/prove_pin_fails.sh`'s
unquoted `--paths`/`--tests` word-splitting, and a `shellcheck` post-edit hook. None are
pinned or asserted by this bundle. The first two items' stated rationale each required a
correction during this review pass — see the two refuted rows in the Corrected-design map
above; both deferral decisions are unchanged, only the reasons/numbers originally given were
wrong (a nonexistent `hardware/` directory exemption; a test count of 6 where the real count
is 10).
