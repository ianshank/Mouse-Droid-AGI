# Tasks — Documentation reconciliation across live governance surfaces

Quality gate, run before a task is ticked:

```shell
python -m ruff check src/ tests/ tools/ && python -m ruff format --check src/ tests/ tools/
python -m mypy src/ --strict --ignore-missing-imports
```

`bash scripts/validations/F-030.sh` joins this gate from task 4.2 onward, once
the script it runs exists — tasks 1.1 through 4.1 land against the first two
commands only, since the script that would be the third cannot run before it
is written.

Task ordering is binding: each task lands green before the next starts.
Deviations from task wording are recorded inline — declared, not silent.

**Phase 1 — Fix the six verified drifts**

- [x] 1.1 `CLAUDE.md`'s CI/CD Pipeline section: "12-job" (naming six non-existent jobs) →
      "16-job", sourced from `.github/workflows/ci.yml`'s own `# Stage N` comments; its
      `docs/claude/surfaces/ci-gates.md` cross-reference line "12-job" → "16-job".
- [x] 1.2 `docs/claude/surfaces/ci-gates.md` fully rewritten: all 16 real jobs by real stage
      number, the 5 real advisory jobs with `since`/`promote_after_days` sourced from
      `.github/advisory_stages.yaml`.
- [x] 1.3 `docs/claude/surfaces/README.md`'s cross-reference: "12-job" → "16-job".
- [x] 1.4 `src/mousedroid/orchestrator/CLAUDE.md` fully rewritten off the four non-existent
      symbols (`RobotOrchestrator`, `state.py`, `factory.py:_build_orchestrator`,
      `ConstitutionalSafetyMonitor`) onto the four real ones
      (`src/mousedroid/orchestrator/orchestrator.py::MouseDroidOrchestrator`,
      `src/mousedroid/factory.py::build_orchestrator`,
      `src/mousedroid/orchestrator/autonomous.py::AutonomousOrchestrator` flagged zero
      production callers, `src/mousedroid/safety/monitor.py::MouseDroidSafetyMonitor`).
- [x] 1.5 `HARNESS_SPEC.md` (3 sites), `tests/agent.md`, `docs/architecture/c4-spec-harness.md`:
      "85%" → "90%" for the `src/mousedroid` gate; the two sites that mention 85% at all now
      scope it explicitly to `tools/claude_hooks/` on the same line.
- [x] 1.6 `docs/CHARTER.md` §5: `growth/` split out of the `meta/`/`scaling/` "not yet
      instantiated" clause into its own correctly-cited sentence
      (`src/mousedroid/factory.py::build_growth_coordinator`, default-OFF, metrics-wired,
      M6 posture). This was the seed fix, not the complete scope — see 3.5.
- [x] 1.7 `NEXT_STEPS.md`: Phase 5 line corrected from "(stretch) ... deferred until
      Phase-3b 30-day soak" to "✅ landed", citing `RoverMuJoCoEnv` and cross-referencing
      `docs/CHARTER.md`'s M5, with a one-sentence note on why the stale wording existed
      (written before the simulator landed, never revisited).

**Phase 2 — Reconcile the skills/agents catalog and pin it**

- [x] 2.1 `SKILLS.md` gains "## Workforce skills (invocable, not covered above)" (15 real
      `.claude/skills/` directories not already covered by an `###` heading) and rewrites
      "## Subagent skills (delegation-facing)" from 7 fictional plugin-namespaced names to
      the real 7 `.claude/agents/` names.
- [x] 2.2 `AGENTS.md`'s subagent-dispatch line corrected to name real agents (`peer-reviewer`,
      `test-engineer`, `config-guardian`, "or any of the other five"); its `.gitignore`
      negation rule clarified for the two already-cascading directory-level negations.
- [x] 2.3 Add `test_every_skill_directory_is_mentioned_in_the_index` and
      `test_every_agent_is_listed_in_the_subagent_skills_table` to
      `tests/regression/test_claude_workforce_aqa.py`, both discovery-based (glob
      `.claude/skills/`/`.claude/agents/`, not a hardcoded roster).
- [x] 2.4 Prove both pins fail: revert `SKILLS.md` to the pre-fix wording (a dropped skill
      mention; the old fictional agent table), confirm red, restore.

**Phase 3 — Pin the two numeric claims and close the doc-inventory gap**

- [x] 3.1 Add `tests/regression/test_doc_reconciliation_aqa.py`:
      `test_ci_job_count_docs_match_the_real_workflow` (parses
      `.github/workflows/ci.yml`'s `jobs:` mapping via `yaml.safe_load`),
      `test_no_live_doc_claims_a_stale_src_coverage_floor` (reads
      `pyproject.toml`'s `fail_under`, line-scoped so a correctly-qualified "85% for
      `tools/claude_hooks/`" mention does not false-positive), and
      `test_tools_claude_hooks_coverage_floor_is_still_85` (reads
      `tools/claude_hooks/config.py`'s `CoverageConfig.tools_line_min`, so a future "fix"
      that tightens it to 90% is itself caught).
- [x] 3.2 Tune both new regexes against the live tree before trusting them — the job-count
      pattern initially needed narrowing to avoid matching "5 jobs run *(advisory)*" (a true
      statement about the advisory subset, not the total), and the coverage regex needed the
      `claude_hooks` line-level qualifier to avoid flagging this sprint's own
      correctly-scoped "85% for `tools/claude_hooks/`" text. Recorded as the worked example
      in `.claude/skills/narrative-correction-sweep/SKILL.md`'s Guardrails section.
- [x] 3.3 Replace `tests/regression/test_ci_gate_wiring_aqa.py`'s `_DOC_GLOBS` (a 5-pattern
      directory-prefix roster that silently missed 15 git-tracked `.md` files, including
      `tests/agent.md` — one of this same sprint's own Phase 1 targets) with a
      `git ls-files -- "*.md"`-sourced `_tracked_docs()`. Deviation: this closes the gap
      generically for the existing `TestOrphanTierNarrativeAccuracy` pin (owned by the
      F-028/F-029 review, not newly authored here) rather than adding a new pin of its own —
      declared here because it touches a file this change does not otherwise own.
- [x] 3.4 Author `.claude/skills/narrative-correction-sweep/SKILL.md` documenting the
      whole-file, whitespace-normalized, `git ls-files`-sourced sweep procedure used in 3.3
      (and, per its own "Why this exists" section, in an earlier F-028/F-029 review round);
      register it in `SKILLS.md`'s Workforce-skills table; confirm clean via
      `python tools/validate_skill_commands.py`.
- [x] 3.5 Self-caught follow-on: apply 3.4's procedure to task 1.6's own `growth/` fix rather
      than trusting the single `docs/CHARTER.md` §5 edit generalized. Found the identical
      stale claim still live in four more places — `docs/CHARTER.md` §1 and §3 (both
      contradicting the already-corrected §5), `README.md`'s cognitive-stack table (`growth/`
      in the wrong bucket), and `docs/architecture.md`'s Level 3d component diagram (cited as
      a same-shape precedent for genuinely-unwired GCP components) — plus one repeat citation
      in `NEXT_STEPS.md` item 0b. All five corrected together; `README.md` gained a third
      table bucket ("Factory-instantiated, default-OFF pending a soak decision") rather than
      forcing `growth/` into either the fully-wired or the not-yet-wired bucket. See D-6.
- [x] 3.6 CodeRabbit review round on PR #202 (10 findings, all verified against the tree
      before fixing): added `test_orchestrator_claude_md_names_only_real_symbols` to close
      the "F-030 does not execute all declared governance checks" gap for the orchestrator-
      symbol half of that finding (the `growth/`-claim half is recorded under "Explicitly
      deferred" below, not silently dropped); fixed `check=False` → `check=True` in
      `_tracked_docs()` (a failed `git ls-files` was silently reading as "no docs found",
      making the whole sweep vacuously pass); bounded `test_every_agent_is_listed_in_the_-
      subagent_skills_table`'s `table_text` slice to the next `## ` heading instead of
      end-of-file; replaced `test_no_live_doc_claims_a_stale_src_coverage_floor`'s hardcoded
      `85%` search with a claim extracted and compared against `_real_src_coverage_floor()`
      directly, matching `_TOTAL_JOB_COUNT`'s existing discipline; `README.md`'s Growth &
      Distillation description corrected from "KL+CE" to the actually-wired regression
      (MSE) objective (`factory.py::build_growth_coordinator` passes
      `objective="regression"`); `NEXT_STEPS.md`'s roadmap intro still said "Phases 5 and 6
      are deferred" one paragraph above the already-corrected Phase 5 entry -- the same
      instance-not-class miss task 3.5 corrects elsewhere in this bundle, found by a bot
      instead of the sweep this time; `scripts/archive_stale_branches.sh`'s `grep -vx`
      matched `$REMOTE` as a regex, not a fixed string (`-F` added at both sites, plus a
      one-line comment); `.claude/skills/pin-reachability-audit/SKILL.md`'s all-features
      audit read local refs without fetching first; `tasks.md`'s own quality-gate fence had
      no language tag and claimed a gate no early task could satisfy (`bash
      scripts/validations/F-030.sh` did not exist until task 4.2 -- fixed to say so).

**Phase 4 — Catalog and validate**

- [x] 4.1 Confirm the F-030 entry in `features.yaml` (already present from planning) still
      validates against `features.schema.json`.
- [x] 4.2 Write `scripts/validations/F-030.sh`, mark it executable, confirm
      `bash scripts/validations/F-030.sh` exits 0.
- [x] 4.3 Run `python scripts/validate.py --check F-030` and
      `python scripts/validate.py --tier fast`.
- [x] 4.4 Register this change in `openspec/project.md`'s `## Changes` table.

## Explicitly deferred (separate changes, do not fold in)

- **A sentence-scoped regression pin for the `growth/`-wiring claim (CodeRabbit finding,
  PR #202) was attempted and abandoned.** The first three F-030 pins (CI job count,
  coverage floor, orchestrator symbols) each protect a claim with one consistent, checkable
  shape. The `growth/`-wiring claim does not: across the five files this bundle corrected it
  in, the same idea was phrased five different ways ("not yet instantiated", "not yet
  wired", "implemented-but-not-wired", "unwired", enumerated three-item lists) at distances
  from 20 to over 300 characters from the word `growth`. A sentence-scoped sweep (splitting
  each of the four live target files on `.` boundaries, matching the established
  `_ONLY_COVERAGE_CLAIM`/`_CLAIM_QUALIFIERS` idiom) was written and tested directly against
  the tree before being trusted, per this same bundle's own `narrative-correction-sweep`
  discipline -- and it produced two false positives on the first real run: a `README.md`
  code-fence file-tree listing swallowed into one giant "sentence" by the period-boundary
  splitter, and a legitimate, pre-existing `NEXT_STEPS.md` sentence (from Session 009's
  F-022 closeout, predating this bundle) that correctly describes `growth/` as wired but
  uses none of the pin's qualifier phrases. Both are real engineering problems with the
  sweep, not tuning-away edge cases. Deferred rather than shipped fragile: a pin that
  false-positives on legitimate text erodes trust in every pin beside it, and a pin tuned
  just enough to silence today's false positives without a principled fix is liable to
  misfire on the next legitimate rewrite. `test_orchestrator_claude_md_names_only_real_symbols`
  in `tests/regression/test_doc_reconciliation_aqa.py` ships instead, covering the tractable
  half of the same CodeRabbit finding (orchestrator-symbol existence). Revisit if a future
  editor has a cleaner mechanism -- e.g. requiring new prose about `growth/` wiring to link
  to `docs/CHARTER.md` §5 as the single source of truth, checked by presence of the link
  rather than by parsing the claim's own words.
- **Mock Hailo-8 output shapes are not schema-driven.**
  `src/mousedroid/hardware/accelerator/hailo_runtime.py::MockHailoRuntime.DEFAULT_OUTPUT_SHAPES`
  (class-level constants `{"yolo": (25200, 85), "feature_extractor": (256,)}`) are not
  sourced from `HailoConfig`. Deferred because: they are already overridable via
  constructor injection (the `output_shapes` parameter), so no test or consumer is blocked;
  the file is untouched by any current diff, so `scripts/check_no_hardcoded_values.py`'s
  changed-lines-only scan does not reach these lines today — **correction to the rationale
  as originally framed**: this is not a `hardware/` directory-prefix exemption, and no such
  exemption exists (`ALLOWED_DIR_PREFIXES` in that script names exactly four prefixes —
  `src/mousedroid/config/schema/`, `src/mousedroid/telemetry/metrics/`,
  `src/mousedroid/telemetry/server/`, `src/mousedroid/validation/runtime/` — and
  `hardware/` is not one of them; verified by reading the script directly). A genuinely new
  hardcoded literal touching these lines in a future diff would be caught like any other
  file. The values are intrinsic to a specific YOLO architecture's output tensor shape
  (anchor-grid count, embedding dimension), not an operational tunable a deployment would
  ever need to change; and moving them to a Pydantic schema field would add ceremony to
  mock-only test scaffolding without closing a real risk gap.
- **`scripts/prove_pin_fails.sh`'s `--paths`/`--tests` arguments are unquoted,
  space-separated strings** (`for f in ${PATHS}`, documented
  `# shellcheck disable=SC2086`). Two narrow edge cases are unhandled: (a) a glob character
  in `--paths`/`--tests` undergoes incidental shell pathname expansion rather than being
  treated as an intentional, tested feature or explicitly rejected; (b) a pytest
  parametrized test ID containing a space (e.g. `test_x.py::test_y[value with space]`) would
  be mis-split by word-splitting. Deferred because the tool is dev/CI-only (never runtime
  robot code), already has passing coverage in
  `tests/unit/scripts/test_prove_pin_fails.py` (10 tests collected and passing at the time
  of this bundle — `python -m pytest tests/unit/scripts/test_prove_pin_fails.py -q`;
  corrected from "6" in the originally stated rationale) plus prior rounds of adversarial
  hardening this same plan's history records, and the practical failure mode in both cases
  is a clear error message, not silent wrong behaviour.
- **Extending the existing `post_edit` hook to run `shellcheck` over changed `.sh` files,
  and a new `regression_pin_reminder` hook scoped to glob-matched paths.** `.claude/workforce.yaml`'s
  `post_edit:` block already exists and runs `ruff` + `mypy`, but only over `.py` suffixes —
  adding `shellcheck` over `.sh` files is new scope, not a wholly new hook mechanism.
  `regression_pin_reminder` does not exist anywhere in `.claude/settings.json` or
  `.claude/workforce.yaml` today (verified — zero matches) and would be genuinely new.
  Deferred because: `shellcheck` is not installed in this dev container (`which shellcheck`
  finds nothing) and is not wired into `.github/workflows/ci.yml`, `scripts/ci.sh`, or the
  `Makefile` today (verified by grep — zero matches beyond inline `# shellcheck disable=`
  comments that presuppose a run that does not currently happen anywhere); wiring either
  hook properly needs new `tools/claude_hooks/` test coverage under the dedicated
  `make hooks` gate, which is real, non-trivial scope this sprint did not have room for.
  Recorded here as a legitimate future bundle, not silently dropped.
