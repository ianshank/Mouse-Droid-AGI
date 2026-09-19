# Change: `mouse-droid-branch-hygiene-sweep`

**F-number:** F-052 (reserved; F-009–F-014 and F-033 remain burned holes per ADR-013)
**Status:** proposed
**Branch under review:** `claude/openspec-jetson-onnx-delivery-review-y8b3wv` @ `b0759da`
**Compared against:** `18aba56` — the merge base with the repository's default branch,
`claude/markdown-implementation-plan-aVJ2l`.

> **There is no `main` in this repository.** Every "compare to main" instruction in this
> change resolves to the default branch named above. Any plan, script, or CI rule that
> assumes `main` exists is wrong here; `deploy_remote.sh` and `check_branch_coverage.py`
> both derive their base ref rather than naming a branch, which is why they work.

---

## 1. Problem

The `mouse-droid-jetson-onnx-delivery` change (F-050/F-051) landed 78 files,
+14,051/−129, across production code, 1,510 lines of new Bash, Docker, CI and docs.
All 17 CI jobs are green (23 check runs after matrix expansion) and all 38 review threads are resolved. That is the bar CI
enforces — it is not the bar this repository claims to hold.

Four rounds of document review during that change found **none** of its real defects.
The composite world model bypassing warmup, a command-injection introduced by making a
path env-overridable, a provider probe that proved availability rather than the session,
and three separate Windows platform assumptions were each found by something that
*executed*: a test, a CI job on a platform nobody develops on, or an adversarial
reviewer. The generalisation is the premise of this change:

> **A gate that does not run, an inventory that nothing pins, and a document that no
> test reads are all the same defect — they assert without executing.**

This change is a gap-analysis and hygiene sweep of that branch, scoped to the places
where the repository's own stated discipline is not mechanically enforced. It is
deliberately **not** a refactor: §6 records, with evidence, what is already clean and
must not be churned.

## 2. Verified findings

Every finding below was verified against the working tree at `b0759da`. Line numbers
were read, not recalled. Findings the investigation *disproved* are in §6.

### A. Gates that do not run where the work lands

| # | Finding | Evidence |
|---|---|---|
| A-1 | `scripts/check_branch_coverage.py` — the changed-lines branch-coverage gate — runs **only** in `bash scripts/ci.sh`. No GitHub workflow invokes it. | `scripts/ci.sh:178`; `.github/workflows/ci.yml:380-383` declares it "Still local-only by design (need heavy deps)". |
| A-2 | That rationale is stale. The `test` job already installs the heavy deps and runs the full suite, and `local-gates` was created precisely to end "a GitHub-CI-only contributor bypassed all of them" (its own comment, `ci.yml:377-379`). Branch coverage is the one gate that was left behind. | `ci.yml:376-384` vs `ci.yml:428-441` |
| A-3 | The two files this branch put the most new **logic** into are both exempt from that gate: `src/mousedroid/orchestrator/_lifecycle_mixin.py` and `src/mousedroid/factory/orchestrator.py` are in `_ALLOWED_FILES`, admitted as "pure DI wiring" and "orchestrator mixin/_state split products". `_warm_world_model()` is neither — it has an `isinstance` guard, an `asyncio.to_thread` hop, timing, and a failure-propagation path. | `scripts/check_branch_coverage.py:66-99`; `src/mousedroid/orchestrator/_lifecycle_mixin.py:97-160` |
| A-4 | **`shellcheck` is invoked by no gate — despite 19 inline, rule-specific suppressions across 10 files that assume it runs.** `scripts/prove_pin_fails.sh` alone carries 9, and `scripts/rover_wip_guard.sh:124` carries the `SC2064` disable task 1.8 cites by name. *(Revision 1 said "the only two occurrences" — wrong, and the true form is stronger.)* | `grep -rn 'shellcheck disable'` -> 19 across 10 files |
| A-5 | The repository already has a `bash -n` parse-check convention, and the four new operational scripts did not get it. | Convention: `tests/regression/test_host_bootstrap_script.py:5`, `tests/regression/test_jetson_full_validation_script.py:40-50`. Missing for: `scripts/deploy_remote.sh`, `scripts/docker_deploy.sh`, `scripts/download_weights.sh`, `scripts/rover_wip_guard.sh`. |
| A-6 | No workflow exercises `deploy_remote.sh` or `docker_deploy.sh`, not even a dry run. | `grep -rln 'deploy_remote\|docker_deploy' .github/workflows/` → no matches |
| A-7 | `scripts/docker_deploy.sh` (506 lines, and the file carrying the strict promotion gate) has **no unit-tier test**. Its sibling `deploy_remote.sh` has a 607-line one. `docker_deploy.sh` is reached only by regression AQA text assertions. | `tests/unit/scripts/test_deploy_remote_guard.py` (607 lines) vs no `tests/unit/scripts/test_docker_deploy*.py` |

### B. Inventories nothing pins — the mechanism, not just the symptom

| # | Finding | Evidence |
|---|---|---|
| B-1 | `tools/claude_hooks/ratchet_budget_check.py` is **wired and running** as a PostToolUse hook, and is documented in **zero** places: not the hooks runbook (whose table lists 3 of the 4 wired hooks), not `SKILLS.md`, not `AGENTS.md`, not `CLAUDE.md`. | Wired: `.claude/settings.json` PostToolUse. Absent: `docs/runbooks/claude-workforce-hooks.md:17-19`, and `grep -c ratchet_budget_check` = 0 in all three docs. |
| B-2 | **The mechanism that let B-1 drift.** `tests/regression/test_claude_workforce_aqa.py` pins skills→`SKILLS.md` and agents→`AGENTS.md`, and pins that wired hooks reference existing *modules* — but nothing pins that a wired hook is *documented*. Two of three asset classes have a doc-inventory pin; hooks do not. | `test_claude_workforce_aqa.py:290` (skills), `:315` (agents), `:386` + `:789` (hook modules exist). No hooks→runbook test. |
| B-3 | Only 2 of Claude Code's hook events are used: `PreToolUse` and `PostToolUse`. There is no `SessionStart`, `UserPromptSubmit`, `Stop`, `SubagentStop` or `PreCompact` hook. Several repo invariants are session-shaped rather than edit-shaped and currently rely on the operator remembering. | `.claude/settings.json` |
| B-4 | 24 skills exist; **none** covers any reusable action this branch created. `grep -rln 'ceiling\|onnxruntime\|deploy_remote\|docker_deploy\|warmup\|Amdahl' .claude/skills/*/SKILL.md` → zero matches. | `.claude/skills/` (24 dirs) |

### C. Config and schema (audited against the gate's own detector)

| # | Finding | Severity |
|---|---|---|
| C-1 | `scripts/docker_deploy.sh:150` resolves the config overlay with `os.environ.get("MOUSEDROID_CONFIG")` alone. The repository's own resolver honours **two** keys: `_CONFIG_SINGLE_ENV_VARS = ("MOUSEDROID_CONFIG", "MOUSEDROID_JETSON_CONFIG")` (`src/mousedroid/validation/runtime/_shared.py:25`). A rover configured via the legacy key has the **strict promotion gate evaluate the wrong config** — wrong `world_model.engine`, wrong artifact path — and pass. | **HIGH** |
| C-2 | `scripts/docker_deploy.sh:54` falls back through `MOUSEDROID_TELEMETRY_PORT`, which is **not a pydantic-settings key**: `root.py:189-191` sets `env_prefix="MOUSEDROID_"` with `env_nested_delimiter="__"`, so the real key is `MOUSEDROID_TELEMETRY__PORT`. An operator who moves the server port the supported way leaves the promotion health probe pinned to the literal `8080`. | **HIGH** |
| C-3 | 13 env keys these two scripts read are absent from `config/docker.env.example`, which is the input to the `host_env_keys` preflight drift check (`src/mousedroid/validation/preflight.py:440`). None of them is covered by that warning. | MED |
| C-4 | Five new string defaults are pinned only as *truthy*, not by value. `tests/regression/test_f050_aqa.py:243-246` asserts `.onnx_revision` is truthy; no test in the repo asserts `== "main"`, `== "sha256.txt"`, or `== "observe_step.metadata.json"`. `onnx_revision` is the anti-drift pin and the manifest filenames are what the SHA-256 refusal reads — these are the safety-relevant ones. | MED |
| C-5 | `tools/claude_hooks/config.py:353` carries `hardcoded_ok` fallback `ceiling=24, warn_threshold=22`; `.claude/workforce.yaml:129-130` says `26` / `24`. The `noqa` and `type_ignore` fallbacks are in sync; only this one was left behind across the 24→28→26 history. The docstring at `:339-341` claims the fallbacks reproduce the real budgets "exactly" — that claim is now false, and a run with `workforce.yaml` unreadable reports a false breach. | MED |
| C-6 | `Dockerfile.jetson` repeats the version floors of at least 15 dependencies that `pyproject.toml` extras already declare — in a file that, at `:79`, states the rule it violates: "the version floors stay in `pyproject.toml`'s `[onnx_world_model]` extra so there is one source of truth". | MED |
| C-7 | `Dockerfile.jetson:161-162` bakes the Python minor version into a `COPY --from` path (`/usr/local/lib/python3.10/dist-packages/llama_cpp`). A base-image Python bump breaks it silently. `r36.4.0` is also spelt twice (`:12`, `:14`). | LOW |

**Ratchet headroom — the finding that unblocks the rest.** All three suppression budgets
sit at ceiling (`noqa` 19/19, `type_ignore` 8/8, `hardcoded_ok` 26/26), so any fix
needing a new suppression is blocked before it is written. Five markers are reclaimable
without spending anything:

- Three suppress **nothing**: their values are already in the gate's
  `ALLOWED_NUMERIC_VALUES = {0.0, 1.0, -1.0}` —
  `src/mousedroid/comms/_utils.py:23` (`= 1`), `:26` (`= 0`),
  `src/mousedroid/comms/command_set.py:71` (`= 1`). Deleting the marker changes nothing.
- Two duplicate a constant that already exists, which is *verbatim* the debt class the
  28→26 ratchet already resolved once (`.claude/workforce.yaml:123-128`):
  `src/mousedroid/validation/latency_stats.py:30` and
  `src/mousedroid/comms/command_set.py:68` both spell `1000.0` where
  `src/mousedroid/constants.py:67` `MILLISECONDS_PER_SECOND` exists.

That is 26 → 21. The remaining 21 were each verified to suppress a real finding and stay.

### D. Advisory ladder and supply chain

| # | Finding |
|---|---|
| D-1 | `test-windows` is due for a promotion decision: `since: 2026-08-20`, `promote_after_days: 30`. `check_advisory_promotions.py` reports "within window" today (2026-09-19) only because its comparison is strict `>`. This branch drove four consecutive rounds of Windows fixes to green — that is exactly the evidence the window was opened to collect. |
| D-2 | `onnx-world-model-extras` (180 d) and `mlflow-extras` (180 d) both defer to a "7-consecutive-green-run" bar that **no tooling counts**. The `onnx-world-model-extras` entry records being re-extended once because the pass "could not cheaply re-derive the current streak". A calendar proxy for a non-calendar rule will keep producing that outcome. |
| D-3 | `.github/dependabot.yml` covers `github-actions` and `pip` only — no `docker` ecosystem entry, though `Dockerfile.jetson` and `Dockerfile.dev` both pin base images by tag. This branch's entire ONNX-provider probe exists *because* the base image's wheel content silently decides whether the rover runs accelerated or CPU-only. That is the exact class of drift a `docker` ecosystem entry watches. |

## 3. Scope

**In scope.** Closing the gaps in §2 by making the existing discipline execute: wire the
gates that do not run, pin the inventories that nothing checks, fix the two HIGH config
defects, reclaim ratchet headroom, extract the reusable actions this branch proved out
into skills, and reconcile the documentation surfaces the branch moved.

**Out of scope.**

- **Decomposing any "god file."** See §6 — the measurement does not support it.
- Re-opening the task-2.1 ceiling decision. The computed end-to-end ceiling was
  1.0016x–1.0039x and Phases 5–6 stood down; nothing here revisits that.
- Anything requiring the rover. F-050 tasks 7.3–7.5 and F-051 task 8.7 (the offline
  rollback drill) stay open and rover-gated. This change does not pretend to close them.
- Training real weights. `build_weight_update_loader` returning `None` unconditionally is
  recorded as known debt, not fixed here.

## 4. Success criteria

1. `check_branch_coverage.py` runs in GitHub CI, and no file containing branching logic
   is exempt from it on the grounds of being "pure DI wiring."
2. `shellcheck` and `bash -n` gate every tracked `.sh` file, with any suppression
   declared inline and counted.
3. A wired hook that no document mentions fails a test.
4. `docker_deploy.sh` resolves config through the repository's own resolver and reads the
   telemetry port from `Settings`, proven by a negative test that sets the legacy key.
5. The five safety-relevant string defaults are pinned by value in
   `tests/regression/test_f050_backwards_compat.py`.
6. `hardcoded_ok` measures 21 with the ceiling ratcheted to match. `noqa` and
   `type_ignore` unchanged.
7. `test-windows` has a recorded promotion decision — promoted, or re-extended with a
   verified count and a reason.
8. A `docker` ecosystem entry exists in `.github/dependabot.yml`.
9. `make gates` and `make test` pass; `python -m tools.ratchet_budgets --strict` exits 0.
10. Every claim in this bundle that says "verified" cites a line that resolves on disk.

## 5. Non-goals that look like goals

Three things this change deliberately does **not** do, because the request could be read
as asking for them and the evidence says not to:

- **"God-file decomposition."** `scripts/analyze_observe_step_ceiling.py` is 1,111 lines
  — the largest file this branch added — and it is *not* a god file. It holds 29
  top-level definitions and, measured with `ruff --isolated --select C901
  --config 'lint.mccabe.max-complexity = 15'`, **not one function exceeds the
  ceiling**. Decomposing it would be churn.
- **"Fix ruff/mypy/numpy."** There is nothing to fix. `mypy --strict` reports
  `Success: no issues found in 424 source files`. `ruff --select NPY` (including
  `NPY002`) is clean across `src/` and `training/`. Saying so is the deliverable.
- **"Add dependabot."** It exists (`.github/dependabot.yml`, 3,707 bytes). The gap is
  ecosystem coverage (D-3), not the file.

## 6. Verified clean — do not churn

Recorded so a later pass does not "fix" these and so the claims above stay falsifiable.

| Claim | Evidence |
|---|---|
| `mypy --strict` clean | `Success: no issues found in 424 source files` |
| `ruff NPY`, incl. `NPY002`, clean | `ruff check src/ training/ --select NPY` → `All checks passed!` |
| No new complexity offender in `scripts/` | With per-file-ignores bypassed at the repo ceiling of 15, exactly 3 offenders — the same three `pyproject.toml:282-287` already names. This branch added none. |
| All 6 new shell scripts parse | `bash -n` clean on each |
| Every `validation_command` resolves | All 22 `scripts/validations/F-*.sh` referenced by `features.yaml` exist on disk. F-047/F-049 are `deferred` with no command — correct, not a gap. |
| `factory/`, `orchestrator/_`, `config/schema/` allowlists earned nothing on this delta | Zero literals on the added lines would be flagged even with the directory allowlist removed |
| The four new/changed world-model and orchestrator modules carry no unschematised tunable | `observe_step_timing.py` contains no numeric literal at all; buckets come from `MetricsConfig`; `_warm_world_model` reads only `time.perf_counter()` |
| All 7 new config fields carry `Field(default=..., description=...)` | Asserted programmatically at `tests/regression/test_f050_aqa.py:219-233`; both strict switches default `False` and `revision="main"` reproduces `hf_hub_download`'s own default, so existing YAML loads byte-identically |
| No `getattr(cfg, "field", default)` schema bypass, and no `os.environ` read outside `Settings` in `src/` on this delta | Every added line scanned |
| The `ObserveStepTimingMixin` `TYPE_CHECKING`-only `_StateHookBase` is load-bearing | It exists to avoid spending a suppression from an exhausted budget. Three simpler shapes were tried and each cost one. Preserve it. |

## 7. Authoritative counterparts

Per `openspec/project.md`, this tree is documentation-only and the repo-native artifacts
win on disagreement:

- `features.yaml` — F-052
- `scripts/validations/F-052.sh` — the harness-executed proof
- `.github/workflows/ci.yml`, `scripts/ci.sh`, `Makefile` — the gates
- `.claude/workforce.yaml`, `.claude/settings.json` — workforce configuration
- `docs/architecture/ADR-*.md` — any decision this change ratifies
