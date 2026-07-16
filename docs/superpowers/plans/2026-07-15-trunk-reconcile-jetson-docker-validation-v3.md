# Trunk Reconciliation + Jetson Docker Deploy + Full Smoke/E2E Validation (v3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> At execution time, copy this plan to `docs/superpowers/plans/2026-07-15-trunk-reconcile-jetson-docker-validation-v3.md` (the repo's canonical plan location) and commit it with PR-B.

**Goal:** Salvage the valuable in-flight work stranded on the obsolete `refactor/onnx-default-providers-common` branch into three focused trunk PRs, run the full local CI at trunk tip in an isolated worktree, deploy trunk to the Jetson via the Docker path (conditional image rebuild), run the complete 3-phase on-device smoke/e2e validation while monitoring logs live, and hand off into the Phase 6 shadow-soak milestone.

**Architecture:** Operational + small-fix plan on two decoupled tracks. **Track 1 (code, async):** salvage the stranded working-tree fixes into 3 focused trunk PRs. **Track 2 (ops, runs today):** deploy trunk to the Jetson and run the full on-device validation — it composes existing tooling (`scripts/ci.sh`, `scripts/jetson_full_validation.sh`, `scripts/sync_jetson_overlay.sh`, `docker compose`, `mousedroid.cli.preflight --trend`). The two tracks share exactly one dependency: the rover validation is cleaner with PR-B's hardened `jetson_full_validation.sh`, which is applied to the rover from the *pushed* PR-B branch (a cherry-pick) — **not** gated on PR-B *merging*. Isolation via `git worktree` (primary tree untouched until W7). The Jetson deploy uses the git-based Docker spine (bind-mounted source at `/opt/mousedroid`), NOT `deploy_remote.sh` (parallel systemd/venv layout; its `rsync --delete` would destroy rover work) and NOT `docker_deploy.sh` (it forces `--no-cache`).

## Peer-review corrections (verified against the repo, 2026-07-15)

This plan's load-bearing claims were verified directly (not taken from subagent summaries):

1. **gateway.py diff** verified verbatim — `except OSError` → `except (OSError, ValueError) as exc`, event `llm_gateway_degraded_model_not_found` → `llm_gateway_degraded_model_error`, `error=str(exc)` added. `69bf64f` touches only `test_anthropic_gateway_wiring.py` (+2 lines: `Path` import + `cfg.llm.model_path = Path("/tmp/does_not_exist_for_test.gguf")`). PR-A is accurate.
2. **`scripts/ci.sh` `-m "not hardware"` on the performance stage is ALREADY on trunk** (`scripts/ci.sh:84`, landed via #160). → the uncommitted `ci.sh` hunk is **dropped** from PR-B (definitive, not conditional).
3. **None of the 6 `jetson_full_validation.sh` hardening hunks are on trunk** (trunk still has `for stage in system usbc gpio ...` with `system` blocking). → PR-B is genuinely net-new and needed.
4. **mypy is pinned `2.1.0` on the branch, bumped to `2.2.0` on trunk** (#156 `767e566`). → PR-C's "re-verify type-ignores under 2.2.0" premise is sound; the worktree venv (built from trunk's pyproject) resolves mypy 2.2.0.
5. **ruff `select` has no `NPY`** (`["E","W","F","I","N","UP","B","A","C4","SIM","RUF","PT","S","ANN","D","T20"]`). numpy is grep-clean of deprecated aliases. Enabling `NPY` stays a deferred follow-up.
6. **#141 (`467c45b`) confirms the branch's ONNX refactor already merged** → the branch is obsolete; only `69bf64f` is net-new committed work.
7. **STRUCTURAL:** Track 2 no longer blocks on PRs *merging*. The deploy targets trunk tip (`21463c3` or later); PR-B's hardening reaches the rover via its pushed branch; PR-A (gateway ValueError) and PR-C (types/isolation) are irrelevant to the rover run (the rover's Phi-3 model_path is valid, so the ValueError path never fires) and merge on their own review cadence.

**Tech Stack:** git worktree • ruff 0.8.0 (pinned) • mypy 2.2.0 --strict • pytest `--import-mode=importlib` (unit/property/integration/performance/regression/e2e/hardware tiers) • Docker + compose on Jetson (L4T r36.4.0 base) • ssh alias `jetson` • structlog • Prometheus/Grafana/Loki (already running rover-side).

## Context (why this plan exists)

The 2026-07-11 trunk-sync plan was executed on 2026-07-12; its two findings-fix PRs are now merged (**#161** OOM-guard on 07-12, **#160** Windows skips + hardware gate on 07-15), so trunk `origin/claude/markdown-implementation-plan-aVJ2l` @ `21463c3` is clean and ready for a v3 validation pass. Meanwhile:

- The current branch `refactor/onnx-default-providers-common` is **obsolete as a branch**: its ONNX refactor already merged to trunk as PR #141 (squash `467c45b`). Net-new committed work = one unpushed 2-line test fix (`69bf64f`).
- The working tree holds **valuable uncommitted fallout fixes** from the 07-12 on-device run: a real `llm_gateway.py` ValueError-degradation fix, six `jetson_full_validation.sh` hardening hunks, a `ci.sh` perf-stage hardware gate, three test-isolation fixes, four mypy type-ignores (authored vs mypy 2.1.0; trunk is now on 2.2.0 — must re-verify), one format reflow. Some hunks may already be upstream via #160/#161 — reconciliation required.
- The rover sits on a rover-local commit lineage (`7fdb9e3`) with a 764-line uncommitted diff (mostly line-ending churn) that must be preserved before any sync.
- The deployed image pin is `deployments/jetson-image.json` @ `032942b` (image built 2026-06-02); source is bind-mounted, so deploy = git sync + conditional rebuild.
- Roadmap-wise, Phase 6 on-device learning is code-complete and sim-validated (#135/#137/#138); its next gate is a ≥30-day shadow soak on the rover — this plan's green run is the precondition.

**User decisions (2026-07-15):** conditional Docker rebuild • 3 focused salvage PRs • preserve rover WIP on a rover-local branch • Phase 6 shadow soak is the next milestone after green.

## Global Constraints

- **Trunk ref:** `origin/claude/markdown-implementation-plan-aVJ2l` (tip `21463c3` at plan time). `origin/main` does NOT exist — never fetch it. Expose as `TRUNK_REF` env var.
- **No hardcoded values:** every host, port, path, ref, timeout, container name comes from an env var with a documented default (`MOUSEDROID_*` prefix, `__` nested delimiter for schema fields).
- **Primary working tree untouched** until Task W7 cleanup: no `git checkout/reset/stash pop/clean` against it. Isolation via `git worktree add --detach`.
- **Rover work preserved before sync:** commit to `rover/wip-20260715` rover-local branch; archive whitespace-insensitive diff to workstation. Never `deploy_remote.sh` (rsync --delete); never `git clean` on the rover.
- **Ruff pinned 0.8.0**, mypy 2.2.0 (trunk #156), pytest always `--import-mode=importlib`, coverage gate `--cov-fail-under=85`.
- **Install `.[dev,telemetry,mcp]`** in validation venvs — `.[dev]` alone yields 6 fake mypy errors (2026-07-12 lesson).
- **`REPORT_ROOT` outside the worktree:** `${HOME}/mousedroid-trunk-sync-reports` (worktree add fails if target dir exists — 2026-07-12 lesson).
- **Secrets presence-checked only** (`[ -n "$VAR" ]`); never echoed, never passed on an SSH command line.
- **No motion:** `MOUSEDROID_ESP32__ENABLED=false` for hardware pytest; ESP32 is functionally dead — `serial/motor/power` (+ `system` after PR-B) smoke stages are expected WARN, not FAIL.
- **Cold-then-warm on Jetson:** Phase 2 stops the container; the script's EXIT trap always restarts it — verify container healthy at the end regardless of outcome.
- **structlog only** — never `print()`. Backwards-compatible config only (new fields default).
- **Commit messages end with:** `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; PR bodies end with the Claude Code attribution footer.

## Static-analysis findings inventory (the "take note of ruff/lint/mypy/numpy" ask)

| Tool | Finding | Disposition in this plan |
|------|---------|--------------------------|
| ruff format | Untracked `ruff_format_output.txt` claims **71 files would be reformatted** — but trunk CI (pinned ruff 0.8.0) is green, so this snapshot was almost certainly produced by a mismatched ruff version (dependabot 0.15.x is open as PR #149, unmerged). | Task W6 re-runs `ruff format --check` with pinned 0.8.0 to settle it; the stale file is deleted in W7. Do NOT mass-reformat on the strength of the stale snapshot. |
| mypy | 4 working-tree `# type: ignore` additions for torch-stub strictness (`torch.optim.Adam`, `torch.jit.save`) were authored against mypy 2.1.0; trunk bumped to 2.2.0 (#156). Under `--strict`, stale ignores FAIL via `warn_unused_ignores`. | Task 4 (PR-C) re-verifies each on trunk's mypy before including any. |
| numpy | Grep-verified clean: zero deprecated aliases (`np.bool/int/float/object/str/...`) in `src/` and `tests/`. However ruff's `NPY` rule category is NOT enabled in `[tool.ruff.lint] select`, and the numpy pin (`>=1.24,!=2.0.0,!=2.0.1`) allows 2.x — NPY201-style migration risk is unlinted. | Recorded as a finding; enabling `NPY` is deliberately out of scope (separate follow-up PR candidate, noted in Task 12 handoff). |
| ruff check | No known violations on trunk (CI green). Working-tree hunks must pass `ruff check` before each PR. | Per-PR gate in Tasks 2–4; full-tree audit in Task 6. |
| MuJoCo | `MUJOCO_LOG.TXT` = harmless sim-instability warning (NaN QACC at DOF 6, 2026-06-07), not a crash. | Deleted in W7. |

---

## Execution order & dependencies

Two decoupled tracks. Tasks are numbered by concern, not strict run order.

**Shared prerequisite:** W1 → W2 (worktree + triage) and W6 Steps 1–2 (build the venv). The venv is numbered W6 for narrative grouping but its setup steps run early — W3–W5 gates depend on `$PY`/`$VENV_DIR`.

**Track 1 — code salvage (async, human-reviewed):**
- **W3 (PR-A), W4 (PR-B), W5 (PR-C)** — independent; author in any order. Each ends at `gh pr create`. **Author W4 (PR-B) first** so its branch is pushed before Track 2 needs the hardened validation script.
- **W6 Steps 3–7** (full local CI + findings + trend) — run against a trunk-tip worktree (or a temporary branch combining all three PRs for a pre-merge dry run). This is the workstation mirror of "run all smoke/e2e."
- PRs A/B/C merge on their own CI + review cadence. **No operational task blocks on their merge.**

**Track 2 — Jetson deploy + validation (runs today, in parallel with Track 1 review):**
- **J1** (preserve rover WIP) → **J2** (deploy rover to trunk tip — needs only that PR-B's *branch is pushed*, not merged) → **J3** (overlay/env) → **J4** (conditional rebuild) → **J5** (health) → **J6** (full validation + live monitoring) → **J8** (handoff). **J7** is the rollback contingency.
- J2 layers PR-B's hardened `jetson_full_validation.sh` onto the rover from the pushed PR-B branch (cherry-pick). If PR-B isn't authored yet, J2 falls back to trunk's current script (the run will show the known 07-12 WARN/FAIL noise that PR-B fixes — acceptable but noisier).

**Cleanup (last):**
- **W7 (primary-tree cleanup)** — only after all 3 PRs merge AND both local + Jetson validation are green. Discards the working-tree copies once every hunk is confirmed upstream.

**Interrupt-safety:** every destructive step (W7 delete/restore, J2 detached checkout, J4 rebuild) is gated on a preserve-and-verify predecessor (W1 snapshot, J1 rover WIP branch, J4-B2 rollback tag). Nothing is discarded before its replacement is confirmed upstream/committed.

---

# PART 1 — Workstation: salvage the branch, land 3 PRs, run local validation, clean up

**Conventions for every Part-1 task**

- Run in **Git Bash** at the repo root (`$REPO_ROOT` — default `/c/Users/iansh/OneDrive/Documents/Gronk-Droid-Jetson-Nano`). The primary tree is on branch `refactor/onnx-default-providers-common` and MUST stay dirty and untouched until Task W7.
- Session setup (re-run at the top of any resumed session):

```bash
export REPO_ROOT="${REPO_ROOT:-/c/Users/iansh/OneDrive/Documents/Gronk-Droid-Jetson-Nano}"
export TRUNK_REF=origin/claude/markdown-implementation-plan-aVJ2l
export TRUNK_SHORT=claude/markdown-implementation-plan-aVJ2l
export WORKTREE_DIR="$HOME/mousedroid-trunk-sync"        # sibling of the primary tree, NOT inside it
export VENV_DIR="$WORKTREE_DIR/.venv"
export STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export REPORT_ROOT="$HOME/mousedroid-trunk-sync-reports"  # OUTSIDE the worktree (worktree add fails if target exists)
mkdir -p "$REPORT_ROOT/$STAMP/local"
```

- Never `git checkout/reset/stash/clean` in the primary tree. Isolation is a detached worktree.

---

## Task W1: Snapshot primary tree + create the trunk-tip reconciliation worktree

**Files:**
- Read: `.git/HEAD`, `git status --porcelain`
- Create: `$WORKTREE_DIR/` (detached worktree at trunk tip)
- Write: `$REPORT_ROOT/$STAMP/primary_tree_before.log`

**Interfaces:**
- Consumes: session env (`TRUNK_REF`, `WORKTREE_DIR`).
- Produces: a clean trunk checkout; a byte-exact snapshot of the primary tree's dirty state (consumed by W7's restoration proof).

- [ ] **Step 1: Snapshot the primary tree's dirty state (restoration baseline)**

```bash
cd /c/Users/iansh/OneDrive/Documents/Gronk-Droid-Jetson-Nano
{ echo "=== HEAD ==="; git rev-parse HEAD
  echo "=== branch ==="; git rev-parse --abbrev-ref HEAD
  echo "=== porcelain ==="; git status --porcelain | sort
} > "$REPORT_ROOT/$STAMP/primary_tree_before.log"
cat "$REPORT_ROOT/$STAMP/primary_tree_before.log"
```

Expected: HEAD `69bf64f`, branch `refactor/onnx-default-providers-common`, 11 `M` + 5 `??` lines (incl. the `docs/superpowers/plans/2026-07-11-...md` untracked plan doc).

- [ ] **Step 2: Fetch trunk (narrow, no tags)**

```bash
git fetch --no-tags origin "$TRUNK_SHORT"
TRUNK_SHA="$(git rev-parse "$TRUNK_REF")"
echo "trunk_sha=$TRUNK_SHA" | tee "$REPORT_ROOT/$STAMP/env.log"
```

Expected: `trunk_sha=21463c3...` (or later if PRs already merged). `origin/main` is NEVER fetched — it does not exist on this remote.

- [ ] **Step 3: Create the detached worktree at trunk tip**

```bash
git worktree add --detach "$WORKTREE_DIR" "$TRUNK_SHA"
```

Expected: `Preparing worktree (detached HEAD 21463c3)` / `HEAD is now at 21463c3 fix(ci): skip Windows-incompatible...`. If `$WORKTREE_DIR` already exists, `git worktree add` errors — remove the stale worktree (`git worktree remove "$WORKTREE_DIR" --force`) first.

- [ ] **Step 4: Verify isolation (primary tree byte-for-byte unchanged)**

```bash
diff "$REPORT_ROOT/$STAMP/primary_tree_before.log" \
     <(cd /c/Users/iansh/OneDrive/Documents/Gronk-Droid-Jetson-Nano && { echo "=== HEAD ==="; git rev-parse HEAD; echo "=== branch ==="; git rev-parse --abbrev-ref HEAD; echo "=== porcelain ==="; git status --porcelain | sort; }) \
  && echo "primary tree UNCHANGED — good"
git -C "$WORKTREE_DIR" status --porcelain | wc -l   # expect 0
```

Expected: no diff, worktree porcelain = 0 lines. If the primary tree changed, STOP.

---

## Task W2: Port the uncommitted diff into the worktree + triage what is already upstream

**Files:**
- Read (primary tree): the 11 modified files
- Write: `$REPORT_ROOT/$STAMP/uncommitted.patch`, `$WORKTREE_DIR/*.rej` (transient)

**Interfaces:**
- Consumes: primary-tree working diff.
- Produces: a per-file classification — `already-upstream` (hunk applies as no-op / rejects because trunk already has it) vs `net-new` (applies cleanly) — driving which PR each hunk lands in.

- [ ] **Step 1: Capture the primary tree's uncommitted diff as a patch (CRLF-safe)**

```bash
cd /c/Users/iansh/OneDrive/Documents/Gronk-Droid-Jetson-Nano
git diff > "$REPORT_ROOT/$STAMP/uncommitted.patch"
wc -l "$REPORT_ROOT/$STAMP/uncommitted.patch"     # ~ +36/-23 → ~90-120 lines
```

Expected: a non-empty unified diff covering all 11 files.

- [ ] **Step 2: 3-way apply into the worktree, keeping rejects for triage (does NOT commit)**

```bash
cd "$WORKTREE_DIR"
git apply --3way --reject "$REPORT_ROOT/$STAMP/uncommitted.patch" 2>&1 | tee "$REPORT_ROOT/$STAMP/apply.log" || true
ls -1 **/*.rej 2>/dev/null || echo "no .rej files"
git status --porcelain
```

Expected: files that are net-new apply cleanly (show as `M`); any hunk already on trunk (suspected: the `ci.sh` `-m "not hardware"` perf-stage gate landed via #160, and possibly some `jetson_full_validation.sh` hunks) either applies as a no-op or leaves a `.rej`. Record which files produced `.rej`.

- [ ] **Step 3: Classify each modified file against trunk**

```bash
for f in src/mousedroid/llm_gateway/gateway.py scripts/jetson_full_validation.sh scripts/ci.sh \
         scripts/check_settings_identity.py tests/integration/test_e2e_5sec_run.py \
         tests/unit/config/test_loader_two_overlay_composition.py tests/unit/test_tool_registry.py \
         src/mousedroid/efficiency/tensorrt.py src/mousedroid/learning/offline_rl.py \
         src/mousedroid/training/rssm_pretrainer.py src/mousedroid/world_model/latent_utils.py; do
  echo "=== $f ==="
  git diff HEAD -- "$f" | head -1   # empty line ⇒ hunk already matches trunk (already upstream)
done | tee "$REPORT_ROOT/$STAMP/classification.log"
```

Expected: files whose `git diff HEAD` is empty were already on trunk (drop them). The rest are net-new and get grouped into PR-A/B/C below.

- [ ] **Step 4: Reset the worktree to a clean trunk tip before starting the PR branches**

```bash
cd "$WORKTREE_DIR"
git checkout -- . && rm -f **/*.rej && git status --porcelain | wc -l   # expect 0
```

Expected: 0. The classification is captured; each PR branch re-applies only its own hunks from `uncommitted.patch`. (The `git apply` was a triage probe, not the delivery mechanism.)

---

## Task W3: PR-A — `fix(llm-gateway): degrade gracefully on ValueError model-load failures`

**Files:**
- Modify: `src/mousedroid/llm_gateway/gateway.py`
- Test: `tests/integration/test_anthropic_gateway_wiring.py` (cherry-pick `69bf64f`)

**Interfaces:**
- Consumes: clean worktree (W2 Step 4); the working-tree `gateway.py` hunk from `uncommitted.patch`.
- Produces: merged PR-A on trunk (consumed by J2 ancestry check + J6 `llm_gateway_degraded_model_error` monitor).

- [ ] **Step 1: Branch from trunk tip**

```bash
cd "$WORKTREE_DIR"
git switch -c fix/llm-gateway-valueerror-degradation
```

- [ ] **Step 2: Cherry-pick the committed test `69bf64f` (the failing test, first)**

```bash
git cherry-pick 69bf64f
```

Expected: applies the +2 lines to `tests/integration/test_anthropic_gateway_wiring.py` (forcing secondary-backend degradation via a non-existent `model_path`). If the cherry-pick conflicts, resolve to keep both trunk context and the test change.

- [ ] **Step 3: Run the test WITHOUT the src fix — watch it FAIL (TDD red)**

```bash
"$VENV_DIR/Scripts/python.exe" -m pytest tests/integration/test_anthropic_gateway_wiring.py --import-mode=importlib -v 2>&1 | tail -20
```

Expected: FAIL — the gateway currently only catches `OSError`, so the `ValueError` from the bad `model_path` propagates instead of degrading. (If the venv isn't built yet, do Task W6 Step 1–2 first, or run against the primary tree's interpreter.)

- [ ] **Step 4: Apply ONLY the gateway.py hunk from the captured patch**

```bash
git apply -p1 --include='src/mousedroid/llm_gateway/gateway.py' "$REPORT_ROOT/$STAMP/uncommitted.patch"
git diff --stat
```

Expected: `gateway.py` shows the `except OSError` → `except (OSError, ValueError) as exc` broadening, the log-event rename to `llm_gateway_degraded_model_error`, and the added `error=str(exc)` field.

- [ ] **Step 5: Run the test WITH the fix — watch it PASS (TDD green)**

```bash
"$VENV_DIR/Scripts/python.exe" -m pytest tests/integration/test_anthropic_gateway_wiring.py --import-mode=importlib -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 6: Local gates on touched files**

```bash
"$VENV_DIR/Scripts/python.exe" -m ruff check src/mousedroid/llm_gateway/gateway.py tests/integration/test_anthropic_gateway_wiring.py
"$VENV_DIR/Scripts/python.exe" -m ruff format --check src/mousedroid/llm_gateway/gateway.py
"$VENV_DIR/Scripts/python.exe" -m mypy src/mousedroid/llm_gateway/gateway.py --strict --ignore-missing-imports
```

Expected: all clean.

- [ ] **Step 7: Commit, push, open PR**

```bash
git add src/mousedroid/llm_gateway/gateway.py tests/integration/test_anthropic_gateway_wiring.py
git commit -m "fix(llm-gateway): degrade gracefully on ValueError model-load failures

llama_cpp raises ValueError (not just OSError) on an invalid/corrupt model
file. Broaden the except clause so the gateway degrades to the fallback
backend instead of crashing; rename the log event to
llm_gateway_degraded_model_error and attach error=str(exc).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin fix/llm-gateway-valueerror-degradation
gh pr create --base "$TRUNK_SHORT" --title "fix(llm-gateway): degrade gracefully on ValueError model-load failures" --body "$(cat <<'EOF'
## What
Broaden the model-load except clause in the LLM gateway from `OSError` to
`(OSError, ValueError)` so a corrupt/invalid GGUF path degrades to the
fallback backend instead of propagating. Log event renamed to
`llm_gateway_degraded_model_error` with an `error` field.

## Why
Surfaced by the 2026-07-12 on-device run: `llama_cpp` raises `ValueError`
on some bad model files, which the old `except OSError` missed, crashing
the gateway instead of failing over.

## Test
`tests/integration/test_anthropic_gateway_wiring.py` (from 69bf64f) forces
secondary-backend degradation via a non-existent `model_path`; red before
the fix, green after.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR-A opens against trunk.

---

## Task W4: PR-B — `fix(validation): jetson full-validation hardening + record the trunk-sync plan`

**Files:**
- Modify: `scripts/jetson_full_validation.sh`, `scripts/ci.sh` (ONLY hunks not already upstream per W2 Step 3)
- Add: `docs/superpowers/plans/2026-07-11-trunk-sync-local-and-jetson-e2e-validation.md` (+ this v3 plan, see Step 4)

**Interfaces:**
- Consumes: W2 classification (drop the `ci.sh` hunk if #160 already has it).
- Produces: merged PR-B on trunk so the rover pulls the hardened `jetson_full_validation.sh` before Task J6.

- [ ] **Step 1: Branch from trunk tip**

```bash
cd "$WORKTREE_DIR"
git switch "$TRUNK_SHA" --detach && git rev-parse HEAD && git switch -c fix/jetson-full-validation-hardening
```

- [ ] **Step 2: Apply ONLY the `jetson_full_validation.sh` hunks (the `ci.sh` hunk is verified already-upstream — DROP it)**

```bash
git apply -p1 --include='scripts/jetson_full_validation.sh' "$REPORT_ROOT/$STAMP/uncommitted.patch"
git diff --stat
```

Expected: `jetson_full_validation.sh` shows the 6 hunks (perf-tests-on-device, `safe.directory`, `sleep 10`, mic-disable, `system` stage → non-blocking, CPU-forced translate probe). **Do NOT apply the `ci.sh` hunk** — trunk's `scripts/ci.sh:84` already carries `tests/performance/ -m "not hardware"` (landed via #160), so the working-tree hunk is a stale no-op against the branch's older ci.sh. Verify it's redundant: `git show "$TRUNK_REF":scripts/ci.sh | grep -n 'performance/ -m "not hardware"'` returns a hit.

- [ ] **Step 3: Syntax-check both scripts + run the meta-sanity test**

```bash
bash -n scripts/jetson_full_validation.sh && echo "jfv syntax OK"
bash -n scripts/ci.sh && echo "ci syntax OK"
"$VENV_DIR/Scripts/python.exe" -m pytest tests/smoke/test_jetson_full_validation_sanity.py --import-mode=importlib -v 2>&1 | tail -15
```

Expected: both `syntax OK`; the sanity test (which parses the validation script's structure) PASSes.

- [ ] **Step 4: Add the plan docs (record lessons-learned on trunk)**

```bash
cp /c/Users/iansh/OneDrive/Documents/Gronk-Droid-Jetson-Nano/docs/superpowers/plans/2026-07-11-trunk-sync-local-and-jetson-e2e-validation.md docs/superpowers/plans/
cp /c/Users/iansh/.claude/plans/please-scan-codebase-branch-greedy-bengio.md docs/superpowers/plans/2026-07-15-trunk-reconcile-jetson-docker-validation-v3.md
git add docs/superpowers/plans/2026-07-11-trunk-sync-local-and-jetson-e2e-validation.md docs/superpowers/plans/2026-07-15-trunk-reconcile-jetson-docker-validation-v3.md
```

- [ ] **Step 5: Lint the scripts (tools/scripts are in ruff scope), commit, push, PR**

```bash
"$VENV_DIR/Scripts/python.exe" -m ruff check scripts/ 2>&1 | tail -5   # scripts/*.py only; .sh files are not linted by ruff
git add scripts/jetson_full_validation.sh scripts/ci.sh 2>/dev/null
git commit -m "fix(validation): jetson full-validation hardening from the 2026-07-12 on-device run

- Phase 2 hardware pytest now also runs tests/performance/ (endurance test
  moved on-device; ci.sh perf stage already gates it out via #160).
- git safe.directory before in-container ci.sh (dubious-ownership fix).
- sleep 10 after docker stop (device-handle settle before cold probes).
- MOUSEDROID_MICROPHONE__ENABLED=false on real preflight (flaky mic enum).
- system smoke stage moved to non-blocking (CUDA/TensorRT import flap).
- translate_mission probe forced to CPU (MOUSEDROID_LLM__N_GPU_LAYERS=0)
  to avoid iGPU contention with the world model.
- Record the 2026-07-11 + 2026-07-15 validation plans for the audit trail.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin fix/jetson-full-validation-hardening
gh pr create --base "$TRUNK_SHORT" --title "fix(validation): jetson full-validation hardening (2026-07-12 on-device run)" --body "$(cat <<'EOF'
## What
Six operational-hardening hunks to `scripts/jetson_full_validation.sh` from
the 2026-07-12 on-device run, plus the recorded trunk-sync plans.

## Why
Each hunk fixes a concrete failure observed on the rover: dubious-ownership
git errors in-container, device-handle races after `docker stop`, flaky mic
enumeration, a `system`-stage CUDA/TensorRT import flap, and iGPU contention
on the cloud-LLM translate probe. No motion is ever armed.

## Test
`bash -n` on both scripts; `tests/smoke/test_jetson_full_validation_sanity.py`
green.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR-B opens.

---

## Task W5: PR-C — `chore(types): re-verify torch type-ignores under mypy 2.2.0 + test isolation`

**Files:**
- Modify (conditionally): `src/mousedroid/efficiency/tensorrt.py`, `src/mousedroid/learning/offline_rl.py`, `src/mousedroid/training/rssm_pretrainer.py`, `src/mousedroid/world_model/latent_utils.py`
- Modify: `tests/unit/config/test_loader_two_overlay_composition.py`, `tests/unit/test_tool_registry.py`, `tests/integration/test_e2e_5sec_run.py`

**Interfaces:**
- Consumes: trunk's mypy `2.2.0` (bumped via #156) — the type-ignores were authored against 2.1.0 and may now be unused (`warn_unused_ignores` under `--strict` would FAIL on a stale ignore).
- Produces: merged PR-C; only the type-ignores mypy 2.2.0 still requires + the test-isolation fixes.

- [ ] **Step 1: Branch from trunk tip**

```bash
cd "$WORKTREE_DIR"
git switch "$TRUNK_SHA" --detach && git rev-parse HEAD && git switch -c chore/types-mypy22-and-test-isolation
```

- [ ] **Step 2: Establish the mypy 2.2.0 baseline WITHOUT the type-ignore hunks**

```bash
"$VENV_DIR/Scripts/python.exe" -m mypy --version   # expect mypy 2.2.0
"$VENV_DIR/Scripts/python.exe" -m mypy src/mousedroid/efficiency/tensorrt.py src/mousedroid/learning/offline_rl.py src/mousedroid/training/rssm_pretrainer.py src/mousedroid/world_model/latent_utils.py --strict --ignore-missing-imports 2>&1 | tee "$REPORT_ROOT/$STAMP/local/mypy_baseline_typeignores.log" | tail -20
```

Expected: EITHER `Success` (trunk's mypy 2.2.0 no longer needs the ignores → **drop all four hunks**, this PR becomes test-isolation-only) OR a list of `torch.optim.Adam`/`torch.jit.save` errors (→ add only the hunks that resolve them).

- [ ] **Step 3: Apply type-ignore hunks ONLY for files mypy still flags**

```bash
# For each file that FAILED in Step 2, apply its hunk; skip files that passed.
for f in src/mousedroid/efficiency/tensorrt.py src/mousedroid/learning/offline_rl.py src/mousedroid/training/rssm_pretrainer.py src/mousedroid/world_model/latent_utils.py; do
  if grep -q "$f" "$REPORT_ROOT/$STAMP/local/mypy_baseline_typeignores.log"; then
    git apply -p1 --include="$f" "$REPORT_ROOT/$STAMP/uncommitted.patch" && echo "applied $f"
  else
    echo "skip $f (mypy 2.2.0 clean without ignore)"
  fi
done
```

Expected: applies only the still-needed hunks. If a hunk's `type: ignore[code]` uses a code mypy 2.2.0 renamed, adjust the bracketed code to match the Step-2 output.

- [ ] **Step 4: Apply the three test-isolation hunks**

```bash
for f in tests/unit/config/test_loader_two_overlay_composition.py tests/unit/test_tool_registry.py tests/integration/test_e2e_5sec_run.py; do
  git apply -p1 --include="$f" "$REPORT_ROOT/$STAMP/uncommitted.patch" && echo "applied $f"
done
git diff --stat
```

Expected: the monkeypatch.delenv, the `resolve_credentials` RuntimeError patch, and the two `llm={"enabled": False}` fixture additions.

- [ ] **Step 5: Verify mypy strict clean + targeted tests pass**

```bash
"$VENV_DIR/Scripts/python.exe" -m mypy src/mousedroid --strict --ignore-missing-imports 2>&1 | tail -3   # Success: no issues
"$VENV_DIR/Scripts/python.exe" -m pytest tests/unit/config/test_loader_two_overlay_composition.py tests/unit/test_tool_registry.py tests/integration/test_e2e_5sec_run.py --import-mode=importlib -v 2>&1 | tail -15
```

Expected: mypy `Success`; all three test files PASS. Critically, run once with `MOUSEDROID_LLM__N_GPU_LAYERS` set in the env to prove the isolation fix works:
```bash
MOUSEDROID_LLM__N_GPU_LAYERS=7 "$VENV_DIR/Scripts/python.exe" -m pytest tests/unit/config/test_loader_two_overlay_composition.py --import-mode=importlib -q 2>&1 | tail -3
```
Expected: still PASS (the `monkeypatch.delenv` neutralizes the leaked var).

- [ ] **Step 6: Lint, commit, push, PR**

```bash
"$VENV_DIR/Scripts/python.exe" -m ruff check src/ tests/ 2>&1 | tail -3
git add -A
git commit -m "chore(types): re-verify torch type-ignores under mypy 2.2.0 + test isolation

- Re-verified the torch-stub type: ignore additions against trunk's mypy
  2.2.0 (#156); kept only those still required by warn_unused_ignores.
- test_loader_two_overlay_composition: monkeypatch.delenv the leaked
  MOUSEDROID_LLM__N_GPU_LAYERS so overlay-merge assertions are hermetic.
- test_tool_registry: patch cloud._auth.resolve_credentials to raise so
  diagnose_cloud is deterministic regardless of ambient ADC.
- test_e2e_5sec_run: llm.enabled=False in both Settings fixtures.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin chore/types-mypy22-and-test-isolation
gh pr create --base "$TRUNK_SHORT" --title "chore(types): re-verify torch type-ignores under mypy 2.2.0 + test isolation" --body "$(cat <<'EOF'
## What
Re-verifies the four torch type-ignore hunks against trunk's mypy 2.2.0
(only those still required survive) and lands three env-leak/ADC
test-isolation fixes.

## Why
The type-ignores were authored against mypy 2.1.0; #156 bumped trunk to
2.2.0, under which a stale ignore would FAIL via `warn_unused_ignores`.
The test fixes make three suites hermetic against a leaked
`MOUSEDROID_LLM__N_GPU_LAYERS` and ambient cloud credentials.

## Test
`mypy --strict` clean on `src/mousedroid`; the three touched suites green,
including a run with the env var deliberately set.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR-C opens.

---

## Task W6: Local trunk-sync v3 validation + static-analysis findings catalog

**Files:**
- Create: `$VENV_DIR/`
- Write: `$REPORT_ROOT/$STAMP/local/{deps,ci,ruff_check,ruff_format,mypy,numpy_notes}.log`, `$REPORT_ROOT/$STAMP/findings.md`, `SUMMARY.md`

**Interfaces:**
- Consumes: a worktree checkout (either trunk tip, or a temporary branch with all three PRs applied for a pre-merge dry run).
- Produces: the green/red stage matrix + the ruff/format/mypy/numpy catalog the user asked for.

- [ ] **Step 1: Create the venv (note: this must precede W3–W5 test steps; do it first in practice)**

```bash
cd "$WORKTREE_DIR"
python -m venv "$VENV_DIR"
export PY="$VENV_DIR/Scripts/python.exe"
"$PY" -m pip install --upgrade pip wheel setuptools 2>&1 | tail -2 | tee "$REPORT_ROOT/$STAMP/local/deps.log"
```

- [ ] **Step 2: Editable install with the FULL extras the CI uses**

```bash
"$PY" -m pip install -e ".[dev,telemetry,mcp]" 2>&1 | tee -a "$REPORT_ROOT/$STAMP/local/deps.log" | tail -20
"$PY" -c "import mousedroid; print('mousedroid from', mousedroid.__file__)"   # eyeball: path must be under $WORKTREE_DIR, NOT the primary tree
"$PY" -m ruff --version    # MUST print 'ruff 0.8.0'
"$PY" -m mypy --version    # expect 'mypy 2.2.0' (trunk #156)
```

Expected: `ruff 0.8.0` and `mypy 2.2.0` (if ruff differs, pyproject drift — STOP; the format comparison would be invalid). The printed `mousedroid.__file__` must start with `$WORKTREE_DIR` — if it points at the primary tree, a global editable install is shadowing it; re-create the venv with `PYTHONNOUSERSITE=1` set (per the `editable_install_worktree` memory).

- [ ] **Step 3: Full local CI via the canonical script**

```bash
export MOUSEDROID_PYTHON="$PY"; export MOUSEDROID_MOCK_HARDWARE=true; export PYTHONNOUSERSITE=1
set +e; bash scripts/ci.sh 2>&1 | tee "$REPORT_ROOT/$STAMP/local/ci.log"; CI_RC=${PIPESTATUS[0]}; set -e
echo "local_ci_rc=$CI_RC" | tee -a "$REPORT_ROOT/$STAMP/env.log"
awk '/^=== / {s=$0} END {print "last_stage=" s}' "$REPORT_ROOT/$STAMP/local/ci.log" | tee -a "$REPORT_ROOT/$STAMP/env.log"
```

Expected: `=== All checks passed ===`, `local_ci_rc=0`. If a stage fails, the last `===` marker names it.

- [ ] **Step 4: Static-analysis catalog (settles the 71-file format question with pinned ruff)**

```bash
set +e
"$PY" -m ruff check src/ tests/ tools/ --output-format=concise > "$REPORT_ROOT/$STAMP/local/ruff_check.log" 2>&1; echo "ruff_check_rc=$?" | tee -a "$REPORT_ROOT/$STAMP/env.log"
"$PY" -m ruff format --check src/ tests/ tools/ scripts/ > "$REPORT_ROOT/$STAMP/local/ruff_format.log" 2>&1; echo "ruff_format_rc=$?" | tee -a "$REPORT_ROOT/$STAMP/env.log"
"$PY" -m mypy src/mousedroid --strict --ignore-missing-imports > "$REPORT_ROOT/$STAMP/local/mypy.log" 2>&1; echo "mypy_rc=$?" | tee -a "$REPORT_ROOT/$STAMP/env.log"
set -e
grep -c "^Would reformat" "$REPORT_ROOT/$STAMP/local/ruff_format.log" || echo 0
tail -3 "$REPORT_ROOT/$STAMP/local/mypy.log"
```

Expected: with pinned ruff 0.8.0, `ruff_format_rc=0` and zero `Would reformat` lines (confirming the untracked `ruff_format_output.txt`'s "71 files" was a mismatched-version artifact). `mypy_rc=0`. If `ruff format --check` DOES report files, the debt is real — record it as a follow-up PR, do not mass-reformat here.

- [ ] **Step 5: numpy sweep (explicit user ask)**

```bash
{
  echo "# numpy findings — $STAMP"
  echo "## Deprecated bare aliases in src/ + tests/ (expect none)"; echo '```'
  grep -rnE 'np\.(bool|int|float|object|str|complex|long|unicode)[^0-9a-zA-Z_]' src/ tests/ || echo "(none — clean)"
  echo '```'
  echo "## Ruff NPY category enabled? (currently NO — noted as a follow-up)"; echo '```'
  grep -A2 '^\s*select' pyproject.toml | grep -o 'NPY' || echo "NPY not in select — NPY001/002/201 unenforced"
  echo '```'
} > "$REPORT_ROOT/$STAMP/local/numpy_notes.log"
cat "$REPORT_ROOT/$STAMP/local/numpy_notes.log"
```

Expected: `(none — clean)` and `NPY not in select`. Matches the recon finding.

- [ ] **Step 6: Assemble findings.md + SUMMARY.md**

```bash
{
  echo "# Trunk-sync v3 findings — $STAMP"
  echo "trunk_sha=$(git rev-parse HEAD)"; echo
  echo "| Tool | Exit code |"
  echo "| --- | --- |"
  for k in ruff_check_rc ruff_format_rc mypy_rc local_ci_rc; do
    echo "| $k | $(grep "^$k=" "$REPORT_ROOT/$STAMP/env.log" | cut -d= -f2) |"
  done
  echo; cat "$REPORT_ROOT/$STAMP/local/numpy_notes.log"
} > "$REPORT_ROOT/$STAMP/findings.md"
cp "$REPORT_ROOT/$STAMP/findings.md" "$REPORT_ROOT/$STAMP/SUMMARY.md"
cat "$REPORT_ROOT/$STAMP/findings.md"
```

Expected: a self-contained catalog. This is the workstation-side deliverable.

- [ ] **Step 7: Preflight trend gate (local baseline)**

```bash
JOURNAL="$REPORT_ROOT/harness_journal.jsonl"
"$PY" -m mousedroid.cli.preflight --journal-path "$JOURNAL" 2>&1 | tail -5
set +e; "$PY" -m mousedroid.cli.preflight --journal-path "$JOURNAL" --trend 2>&1 | tee "$REPORT_ROOT/$STAMP/local/preflight_trend.log"; TREND_RC=${PIPESTATUS[0]}; echo "preflight_trend_rc=$TREND_RC" | tee -a "$REPORT_ROOT/$STAMP/env.log"; set -e
```

Expected: rc=0. On the first run (<2 journal entries) rc=0 means `baseline (first run)`, not PASS.

---

## Task W7: Primary-tree cleanup (ONLY after all 3 PRs merge AND validation is green)

**Files:**
- Delete (primary tree, untracked): `MUJOCO_LOG.TXT`, `ruff_format_output.txt`, `sync.tar.gz`, `tools_sync.tar.gz`
- Restore (primary tree, tracked): the 11 modified files
- Delete: local + remote branch `refactor/onnx-default-providers-common`; the worktree

**Interfaces:**
- Consumes: confirmation that every uncommitted hunk is now upstream (via merged PRs A/B/C).
- Produces: a clean primary tree on trunk; no dangling worktrees/branches.

- [ ] **Step 1: Confirm all three PRs merged**

```bash
gh pr view fix/llm-gateway-valueerror-degradation --json state -q .state
gh pr view fix/jetson-full-validation-hardening --json state -q .state
gh pr view chore/types-mypy22-and-test-isolation --json state -q .state
```

Expected: three `MERGED`. If any is `OPEN`, STOP — cleanup is premature.

- [ ] **Step 2: Fetch trunk and prove every uncommitted hunk is now upstream**

```bash
cd /c/Users/iansh/OneDrive/Documents/Gronk-Droid-Jetson-Nano
git fetch --no-tags origin "$TRUNK_SHORT"
for f in src/mousedroid/llm_gateway/gateway.py scripts/jetson_full_validation.sh \
         tests/integration/test_e2e_5sec_run.py tests/unit/config/test_loader_two_overlay_composition.py \
         tests/unit/test_tool_registry.py; do
  echo "=== $f ==="
  git diff "$TRUNK_REF" -- "$f" | head -5   # empty ⇒ working copy already matches merged trunk
done
```

Expected: empty diffs for the files whose hunks landed (a non-empty diff means that file's change did NOT fully land — investigate before discarding). Type-ignore files may legitimately differ if mypy 2.2.0 dropped them.

- [ ] **Step 3: Delete the stale untracked artifacts**

```bash
rm -f MUJOCO_LOG.TXT ruff_format_output.txt sync.tar.gz tools_sync.tar.gz
git status --porcelain | grep -E '^\?\?' || echo "no stray untracked artifacts"
```

Expected: the 4 artifacts gone; the plan doc `docs/superpowers/plans/2026-07-11-...md` is now tracked-on-trunk (landed via PR-B) so it no longer shows as `??` after Step 4's checkout.

- [ ] **Step 4: Restore the tracked files, switch to trunk**

```bash
git checkout -- .
git checkout -B "$TRUNK_SHORT" "$TRUNK_REF"
git status --porcelain | wc -l   # expect 0
```

Expected: clean tree on trunk. (Only run this once every hunk is confirmed upstream — otherwise it discards unmerged work.)

- [ ] **Step 5: Delete the obsolete branch (local + remote) and prune the worktree**

```bash
git branch -D refactor/onnx-default-providers-common
git push origin --delete refactor/onnx-default-providers-common
git worktree remove "$WORKTREE_DIR" --force && git worktree prune
git worktree list
```

Expected: branch gone locally and remotely (its ONNX work already merged as PR #141); worktree list shows only the primary tree.

---


# PART 2 — Jetson: preserve WIP, deploy trunk tip, full smoke/e2e validation, rollback, handoff

**Conventions for every Part-2 task**

- Run all workstation commands in **Git Bash** (not PowerShell — PowerShell 5.1 `>` redirects write UTF-16 and corrupt diff files).
- One-time workstation session setup (re-run at the top of any resumed session):

```bash
export TRUNK=claude/markdown-implementation-plan-aVJ2l
export RUN_TAG=20260715
export WORK_ROOT="$HOME/mousedroid-trunk-sync-reports"
# Minimum ancestor SHA: trunk tip that carries #160 + #161 (OOM-guard + hardware gate)
export DEPLOY_BASELINE_SHA="${DEPLOY_BASELINE_SHA:-21463c3}"
# Health-check tunables (override for slower/faster hardware)
export HEALTH_POLL_RETRIES="${HEALTH_POLL_RETRIES:-40}"
export HEALTH_POLL_INTERVAL_S="${HEALTH_POLL_INTERVAL_S:-5}"
export HEALTH_HTTP_RETRIES="${HEALTH_HTTP_RETRIES:-10}"
export HEALTH_HTTP_INTERVAL_S="${HEALTH_HTTP_INTERVAL_S:-3}"
export HEALTH_HTTP_TIMEOUT_S="${HEALTH_HTTP_TIMEOUT_S:-5}"
mkdir -p "$WORK_ROOT/$RUN_TAG"
touch "$WORK_ROOT/$RUN_TAG/env.log"
```

- All rover commands go through `ssh jetson '...'`. Use `ssh -t jetson '...'` for anything with `sudo`. If any bare `docker` command fails with `permission denied ... docker.sock`, prefix with `sudo` and switch that step to `ssh -t`.
- Secrets (`ANTHROPIC_API_KEY`, `MOUSEDROID_TELEMETRY_TOKEN`) already live rover-side in `/etc/mousedroid/docker.env`. Never echo them, never pass on an SSH command line.
- `$DEPLOY_SHA` / `$ROLLBACK_SHA` are appended to `$WORK_ROOT/$RUN_TAG/env.log` as `KEY=VALUE` lines; restore in a new shell with `set -a; source "$WORK_ROOT/$RUN_TAG/env.log"; set +a`.

---

## Task J1: Preserve rover WIP on `rover/wip-20260715`

**Files:**
- Rover: `/opt/mousedroid` working tree (9 modified files, 764+/758− + untracked `features.schema.json`, `reports/endurance/*.json`)
- Workstation (write): `$WORK_ROOT/20260715/rover-wip/{rover_wip_uncommitted.diff,rover_wip_status.txt,rover_local_commits.txt,rover_wip_branch.diff}`, `env.log`

**Interfaces:**
- Consumes: rover at HEAD `7fdb9e3` with dirty tree (recon snapshot 2026-07-15).
- Produces: `ROLLBACK_SHA` (consumed by Tasks J2, J7) and the salvage-triage diff archive (consumed by the later rover-WIP salvage plan).

- [ ] **Step 1: Confirm the expected starting state (abort if it doesn't match)**

```bash
ssh jetson 'git -C /opt/mousedroid rev-parse --short HEAD && git -C /opt/mousedroid status --porcelain | wc -l'
```

Expected: `7fdb9e3` and a nonzero file count (~11). If HEAD is not `7fdb9e3`, STOP — the rover moved since recon; re-probe before proceeding.

- [ ] **Step 2: Archive the uncommitted diff to the workstation BEFORE touching git**

```bash
mkdir -p "$WORK_ROOT/$RUN_TAG/rover-wip"
ssh jetson 'git -C /opt/mousedroid diff --ignore-cr-at-eol' > "$WORK_ROOT/$RUN_TAG/rover-wip/rover_wip_uncommitted.diff"
ssh jetson 'git -C /opt/mousedroid status --porcelain'      > "$WORK_ROOT/$RUN_TAG/rover-wip/rover_wip_status.txt"
ssh jetson 'git -C /opt/mousedroid log --oneline -8'        > "$WORK_ROOT/$RUN_TAG/rover-wip/rover_local_commits.txt"
wc -l "$WORK_ROOT/$RUN_TAG/rover-wip/"*
```

Expected: `rover_wip_uncommitted.diff` non-empty (the `--ignore-cr-at-eol` view is the *substantive* subset of the 764-line churn — this is the salvage-triage input); status lists `jetson_csi.py`, both hardware test files, the script hunks, plus `??` lines.

- [ ] **Step 3: Untracked handling — gitignore check for endurance reports**

```bash
ssh jetson 'cd /opt/mousedroid && git check-ignore -v reports/endurance/*.json; echo "check_ignore_rc=$?"'
```

Expected: rc=0 (ignored → `git add -A` correctly skips them) or rc=1 (not ignored → they get committed). Either is fine — never `-f` force-add. `features.schema.json` is committed in both cases.

- [ ] **Step 4: Git identity guard**

```bash
ssh jetson 'git -C /opt/mousedroid config user.email || echo IDENTITY_UNSET'
```

Expected: an email or `IDENTITY_UNSET` (Step 5 carries inline `-c` identity flags, so no rover config change needed).

- [ ] **Step 5: Branch + commit everything**

```bash
ssh jetson 'cd /opt/mousedroid && git switch -c rover/wip-20260715 && git add -A && git -c user.name="Rover Operator" -c user.email="operator@example.com" commit -m "wip: rover-local state as of 2026-07-15 (pre trunk-sync checkpoint; CSI camera + hardware-test WIP)"'
```

Expected: `Switched to a new branch 'rover/wip-20260715'` then a commit summary listing ~9–11 files, 764+ insertions.

- [ ] **Step 6: Proof of preservation**

```bash
ssh jetson 'cd /opt/mousedroid && git log --oneline -1 && git status --porcelain && echo STATUS_EMPTY_OK'
```

Expected: one wip commit line, then `STATUS_EMPTY_OK` with no porcelain lines before it.

- [ ] **Step 7: Archive the full branch diff + record rollback SHA**

```bash
ssh jetson 'git -C /opt/mousedroid diff 7fdb9e3..rover/wip-20260715 --ignore-cr-at-eol' > "$WORK_ROOT/$RUN_TAG/rover-wip/rover_wip_branch.diff"
ROLLBACK_SHA=$(ssh jetson 'git -C /opt/mousedroid rev-parse rover/wip-20260715')
echo "ROLLBACK_SHA=$ROLLBACK_SHA" | tee -a "$WORK_ROOT/$RUN_TAG/env.log"
```

Expected: branch diff ≥ the uncommitted diff (adds `features.schema.json`); a 40-char `ROLLBACK_SHA` in env.log.

---

## Task J2: Bring the rover to trunk tip (detached) + layer PR-B hardening

**Files:**
- Rover: `/opt/mousedroid` (git only — no clean, ever)
- Workstation (write): `env.log` (`DEPLOY_SHA=`, `VALIDATION_SCRIPT_SOURCE=` lines)

**Interfaces:**
- Consumes: `ROLLBACK_SHA` recorded (J1). **Does NOT require the salvage PRs to be merged** — it needs only that PR-B's branch `fix/jetson-full-validation-hardening` is *pushed* (from W4).
- Produces: `DEPLOY_SHA` (consumed by J4, J6, J8); the rover's `jetson_full_validation.sh` carries PR-B's hardening.

- [ ] **Step 1: Fetch trunk on the rover**

```bash
ssh jetson "git -C /opt/mousedroid fetch origin $TRUNK"
```

Expected: fetch summary ending `... origin/claude/markdown-implementation-plan-aVJ2l`.
**Contingency (rover can't auth to GitHub) — bundle from the workstation:**

```bash
cd "$REPO_ROOT"
git fetch origin "$TRUNK"
git bundle create "$WORK_ROOT/$RUN_TAG/trunk.bundle" "origin/$TRUNK"
scp "$WORK_ROOT/$RUN_TAG/trunk.bundle" jetson:/tmp/trunk.bundle
ssh jetson "git -C /opt/mousedroid fetch /tmp/trunk.bundle refs/remotes/origin/$TRUNK:refs/heads/trunk-sync-20260715"
```

- [ ] **Step 2: Resolve DEPLOY_SHA at run time (never hardcode) and sanity-check it is at-or-after the current trunk baseline**

```bash
DEPLOY_SHA=$(ssh jetson "git -C /opt/mousedroid rev-parse origin/$TRUNK")
echo "DEPLOY_SHA=$DEPLOY_SHA" | tee -a "$WORK_ROOT/$RUN_TAG/env.log"
ssh jetson "git -C /opt/mousedroid merge-base --is-ancestor $DEPLOY_BASELINE_SHA $DEPLOY_SHA && echo ancestry_ok"
```

Expected: `ancestry_ok` (trunk tip is at or ahead of `$DEPLOY_BASELINE_SHA`, which already carries #160 + #161). If the salvage PRs have merged by now, they're simply included — but the deploy does **not** wait for them.

- [ ] **Step 3: Detached checkout (rover-local branches untouched)**

```bash
ssh jetson "git -C /opt/mousedroid checkout --detach $DEPLOY_SHA"
```

Expected: `HEAD is now at <sha7> ...`. Git removes wip-only tracked files (e.g. `features.schema.json`) from the worktree — safe, they live in the `rover/wip-20260715` commit. **NO `git clean` under any circumstances** (`/opt/mousedroid/venv`, reports, model weights live untracked in-tree).

- [ ] **Step 4: Layer PR-B's hardened validation script onto the detached checkout (from the pushed branch, no merge needed)**

```bash
if ssh jetson "git -C /opt/mousedroid fetch origin fix/jetson-full-validation-hardening 2>/dev/null && git -C /opt/mousedroid rev-parse origin/fix/jetson-full-validation-hardening" >/dev/null 2>&1; then
  # Take just the hardened script off the PR-B branch onto the detached HEAD (surgical, no full merge/rebase).
  ssh jetson "git -C /opt/mousedroid checkout origin/fix/jetson-full-validation-hardening -- scripts/jetson_full_validation.sh"
  echo "VALIDATION_SCRIPT_SOURCE=pr-b-branch" | tee -a "$WORK_ROOT/$RUN_TAG/env.log"
else
  echo "VALIDATION_SCRIPT_SOURCE=trunk-current (PR-B not pushed yet — expect known 07-12 WARN/FAIL noise)" | tee -a "$WORK_ROOT/$RUN_TAG/env.log"
fi
```

Expected: `VALIDATION_SCRIPT_SOURCE=pr-b-branch` and `git status` now shows `scripts/jetson_full_validation.sh` as a single staged/modified file on the detached HEAD (that's intentional — it's PR-B's hardening applied ahead of merge). If PR-B isn't pushed, the run proceeds on trunk's current script with louder-but-known noise. **Note:** once PR-B merges to trunk, a subsequent `git checkout --detach $DEPLOY_SHA` gets the hardening natively and this step becomes a no-op.

- [ ] **Step 5: Verify state (one intentional modified file if Step 4 layered PR-B)**

```bash
ssh jetson 'git -C /opt/mousedroid rev-parse HEAD' | grep -x "$DEPLOY_SHA" && echo HEAD_OK
ssh jetson 'git -C /opt/mousedroid status --porcelain' | tee "$WORK_ROOT/$RUN_TAG/post_checkout_status.txt" | wc -l
```

Expected: `HEAD_OK`, and `0` lines if Step 4 fell back to trunk's script, or exactly `1` line (`M scripts/jetson_full_validation.sh`) if Step 4 layered PR-B. Any OTHER modified file is unexpected EOL churn: `ssh jetson 'git -C /opt/mousedroid diff --ignore-cr-at-eol -- . ":(exclude)scripts/jetson_full_validation.sh" | head -5'` — empty means CR-only noise (note and continue); non-empty means STOP and triage.

---

## Task J3: Overlay sync + env presence checks

**Files:**
- Rover: `scripts/sync_jetson_overlay.sh`, `/etc/mousedroid/jetson_production.yaml` (+ extra overlay pairs), `/etc/mousedroid/docker.env` (read-count only)

**Interfaces:**
- Consumes: rover at `DEPLOY_SHA` (J2).
- Produces: drift-free overlays + confirmed secret presence, gating any container recreate.

- [ ] **Step 1: Sync overlays (repair mode), then strict verify**

```bash
ssh -t jetson 'sudo bash /opt/mousedroid/scripts/sync_jetson_overlay.sh; echo sync_rc=$?'
ssh -t jetson 'sudo bash /opt/mousedroid/scripts/sync_jetson_overlay.sh --verify; echo verify_rc=$?'
```

Expected: `sync_rc=0` (possibly with `overlay_sync_replaced` lines — drift-repair working), then `verify_rc=0` with no drift lines. `verify_rc=1` after a fresh sync → destination unwritable or source overlay missing — STOP and triage.

- [ ] **Step 2: Secrets and LLM knobs — presence/count only, never values**

```bash
ssh -t jetson 'sudo grep -c "^ANTHROPIC_API_KEY=" /etc/mousedroid/docker.env'
ssh -t jetson 'sudo grep -c "^MOUSEDROID_TELEMETRY_TOKEN=" /etc/mousedroid/docker.env'
ssh -t jetson 'sudo grep -c "^MOUSEDROID_LLM__ENABLED=true$" /etc/mousedroid/docker.env'
ssh -t jetson 'sudo grep -c "^MOUSEDROID_LLM__N_GPU_LAYERS=0$" /etc/mousedroid/docker.env'
```

Expected: each prints exactly `1`. Any `0` → STOP (a recreate would boot without the cloud tier or telemetry auth). Never `cat`/`less`/`echo` this file.

---

## Task J4: Conditional rebuild decision + execution

**Files:**
- Workstation repo: `pyproject.toml`, `Dockerfile.jetson`, `docker-compose.jetson.yml`, `deployments/jetson-image.json`
- Rover: `/opt/mousedroid/docker-compose.jetson.yml`, Docker image store

**Interfaces:**
- Consumes: `DEPLOY_SHA` (J2), image pin `032942b` (`deployments/jetson-image.json`).
- Produces: running container on the new source; Branch B additionally produces the `mousedroid:jetson-prev-20260715` rollback tag (consumed by J7) and a `jetson-image.json` re-pin PR (consumed by J8).

- [ ] **Step 1: Rebuild decision (workstation)**

```bash
cd /c/Users/iansh/OneDrive/Documents/Gronk-Droid-Jetson-Nano
git fetch origin "$TRUNK"
git diff --stat 032942b50ab71abf285282a4de7333193f208c38..$DEPLOY_SHA -- pyproject.toml Dockerfile.jetson docker-compose.jetson.yml | tee "$WORK_ROOT/$RUN_TAG/rebuild_decision.txt"
echo "REBUILD_REQUIRED=$([ -s "$WORK_ROOT/$RUN_TAG/rebuild_decision.txt" ] && echo yes || echo no)" | tee -a "$WORK_ROOT/$RUN_TAG/env.log"
```

Expected: empty → **Branch A**; any `--stat` lines → **Branch B**.

### Branch A — no rebuild (bind-mounted source; recreate only)

- [ ] **Step A1: Force-recreate the mousedroid service**

```bash
ssh jetson 'cd /opt/mousedroid && docker compose -f docker-compose.jetson.yml up -d --force-recreate mousedroid'
```

Expected: `Container mousedroid  Recreated` then `Started`. Monitoring stack (separate compose file) untouched.

- [ ] **Step A2: anthropic SDK contingency check (baked in image Stage 4b; must survive recreate)**

```bash
ssh jetson 'docker exec mousedroid python3 -c "import anthropic; print(anthropic.__version__)"'
```

Expected: a version like `0.105.2`. On `ModuleNotFoundError`: temporary in-container fix `docker exec mousedroid python3 -m pip install --no-deps "anthropic>=0.40,<1"` (evaporates on next recreate) and escalate to Branch B before Task J6 — a validation run must not depend on an ephemeral pip install.

### Branch B — rebuild required

- [ ] **Step B1: Disk guard (rover has 9.5 GB free at 83%)**

```bash
ssh jetson 'df -h / && docker system df'
```

Expected: ≥ 8 GiB free on `/`. If short: **OPERATOR-CONSENT step — do not run without explicit user approval in chat**: `docker rmi mousedroid:jetson-prev-20260513`. Caveat when asking: its 20.6 GB is mostly layers shared with `mousedroid:jetson`, so real reclaim may be small — check `docker system df -v` first. Never automatic.

- [ ] **Step B2: Rollback tag BEFORE building**

```bash
ssh jetson 'docker tag mousedroid:jetson mousedroid:jetson-prev-20260715 && docker images --format "{{.Repository}}:{{.Tag}} {{.ID}}" | grep mousedroid'
```

Expected: both tags pointing at the same image ID.

- [ ] **Step B3: Build with layer cache (NOT `docker_deploy.sh` — it uses `--no-cache`), survivable over SSH drop**

```bash
ssh jetson 'cd /opt/mousedroid && nohup docker compose -f docker-compose.jetson.yml build mousedroid > /tmp/jetson_build_20260715.log 2>&1 & echo build_started_pid=$!'
# Poll until done (repeat as needed):
ssh jetson 'tail -3 /tmp/jetson_build_20260715.log; pgrep -f "docker compose .* build" >/dev/null && echo BUILDING || echo BUILD_PROCESS_EXITED'
```

Expected: `CACHED` lines for most stages; completion = `BUILD_PROCESS_EXITED` with the log ending in a successful tag of `mousedroid:jetson`. Verify freshness: `docker images mousedroid:jetson --format "{{.ID}} {{.CreatedSince}}"` → "x minutes ago". Watch thermals: `ssh jetson 'tegrastats --interval 5000 | head -3'`.

- [ ] **Step B4: Bring the service up on the new image**

```bash
ssh jetson 'cd /opt/mousedroid && docker compose -f docker-compose.jetson.yml up -d mousedroid'
```

Expected: `Recreated`/`Started`.

- [ ] **Step B5 (workstation): jetson-image.json re-pin commit + PR (image-pin contract)**

Edit `deployments/jetson-image.json`: `"sha"` → full `$DEPLOY_SHA`, `"deployed_at"` → `date -u +%Y-%m-%dT%H:%M:%SZ`, note appended ("Rebuilt 2026-07-15 on the rover from $DEPLOY_SHA; prev image retagged mousedroid:jetson-prev-20260715"). Then:

```bash
cd /c/Users/iansh/OneDrive/Documents/Gronk-Droid-Jetson-Nano
git switch -c chore/jetson-image-pin-20260715 "origin/$TRUNK"
# (apply the JSON edit)
git add deployments/jetson-image.json
git commit -m "chore(deploy): re-pin jetson image to ${DEPLOY_SHA:0:7} (2026-07-15 rebuild)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin chore/jetson-image-pin-20260715
gh pr create --base "$TRUNK" --title "chore(deploy): re-pin jetson-image.json to ${DEPLOY_SHA:0:7}" --body "Image mousedroid:jetson rebuilt on-rover at \$DEPLOY_SHA on 2026-07-15. Config-compat gate now worktrees this SHA.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

Expected: PR opens; config-compat gate re-validates `config/*.yaml` against the new pin. (Skip on Branch A — the pin contract only binds on rebuild.)

---

## Task J5: Post-deploy health verification

**Files:** none written rover-side; workstation appends to `env.log`.

**Interfaces:**
- Consumes: recreated/rebuilt container (J4).
- Produces: healthy-gate between deploy and validation; failure triggers Task J7.

- [ ] **Step 1: Docker healthcheck poll (bounded — `$HEALTH_POLL_RETRIES` × `$HEALTH_POLL_INTERVAL_S` s ceiling)**

```bash
ssh jetson "for i in \$(seq 1 $HEALTH_POLL_RETRIES); do s=\$(docker inspect mousedroid --format '{{.State.Health.Status}}' 2>/dev/null || echo missing); echo \"poll \$i: \$s\"; [ \"\$s\" = \"healthy\" ] && exit 0; sleep $HEALTH_POLL_INTERVAL_S; done; echo HEALTH_TIMEOUT; exit 1"
```

Expected: a few `starting` polls then `healthy`, exit 0. `HEALTH_TIMEOUT` → Task J7.

- [ ] **Step 2: API health endpoint with retries**

```bash
ssh jetson "for i in \$(seq 1 $HEALTH_HTTP_RETRIES); do curl -fsS -m $HEALTH_HTTP_TIMEOUT_S http://127.0.0.1:8080/api/v1/health && echo && exit 0; sleep $HEALTH_HTTP_INTERVAL_S; done; echo HEALTH_HTTP_TIMEOUT; exit 1"
```

Expected: JSON health payload, exit 0.

- [ ] **Step 3: Startup-error scan (tolerating known ESP32 noise)**

```bash
ssh jetson 'docker logs --tail 200 mousedroid 2>&1 | grep -iE "error|exception|traceback" || echo NO_STARTUP_ERRORS'
```

Expected: `NO_STARTUP_ERRORS`, or only ESP32-serial error lines (dead ESP32 — tolerated). Any Python traceback or config-load failure → Task J7.

---

## Task J6: Full validation run + live log monitoring

**Files:**
- Rover: `scripts/jetson_full_validation.sh`, report tree `/opt/mousedroid/reports/jetson_full_validation/<STAMP_RUN>/`, `/tmp/jfv_20260715_console.log`, journal `reports/jetson_full_validation/harness_journal.jsonl`
- Workstation (write): `$WORK_ROOT/<STAMP_RUN>/jetson/` mirror

**Interfaces:**
- Consumes: healthy container (J5).
- Produces: SUMMARY.md verdict + trend baseline; green output is the precondition for Task J8.

- [ ] **Step 1 (terminal 1): Launch the full run — all phases, cache bypassed**

```bash
ssh -t jetson 'cd /opt/mousedroid && MOUSEDROID_VALIDATION_MISSION="navigate to the cantina" bash scripts/jetson_full_validation.sh --no-cache 2>&1 | tee /tmp/jfv_20260715_console.log; echo jfv_rc=${PIPESTATUS[0]}'
```

All other knobs at defaults (container `mousedroid`, telemetry `http://127.0.0.1:8080`, config `config/jetson_production.yaml`, venv `/opt/mousedroid/venv`, health retries 30, pytest timeout 120 s, log tail 2000). Secrets from `docker.env`/operator profile — nothing on the command line. Duration: tens of minutes (Phase 1 dominates). Expected end: `jfv_rc=0`.

- [ ] **Step 2 (terminal 2, started immediately after Step 1): curated live monitor with reattach loop**

`docker logs -f` dies whenever Phase 2 stops the container — the `while true` reattaches automatically:

```bash
ssh jetson 'while true; do docker logs -f --tail 0 mousedroid 2>&1; echo "[monitor] stream ended (phase-2 stop or recreate) — reattaching in 5s"; sleep 5; done' | grep --line-buffered -E 'esp32_serial_port_overridden|anthropic_gateway_slow|llm_gateway_degraded_model_error|usbc_endpoint_|power_chain_probe_complete|on_device_|error|exception|Traceback'
```

Secondary surfaces for mid-run triage: per-step logs under `reports/jetson_full_validation/<stamp>/`, Grafana/Loki on the rover (promtail ships container logs), auth-exempt `/metrics`.

- [ ] **Step 3: Read the verdict against the expected matrix**

**Must-PASS (blocking):** Phase 1 `static CI (ci.sh)`, `preflight (mock)`, `pillars (dry-run)`; Phase 2 `preflight (real)`, blocking smoke stages `usbc gpio camera lidar audio speaker voice pcie_ssd hailo`, hardware pytest tier, `validate_pillars`; Phase 3 health poll, `translate_mission` probe (runs with `-e MOUSEDROID_LLM__N_GPU_LAYERS=0`), `/metrics` scrape, lidar telemetry probe, structlog grep.

**Expected WARN (non-blocking, ESP32 functionally dead — do not chase):** smoke stages `system serial motor power`, `verify_sensors` ESP32 rows, `esp32_serial_port_overridden` monitor lines. NO MOTION is ever armed.

**Must be ABSENT:** any `on_device_*` event in the monitor (Phase 6 is default-OFF). If one appears → stop-and-triage config drift before accepting the run.

**Abort criteria (stop-and-triage, do not push on):**
- Phase 1 dies rc=137 → OOM despite #161's ulimit/slim-retry guard. Capture `ssh jetson 'free -h; dmesg | tail -30'`; do not immediately rerun.
- Blocking smoke FAIL on `camera` → CSI exclusive-use conflict: confirm the container actually stopped for Phase 2 (`docker ps`), check for stray nvargus clients, `ssh -t jetson 'sudo systemctl restart nvargus-daemon'`, then rerun just the cold phase: `--phases 2`.
- Script exits without writing `SUMMARY.md` → check the EXIT trap restarted the container (`docker ps`); if not: `ssh jetson 'docker start mousedroid'` FIRST, triage second.
- `translate_mission` FAIL with `anthropic_gateway_slow`/`llm_gateway_degraded_model_error` in the monitor → WAN dropout during the cloud probe; verify uplink, rerun warm phase only: `--phases 3`.

**Continue-with-note:** isolated `anthropic_gateway_slow` events on a PASSing probe; WARNs confined to the four ESP32-dependent stages.

- [ ] **Step 4: Pull the report tree to the workstation**

```bash
STAMP_RUN=$(ssh jetson 'ls -1t /opt/mousedroid/reports/jetson_full_validation | grep -E "^[0-9]{8}T[0-9]{6}Z$" | head -1'); echo "STAMP_RUN=$STAMP_RUN" | tee -a "$WORK_ROOT/$RUN_TAG/env.log"
mkdir -p "$WORK_ROOT/$STAMP_RUN/jetson"
scp -r "jetson:/opt/mousedroid/reports/jetson_full_validation/$STAMP_RUN/." "$WORK_ROOT/$STAMP_RUN/jetson/"
scp "jetson:/tmp/jfv_20260715_console.log" "$WORK_ROOT/$STAMP_RUN/jetson/console.log"
ls "$WORK_ROOT/$STAMP_RUN/jetson/SUMMARY.md" && grep -E '^(PASS|WARN|FAIL)' -c "$WORK_ROOT/$STAMP_RUN/jetson/SUMMARY.md" || true
```

Expected: `SUMMARY.md` present locally with the PASS/WARN/FAIL matrix.

- [ ] **Step 5: Trend gate — record + trend in a controlled cold window**

`preflight --journal-path --trend` always RE-RUNS the checks before trending (`src/mousedroid/cli/preflight.py:178-196`), and the warm container holds the CSI camera/lidar exclusively — so record in a brief cold window mirroring Phase 2's exact real-preflight env. **Use this identical env+config for every future trend record — apples-to-apples is what makes the trend gate meaningful.**

```bash
ssh jetson 'set -e; trap '\''docker start mousedroid >/dev/null 2>&1 || true'\'' EXIT; cd /opt/mousedroid; docker stop mousedroid; JOURNAL=reports/jetson_full_validation/harness_journal.jsonl; set +e; env MOUSEDROID_MICROPHONE__ENABLED=false MOUSEDROID_ESP32__ENABLED=false venv/bin/python -m mousedroid.cli.preflight --config config/jetson_production.yaml --journal-path "$JOURNAL" --trend; rc=$?; set -e; echo trend_rc=$rc; wc -l "$JOURNAL"; exit "$rc"'
```

Expected: preflight report, trend report, `trend_rc=0`, container restarted. **First-run semantics:** if `wc -l` shows < 2 journal entries, `trend_rc=0` means "no baseline available", NOT PASS — record the verdict as `baseline (first run)`. `trend_rc=1` = preflight FAIL or a real regression (status downgrade / new FAIL / latency creep past both slow-ratio and floor) → triage before Task J8.

- [ ] **Step 6: Confirm the container ended the day healthy**

Re-run Task J5 Steps 1–2 verbatim. Expected: `healthy` + health JSON. Final gate for declaring the deployment green.

---

## Task J7: Rollback contingency (only if J5 or J6 ends unhealthy/red and triage fails)

**Files:** rover `/opt/mousedroid` (git), Docker tags, `/etc/mousedroid` overlays.

**Interfaces:**
- Consumes: `ROLLBACK_SHA` (= `rover/wip-20260715` tip, J1) and, if Branch B ran, `mousedroid:jetson-prev-20260715` (J4).
- Produces: rover restored to pre-deploy state, healthy.

- [ ] **Step 1: Take the service down (named volume preserved — no `-v`)**

```bash
ssh jetson 'cd /opt/mousedroid && docker compose -f docker-compose.jetson.yml down'
```

Expected: `Container mousedroid Removed`. Monitoring stack (separate compose file) stays up.

- [ ] **Step 2: Restore the pre-sync source**

```bash
set -a; source "$WORK_ROOT/$RUN_TAG/env.log"; set +a
ssh jetson 'git -C /opt/mousedroid restore -- scripts/jetson_full_validation.sh'
ssh jetson "git -C /opt/mousedroid checkout rover/wip-20260715 && git -C /opt/mousedroid rev-parse HEAD" | grep -x "$ROLLBACK_SHA" && echo ROLLBACK_CHECKOUT_OK
```

Expected: `ROLLBACK_CHECKOUT_OK`.

- [ ] **Step 3 (Branch B only): restore the previous image under the compose tag**

```bash
ssh jetson 'docker tag mousedroid:jetson-prev-20260715 mousedroid:jetson && docker images --format "{{.Repository}}:{{.Tag}} {{.ID}}" | grep mousedroid'
```

Expected: `mousedroid:jetson` back on the pre-rebuild image ID. (Branch A: skip — image never changed.)

- [ ] **Step 4: Re-sync overlays for the rolled-back tree, bring it up, verify**

```bash
ssh -t jetson 'sudo bash /opt/mousedroid/scripts/sync_jetson_overlay.sh && sudo bash /opt/mousedroid/scripts/sync_jetson_overlay.sh --verify; echo verify_rc=$?'
ssh jetson 'cd /opt/mousedroid && docker compose -f docker-compose.jetson.yml up -d mousedroid'
```

Then re-run Task J5 Steps 1–3 verbatim. Expected: `healthy`. Record the rollback (what failed, at which task/step, logs pulled) in `$WORK_ROOT/$RUN_TAG/env.log` and STOP — the trunk-sync attempt gets re-planned, not retried blind.

---

## Task J8: Handoff (post-green)

**Files:** none modified in this plan — pointers only.

- [ ] **Step 1: Record the deployment.** Append to `$WORK_ROOT/$RUN_TAG/env.log`: final verdict, `DEPLOY_SHA`, `ROLLBACK_SHA`, `STAMP_RUN`, rebuild branch taken. Follow-up (separate commit): a dated entry in the repo's run record noting the rover now tracks `$DEPLOY_SHA` detached with WIP preserved on `rover/wip-20260715`.
- [ ] **Step 2 (Branch B only): land the `deployments/jetson-image.json` PR** from Task J4 Step B5 — merge only after the config-compat gate passes against the new pin.
- [ ] **Step 3: Phase 6 shadow-soak handoff (SEPARATE follow-up plan — do not enable here).** A green run is the entry criterion for enabling `on_device_learning` in **advisory mode** (`enabled=true`, `enable_hot_swap=false`) via a host-local overlay (the `MOUSEDROID_EXTRA_OVERLAYS` `src:dst` pair mechanism in `sync_jetson_overlay.sh`, per `docs/runbooks/jetson-on-device-learning.md`), followed by a ≥30-day soak watching `mousedroid_on_device_learning_reverted_total{reason}` in Prometheus/Grafana and the 30 Hz tick-latency budget, with the `on_device_*` structlog family in Loki. That plan defines its own abort thresholds (revert-counter mix, `regression_bound` spikes, `_tick_count` stalls). Nothing in the current plan flips any learning flag.
- [ ] **Step 4: Salvage triage pointer.** `$WORK_ROOT/20260715/rover-wip/rover_wip_branch.diff` (CR-noise-free) is the input for deciding which rover WIP hunks (CSI camera work, hardware-test edge cases) become upstream PRs.

---

## Open risks (Part 2)

1. **Disk (9.5 GB free, 83%)**: a Branch-B rebuild plus a new report tree can exhaust `/`. The tempting reclaim (`mousedroid:jetson-prev-20260513`) shares most layers with the live tag, so `docker rmi` may free far less than 20.6 GB — and it is an operator-consent step regardless. Check `docker system df -v` before promising space.
2. **Phase-1 OOM**: ~4.6 GB available RAM; #161's ulimit + slim-mode-retry guards it, but rc=137 remains the canonical abort signature — never loop-retry it.
3. **CSI camera exclusive-use (Phase 2 + trend gate)**: both require the container stopped; the validation EXIT trap restarts it, and the J6 Step 5 command restarts it inline — but an SSH drop mid-cold-window leaves the brain down. Always follow any cold-window failure with `docker ps` + `docker start mousedroid` before triage.
4. **Thermal during Branch-B build**: 20 GB image builds on an Orin Nano can throttle; build at idle (never concurrent with validation) and spot-check `tegrastats`.
5. **WAN dropout during cloud-LLM probes**: `translate_mission` is a blocking Phase-3 step reaching the real Anthropic gateway; `anthropic_gateway_slow`/`llm_gateway_degraded_model_error` + probe FAIL means verify uplink and rerun `--phases 3`, not a code regression.
6. **Trend-journal contamination**: the trend store re-runs preflight at record time; recording in a different context (warm container, mic enabled, ESP32 enabled) fabricates status downgrades that trip every future gate. The cold-window env in J6 Step 5 must be reused verbatim in all future runs.
7. **Rover→GitHub fetch auth**: if the rover's origin credentials lapsed, use the git-bundle contingency in J2 Step 1 — never rsync (`deploy_remote.sh`'s `--delete` would destroy rover-local state).
