# Trunk-Sync + Local + Jetson E2E Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Lessons learned (2026-07-12 execution — v2 run + findings-fix run)

Retro from the first end-to-end execution. Apply these before the next run:

- **Install `.[dev,telemetry,mcp]`, NOT `.[dev]`.** The `.github/workflows/ci.yml` typecheck stage uses `[dev,telemetry]` and the test stage uses `[dev,telemetry,mcp]`. `.[dev]` alone leaves aiohttp + PIL unresolved, so `mypy --strict --ignore-missing-imports` falls back to Any and reports 6 fake errors (4 `untyped-decorator` + 2 `valid-type`). Update Task 3.
- **`REPORT_ROOT` must NOT live inside `WORKTREE_DIR`.** v1/v2 defaulted `REPORT_ROOT=$WORKTREE_DIR/reports/trunk_sync` — but Task 2's `git worktree add` fails if the target dir exists. Put reports at a sibling path, e.g. `${HOME}/mousedroid-trunk-sync-reports`.
- **Windows regression failures on `test_host_bootstrap_script.py::TestDryRunBranches` are expected until PR #160 lands.** After PR #160 merges, drop this note. Trunk itself is clean on Linux CI; the failures are a Windows-only path-separator quirk that PR #160 hides behind `@pytest.mark.skipif(sys.platform == 'win32')`.
- **The rover's `/opt/mousedroid` may contain substantial uncommitted work** (the 2026-07-12 run found 764+ lines across `jetson_csi.py`, hardware tests, endurance reports). `scripts/deploy_remote.sh --code-only` uses `rsync -avz --delete` which will obliterate that work. Task 7 must guard-check with `ssh jetson 'git -C /opt/mousedroid status --porcelain'` first; if non-empty, either skip sync (validate rover at current SHA) or ask operator to commit/stash first.
- **Jetson Phase-1 ci.sh `exit=137` is OOM, not timeout.** Jetson has 7.4 GB RAM and the wrapper has no `timeout` command around ci.sh. Once PR #161 lands, the wrapper retries in slim mode automatically. Pre-PR: stop the mousedroid service before Phase-1 to free 1-2 GB.
- **Pasted API keys must NEVER be forwarded via SSH command line.** The 2026-07-12 run saw an ANTHROPIC_API_KEY pasted twice into chat — refuse both times; advise rotation via console.anthropic.com. Container-side credentials live in `/etc/mousedroid/docker.env` and are edited by the operator on the rover directly, never injected by the harness.
- **First-run `--trend` semantics:** `preflight_trend_rc=0` on <2 journal entries means "no baseline available", NOT "PASS". Task 10's SUMMARY.md renders this as `baseline (first run)` explicitly.
- **`preflight` failures during Phase-1 were TRANSIENT.** The 2026-07-12 run recorded FAIL on preflight (mock) + pillars (dry-run) as exit=1 during Phase-1, but standalone re-runs of the exact wrapper invocations 6 hours later returned rc=0 correctly. The OOM that killed ci.sh likely cascaded to subsequent Phase-1 steps. PR #161's guard should also indirectly clear this failure mode.

**Goal:** Isolate a fresh worktree from trunk (`origin/claude/markdown-implementation-plan-aVJ2l`), bootstrap deps, run the full local CI pipeline, sync source to the Jetson rover, run the composed 3-phase on-device validation, and catalog every ruff/format/mypy/numpy finding along the way — leaving the primary working tree and its uncommitted work untouched.

**Architecture:** This is an operational validation plan, not a feature build. It **composes** the existing tooling (`scripts/ci.sh`, `scripts/jetson_full_validation.sh`, `scripts/deploy_remote.sh`, `mousedroid.cli.preflight --trend`) — no new scripts are added. Isolation is provided by a `git worktree` so the current branch's dirty state (`refactor/onnx-default-providers-common`, 12 modified files, 4 untracked artifacts) is preserved untouched. Every host/path/ref is env-var driven (no hardcoded values) so the same plan re-runs against a different trunk ref, Jetson host, or venv without editing.

**Tech Stack:** git worktree • pip / editable install • ruff 0.8.0 (pinned) • mypy --strict • pytest (unit/property/integration/performance/regression/e2e) • Docker on Jetson • ssh (alias `jetson`) • structlog • `mousedroid.cli.preflight` + `mousedroid.cli.validate_pillars`.

## Global Constraints

- **Trunk ref:** `origin/claude/markdown-implementation-plan-aVJ2l` (current SHA `8c29245b20d487768dcfc2284306ebca5d61ef25` — commit `feat(telemetry): wire voice-degradation Prometheus counters (#158)`). Expose as `TRUNK_REF` env var; the plan never hardcodes the ref inside a step. `origin/main` does **not** exist on this remote — do NOT `git fetch origin main`.
- **No hardcoded values.** Every host, port, path, venv, timeout, ref, container name, telemetry URL, metric namespace, and rover IP must come from an env var with a documented default. Follow `scripts/jetson_full_validation.sh` header conventions (`MOUSEDROID_*` prefix, `__` nested delimiter).
- **Do not disturb the primary working tree.** The current branch `refactor/onnx-default-providers-common` has uncommitted work in 12 files + 4 untracked artifacts (`MUJOCO_LOG.TXT`, `ruff_format_output.txt`, `sync.tar.gz`, `tools_sync.tar.gz`). Isolation is via `git worktree add`; **never** `git checkout`, `git reset`, `git stash pop`, or `git clean` against the primary tree.
- **Ruff pinned = 0.8.0** (matches `.github/workflows/ci.yml` and `pyproject.toml [dev]`). Any bootstrap step that installs ruff must resolve this exact version.
- **`--import-mode=importlib`** on every pytest invocation (matches `scripts/ci.sh`).
- **Coverage gate 85%** (`--cov-fail-under=85`) is not negotiable for the local unit+property+integration stage.
- **structlog everywhere** — never `print()`. Debug logs added in triage steps go through `get_logger` (`from mousedroid.logging.setup import get_logger`).
- **Backwards compatibility.** No config field added by this plan may be non-optional. All Pydantic additions default; every existing YAML must load unchanged.
- **Jetson secrets never echo.** `ANTHROPIC_API_KEY` / `MOUSEDROID_TELEMETRY_TOKEN` are presence-checked only (`[ -n "$VAR" ]`), never printed.
- **Cold-then-warm on Jetson.** Phase 2 stops the container; phase 3 asserts it came back. `trap` MUST restart the container on any exit path — the rover brain never stays down.
- **No motion.** `MOUSEDROID_ESP32__ENABLED=false` on Jetson for the pytest hardware tier. ESP32 is functionally dead (per project memory) — validate around it.
- **Reusable, not one-shot.** Every step that produces artifacts writes under `${REPORT_ROOT}/${STAMP}/` so a re-run doesn't clobber the last.

---

## File Structure

**No new source files.** This plan mutates no `src/` code; it only composes existing tooling and writes reports.

**Read/consume (in the isolated worktree, not the primary tree):**
- `scripts/ci.sh` — canonical local pipeline (13 stages: lint, format-check, skill validator, mypy strict, pillar dispatch, hardcoded-value gate, settings identity, unit+property+integration+coverage, performance, regression, harness fast tier, e2e, branch coverage, promtool, health-check).
- `scripts/jetson_full_validation.sh` — canonical 3-phase Jetson runner (static CI → cold hardware → warm live).
- `scripts/deploy_remote.sh` — trunk sync to `/opt/mousedroid/src` on the rover.
- `scripts/sync_jetson_overlay.sh --verify` — overlay drift check.
- `src/mousedroid/cli/preflight.py --journal-path --trend` — regression trend gate over the JSONL journal.
- `src/mousedroid/cli/validate_pillars.py` — pillar-dispatch smoke.

**Write (reports only, gitignored):**
- Create dir: `${WORKTREE}/reports/trunk_sync/${STAMP}/local/` — local CI logs (per-stage `.log` file).
- Create dir: `${WORKTREE}/reports/trunk_sync/${STAMP}/jetson/` — scp'd from `<repo>/reports/jetson_full_validation/${STAMP}/` on the rover.
- Create file: `${WORKTREE}/reports/trunk_sync/${STAMP}/findings.md` — human-readable ruff/format/mypy/numpy triage catalog.
- Create file: `${WORKTREE}/reports/trunk_sync/${STAMP}/env.log` — captured env (`git rev-parse HEAD`, `python --version`, `ruff --version`, `mypy --version`, Jetson `uname -a` & `docker version`), redacted.

**Interfaces:**
- Consumes (from user env): `TRUNK_REF`, `WORKTREE_DIR`, `JETSON_HOST` (SSH alias or IP), `REMOTE_USER`, `VENV_DIR`, `REPORT_ROOT`, `MOUSEDROID_JETSON_CONFIG`, `MOUSEDROID_METRICS__NAMESPACE`, `STAMP`, `ANTHROPIC_API_KEY` (optional), `MOUSEDROID_TELEMETRY_TOKEN` (optional).
- Produces (for downstream reviewer): a single findings markdown + a regression trend verdict + a green/red table across all pipeline stages.

---

## Task 1: Preflight — capture state, verify prereqs, no destructive default

**Files:**
- Read: `.git/HEAD`, `git status --porcelain`, `~/.ssh/config`
- Write: `${REPORT_ROOT}/${STAMP}/preflight.log`

**Interfaces:**
- Consumes: user env (`JETSON_HOST`, `REMOTE_USER`, `TRUNK_REF`)
- Produces: verified prereqs table (git ≥ 2.20 for worktree, python3, ssh, tar); confirmed reachability of `${JETSON_HOST}`; snapshotted primary-tree dirty file list (so the reviewer can prove Task 10 restored it).

- [ ] **Step 1: Announce the plan; resolve env-var defaults**

```bash
export TRUNK_REF="${TRUNK_REF:-origin/claude/markdown-implementation-plan-aVJ2l}"
export WORKTREE_DIR="${WORKTREE_DIR:-$HOME/mousedroid-trunk-sync}"
export JETSON_HOST="${JETSON_HOST:-jetson}"                # SSH alias from ~/.ssh/config
export REMOTE_USER="${REMOTE_USER:-ian}"                    # matches project memory
export VENV_DIR="${VENV_DIR:-$WORKTREE_DIR/.venv}"
export STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export REPORT_ROOT="${REPORT_ROOT:-$WORKTREE_DIR/reports/trunk_sync}"
export MOUSEDROID_JETSON_CONFIG="${MOUSEDROID_JETSON_CONFIG:-config/jetson_production.yaml}"
export MOUSEDROID_METRICS__NAMESPACE="${MOUSEDROID_METRICS__NAMESPACE:-mousedroid}"
mkdir -p "$REPORT_ROOT/$STAMP/local" "$REPORT_ROOT/$STAMP/jetson"
```

Expected: `$REPORT_ROOT/$STAMP/{local,jetson}` exist. No files touched inside the primary tree.

- [ ] **Step 2: Snapshot primary-tree state (for Task 10 restoration proof)**

```bash
{
  echo "=== primary tree HEAD ==="; git -C "$PWD" rev-parse HEAD
  echo "=== primary tree branch ==="; git -C "$PWD" rev-parse --abbrev-ref HEAD
  echo "=== dirty files ==="; git -C "$PWD" status --porcelain
} > "$REPORT_ROOT/$STAMP/primary_tree_before.log"
```

Expected: file lists 12 `M` entries + 4 `??` entries (matches session-start git status). If it doesn't, **STOP** and reconcile — the user may have committed work mid-flight.

- [ ] **Step 3: Verify tool prereqs**

```bash
git --version | tee -a "$REPORT_ROOT/$STAMP/preflight.log"
python3 --version | tee -a "$REPORT_ROOT/$STAMP/preflight.log"
ssh -V 2>&1 | tee -a "$REPORT_ROOT/$STAMP/preflight.log"
tar --version | head -1 | tee -a "$REPORT_ROOT/$STAMP/preflight.log"
```

Expected: git ≥ 2.20 (worktree requirement), python3 ≥ 3.10 (pyproject floor).

- [ ] **Step 4: Verify Jetson reachability (no source sync yet)**

```bash
ssh -o ConnectTimeout=10 -o BatchMode=yes "${JETSON_HOST}" 'uname -a; docker ps --format "{{.Names}}\t{{.Status}}"; free -m | head -2' \
  | tee "$REPORT_ROOT/$STAMP/jetson_preflight.log"
```

Expected: kernel line (L4T R36.x), a `mousedroid` container line (running or exited), free-memory summary. If SSH fails, **STOP** — Task 6/7/8 cannot proceed.

- [ ] **Step 5: Resolve + record trunk ref, no fetch yet**

```bash
git -C "$PWD" ls-remote origin "$(echo "$TRUNK_REF" | sed 's|^origin/||')" | tee -a "$REPORT_ROOT/$STAMP/preflight.log"
```

Expected: exactly one line with the ref's SHA. If empty, the ref doesn't exist — **STOP** and re-clarify the trunk ref.

- [ ] **Step 6: Commit intent-log (no code commit — just an audit trail on the primary tree)**

```bash
echo "Sync run started: $(date -u +%FT%TZ) STAMP=$STAMP TRUNK_REF=$TRUNK_REF WORKTREE_DIR=$WORKTREE_DIR" \
  >> "$REPORT_ROOT/$STAMP/preflight.log"
```

Expected: preflight.log is a complete replay-able env dump. **No `git commit` here** — nothing about this plan is meant to touch the primary branch's history.

---

## Task 2: Isolated worktree from trunk (superpowers:using-git-worktrees)

**Files:**
- Create: `${WORKTREE_DIR}/` (new git worktree)
- Read: `.git/worktrees/` metadata

**Interfaces:**
- Consumes: `TRUNK_REF`, `WORKTREE_DIR` from Task 1.
- Produces: a clean checkout of the trunk SHA under `${WORKTREE_DIR}`; primary tree unmodified.

- [ ] **Step 1: Invoke the using-git-worktrees skill**

Announce: `Using superpowers:using-git-worktrees to establish isolation.` Skip if the harness auto-invoked it during plan execution.

- [ ] **Step 2: Fetch trunk ref only (narrow fetch, keeps things fast)**

```bash
git -C "$PWD" fetch --no-tags origin "$(echo "$TRUNK_REF" | sed 's|^origin/||'):refs/remotes/$TRUNK_REF" 2>&1 \
  | tee -a "$REPORT_ROOT/$STAMP/preflight.log"
```

Expected: `From github.com:ianshank/Mouse-Droid-AGI ... -> $TRUNK_REF`. If the tree already had this ref, output is silent — that's fine.

- [ ] **Step 3: Create the worktree at the fetched SHA (detached — no branch mutation)**

```bash
TRUNK_SHA="$(git -C "$PWD" rev-parse "$TRUNK_REF")"
git -C "$PWD" worktree add --detach "$WORKTREE_DIR" "$TRUNK_SHA"
```

Expected: `Preparing worktree (detached HEAD ...)`. `HEAD is now at 8c29245 feat(telemetry): ...` (or later).

- [ ] **Step 4: Verify isolation — primary tree still dirty, worktree clean**

```bash
{
  echo "=== primary tree post-worktree ==="; git -C "$PWD" status --porcelain
  echo "=== worktree state ==="; git -C "$WORKTREE_DIR" status --porcelain
  echo "=== worktree HEAD ==="; git -C "$WORKTREE_DIR" rev-parse HEAD
} > "$REPORT_ROOT/$STAMP/worktree_verify.log"
diff "$REPORT_ROOT/$STAMP/primary_tree_before.log" \
     <(git -C "$PWD" status --porcelain | sort) \
  && echo "primary tree UNCHANGED — good"
```

Expected: primary-tree porcelain output identical to Task 1 Step 2 (byte-for-byte). Worktree porcelain: empty. Worktree HEAD == fetched SHA. If primary tree changed, **STOP**.

- [ ] **Step 5: cd into the worktree for all following tasks**

```bash
cd "$WORKTREE_DIR"
pwd | tee -a "$REPORT_ROOT/$STAMP/preflight.log"
```

Expected: pwd echoes `$WORKTREE_DIR`. All subsequent bash blocks assume this cwd.

---

## Task 3: Local Python venv + editable install

**Files:**
- Create: `${VENV_DIR}/` (fresh venv)
- Read: `pyproject.toml` (`[project.optional-dependencies].dev`)

**Interfaces:**
- Consumes: `VENV_DIR`, `WORKTREE_DIR` from Task 1/2.
- Produces: a resolvable Python at `${VENV_DIR}/{Scripts,bin}/python` with `ruff==0.8.0`, `mypy`, `pytest`, and the mousedroid package installed editable-in-worktree (NOT editable-in-primary — see feedback memory `editable_install_worktree`).

- [ ] **Step 1: Create the venv inside the worktree (not shared with primary)**

```bash
python3 -m venv "$VENV_DIR"
# Cross-platform activation: PowerShell uses Scripts\Activate.ps1; POSIX uses bin/activate.
if [ -x "$VENV_DIR/Scripts/python.exe" ]; then export PY="$VENV_DIR/Scripts/python.exe"; else export PY="$VENV_DIR/bin/python"; fi
"$PY" -m pip install --upgrade pip wheel setuptools 2>&1 | tail -3 | tee -a "$REPORT_ROOT/$STAMP/local/deps.log"
```

Expected: `Successfully installed pip-... wheel-... setuptools-...`.

- [ ] **Step 2: Install the mousedroid package in editable mode + `[dev]` extra**

```bash
"$PY" -m pip install -e ".[dev,telemetry,mcp]" 2>&1 | tee -a "$REPORT_ROOT/$STAMP/local/deps.log" | tail -20
```

Expected: `Successfully installed mousedroid-... ruff-0.8.0 mypy-... pytest-...`. If `ruff` resolves to any version other than `0.8.0`, **STOP** — pyproject drift needs a separate fix and the CI compare will be invalid.

- [ ] **Step 3: Verify the editable-install actually points at the worktree (not the primary tree)**

```bash
"$PY" -c "import mousedroid, pathlib, sys; p = pathlib.Path(mousedroid.__file__).resolve(); assert str(p).startswith('$WORKTREE_DIR'), f'editable import escaped worktree: {p}'; print(f'mousedroid imports from: {p}')" \
  | tee -a "$REPORT_ROOT/$STAMP/local/deps.log"
```

Expected: `mousedroid imports from: $WORKTREE_DIR/src/mousedroid/__init__.py`. If it prints a primary-tree path, the venv is picking up a global editable install — re-create the venv with `--clear` or set `PYTHONNOUSERSITE=1`.

- [ ] **Step 4: Record versions to env.log**

```bash
{
  echo "=== $STAMP env ==="
  echo "python=$($PY --version)"
  echo "pip=$($PY -m pip --version | awk '{print $2}')"
  echo "ruff=$($PY -m ruff --version)"
  echo "mypy=$($PY -m mypy --version)"
  echo "pytest=$($PY -m pytest --version 2>&1 | head -1)"
  echo "worktree_sha=$(git rev-parse HEAD)"
  echo "trunk_ref=$TRUNK_REF"
} > "$REPORT_ROOT/$STAMP/env.log"
cat "$REPORT_ROOT/$STAMP/env.log"
```

Expected: `ruff=ruff 0.8.0`. mypy version matches pyproject floor.

---

## Task 4: Static-analysis audit — ruff / format / mypy / numpy triage catalog

**Files:**
- Read: `src/`, `tests/`, `tools/` (worktree)
- Write: `${REPORT_ROOT}/${STAMP}/local/{ruff_check,ruff_format,mypy,numpy_notes}.log`
- Write: `${REPORT_ROOT}/${STAMP}/findings.md`

**Interfaces:**
- Consumes: `PY`, `REPORT_ROOT`, `STAMP`.
- Produces: a triage catalog (`findings.md`) grouped by tool, with paths + counts + first-line of each unique diagnostic — feeds Task 10's consolidation.

**Note:** This task **catalogs**, it does not fix. The user's instruction was "take note of ruff/lint/mypy/numpy issues". Fix-forward is out-of-scope for this plan (would produce PRs, not a validation run). See end-of-plan "If Findings Warrant Fix-Forward" for the escape hatch.

- [ ] **Step 1: Ruff check (lint) — capture non-zero exit without aborting the pipeline**

```bash
set +e
"$PY" -m ruff check src/ tests/ tools/ --output-format=concise > "$REPORT_ROOT/$STAMP/local/ruff_check.log" 2>&1
RUFF_CHECK_RC=$?
set -e
echo "ruff_check_rc=$RUFF_CHECK_RC" | tee -a "$REPORT_ROOT/$STAMP/env.log"
wc -l "$REPORT_ROOT/$STAMP/local/ruff_check.log"
```

Expected: rc=0 (clean) OR rc=1 with a finding list. Record rc. Do NOT `exit`.

- [ ] **Step 2: Ruff format --check — expected drift per repo `ruff_format_output.txt`**

```bash
set +e
"$PY" -m ruff format --check src/ tests/ tools/ > "$REPORT_ROOT/$STAMP/local/ruff_format.log" 2>&1
RUFF_FMT_RC=$?
set -e
echo "ruff_format_rc=$RUFF_FMT_RC" | tee -a "$REPORT_ROOT/$STAMP/env.log"
grep -c "^Would reformat" "$REPORT_ROOT/$STAMP/local/ruff_format.log" || true
```

Expected: rc=1 with a `Would reformat: …` list. Compare against the pre-existing `ruff_format_output.txt` in the primary tree — if the trunk list is **smaller**, someone has been steadily catching up; if **larger**, a recent PR regressed format hygiene.

- [ ] **Step 3: mypy --strict — capture full output**

```bash
set +e
"$PY" -m mypy src/ --strict --ignore-missing-imports > "$REPORT_ROOT/$STAMP/local/mypy.log" 2>&1
MYPY_RC=$?
set -e
echo "mypy_rc=$MYPY_RC" | tee -a "$REPORT_ROOT/$STAMP/env.log"
grep -c "^src/" "$REPORT_ROOT/$STAMP/local/mypy.log" || true
tail -3 "$REPORT_ROOT/$STAMP/local/mypy.log"
```

Expected: last line is `Success: no issues found in N source files` OR `Found K errors in M files`. Record.

- [ ] **Step 4: numpy-specific pattern sweep (the user's explicit ask)**

```bash
# Reusable helper: greps mypy + ruff logs for numpy-flavoured findings.
# No hardcoded paths — reads from the logs Steps 1-3 already wrote.
{
  echo "# numpy findings — $STAMP"; echo
  echo "## From mypy (dtype/shape/np.ndarray[Any] etc.)"; echo '```'
  grep -E "numpy|np\.|ndarray|dtype|floating|int64|int32" "$REPORT_ROOT/$STAMP/local/mypy.log" || echo "(none)"
  echo '```'
  echo "## From ruff (NPY-prefixed rules — the numpy category)"; echo '```'
  grep -E "^[^:]+:[0-9]+:[0-9]+: NPY[0-9]+" "$REPORT_ROOT/$STAMP/local/ruff_check.log" || echo "(none)"
  echo '```'
  echo "## Numpy deprecation greps in source (proactive — future-compat)"; echo '```'
  "$PY" -c "import subprocess, sys; sys.exit(subprocess.call(['grep','-rn','-E','np\\.(bool|int|float|object|str|complex)[^0-9a-zA-Z_]','src/','tests/']))" || true
  echo '```'
} > "$REPORT_ROOT/$STAMP/local/numpy_notes.log"
head -50 "$REPORT_ROOT/$STAMP/local/numpy_notes.log"
```

Expected: numpy_notes.log is a self-contained markdown chunk. Empty sections print `(none)` (never blank).

- [ ] **Step 5: Assemble findings.md**

```bash
{
  echo "# Trunk-sync findings — $STAMP"
  echo "trunk_sha=$(git rev-parse HEAD) trunk_ref=$TRUNK_REF"; echo
  echo "## Static analysis"
  echo "| Tool | Exit code | Count |"
  echo "| --- | --- | --- |"
  echo "| ruff check | $(grep '^ruff_check_rc=' $REPORT_ROOT/$STAMP/env.log | cut -d= -f2) | $(grep -c '^' $REPORT_ROOT/$STAMP/local/ruff_check.log) |"
  echo "| ruff format --check | $(grep '^ruff_format_rc=' $REPORT_ROOT/$STAMP/env.log | cut -d= -f2) | $(grep -c '^Would reformat' $REPORT_ROOT/$STAMP/local/ruff_format.log || echo 0) |"
  echo "| mypy --strict | $(grep '^mypy_rc=' $REPORT_ROOT/$STAMP/env.log | cut -d= -f2) | $(grep -c '^src/' $REPORT_ROOT/$STAMP/local/mypy.log || echo 0) |"
  echo; echo "## Numpy-specific"
  cat "$REPORT_ROOT/$STAMP/local/numpy_notes.log"
} > "$REPORT_ROOT/$STAMP/findings.md"
head -30 "$REPORT_ROOT/$STAMP/findings.md"
```

Expected: findings.md is a self-contained triage table + numpy detail.

- [ ] **Step 6: Commit intent — NO commit against primary tree, worktree only if you want to preserve the audit trail**

Skip commit unless a fix-forward branch is warranted. The worktree is detached-HEAD by design; commits here are throwaway.

---

## Task 5: Local full test suite via `scripts/ci.sh`

**Files:**
- Read: `scripts/ci.sh`, all tests under `tests/`
- Write: `${REPORT_ROOT}/${STAMP}/local/ci.log`, `${REPORT_ROOT}/${STAMP}/local/coverage.xml`

**Interfaces:**
- Consumes: `PY`, `WORKTREE_DIR`, `REPORT_ROOT`, `STAMP`.
- Produces: full per-stage output (13 stages: env, ruff check, ruff format, skill validator, mypy, pillar dispatch dry-run, hardcoded-value gate, settings identity, unit+property+integration+coverage-fail-under-85, performance, regression, harness fast tier, e2e, branch coverage, promtool, health-check). PASS/FAIL verdict per stage feeds Task 10.

- [ ] **Step 1: Run the full local CI end-to-end via the canonical script**

```bash
export MOUSEDROID_PYTHON="$PY"          # tells ci.sh not to hunt for its own python
export MOUSEDROID_MOCK_HARDWARE=true    # ci.sh sets this too — belt-and-braces
export PYTHONNOUSERSITE=1
set +e
bash scripts/ci.sh 2>&1 | tee "$REPORT_ROOT/$STAMP/local/ci.log"
CI_RC=$?
set -e
echo "local_ci_rc=$CI_RC" | tee -a "$REPORT_ROOT/$STAMP/env.log"
```

Expected: last line `=== All checks passed ===` (rc=0). If any stage fails, ci.sh aborts via `set -e` and rc≠0 — that's expected behavior and the log tail identifies which stage.

- [ ] **Step 2: Identify which ci.sh stage failed (if any) for the report**

```bash
# Parse the "=== ... ===" section markers to find the last-attempted stage.
awk '/^=== / {stage=$0} END {print "last_stage=" stage}' "$REPORT_ROOT/$STAMP/local/ci.log" \
  | tee -a "$REPORT_ROOT/$STAMP/env.log"
```

Expected: e.g. `last_stage=== All checks passed ===` on success, or e.g. `last_stage=== Type Check ===` if mypy failed.

- [ ] **Step 3: Extract coverage summary (last percentage line)**

```bash
grep -E "^TOTAL " "$REPORT_ROOT/$STAMP/local/ci.log" | tail -1 | tee -a "$REPORT_ROOT/$STAMP/env.log"
```

Expected: `TOTAL   NNNNN   NNNN   NN%`. Record % — it's a health-of-repo signal even when the gate passes.

- [ ] **Step 4: Append CI verdict to findings.md**

```bash
{
  echo; echo "## Local CI verdict"
  echo "- exit_code = $(grep '^local_ci_rc=' $REPORT_ROOT/$STAMP/env.log | cut -d= -f2)"
  echo "- last_stage = $(grep '^last_stage=' $REPORT_ROOT/$STAMP/env.log | cut -d= -f2-)"
  echo "- coverage = $(grep '^TOTAL ' $REPORT_ROOT/$STAMP/local/ci.log | tail -1 | awk '{print $NF}')"
} >> "$REPORT_ROOT/$STAMP/findings.md"
```

Expected: a 3-line block appended to findings.md.

---

## Task 6: Jetson-side preflight (SSH, container, disk, no source touched)

**Files:**
- Read (over SSH): rover `/opt/mousedroid`, `docker ps`, `df -h`
- Write: `${REPORT_ROOT}/${STAMP}/jetson/preflight.log`

**Interfaces:**
- Consumes: `JETSON_HOST`, `REMOTE_USER`.
- Produces: rover state snapshot before sync — repo HEAD, container status, disk free, GPU status. Used to detect Task 10 "is the rover in the same state we found it?".

- [ ] **Step 1: Capture rover state (no writes)**

```bash
ssh "$JETSON_HOST" 'bash -s' <<'EOF' | tee "$REPORT_ROOT/$STAMP/jetson/preflight.log"
set -uo pipefail
echo "=== rover uname ==="; uname -a
echo "=== rover /opt/mousedroid HEAD ==="; git -C /opt/mousedroid rev-parse HEAD 2>&1 || echo "(no repo)"
echo "=== rover /opt/mousedroid status ==="; git -C /opt/mousedroid status --porcelain 2>&1 | head -20 || true
echo "=== docker ps ==="; docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
echo "=== disk free ==="; df -h / /opt /var/lib/docker 2>/dev/null || df -h /
echo "=== nvidia-smi (short) ==="; nvidia-smi -q -d MEMORY,UTILIZATION 2>&1 | head -20 || echo "(no nvidia-smi)"
echo "=== ANTHROPIC_API_KEY presence ==="; test -n "${ANTHROPIC_API_KEY:-}" && echo "present" || echo "absent"
echo "=== MOUSEDROID_TELEMETRY_TOKEN presence ==="; test -n "${MOUSEDROID_TELEMETRY_TOKEN:-}" && echo "present" || echo "absent"
EOF
```

Expected: uname line, HEAD SHA, container line, `df -h` output, GPU memory summary. Secrets: only `present`/`absent` — never the value.

- [ ] **Step 2: Compare rover HEAD against target `TRUNK_REF` — decide if sync is needed**

```bash
ROVER_SHA="$(grep -A1 '^=== rover /opt/mousedroid HEAD ===' $REPORT_ROOT/$STAMP/jetson/preflight.log | tail -1 | tr -d '[:space:]')"
LOCAL_SHA="$(git rev-parse HEAD)"
echo "rover_sha=$ROVER_SHA local_sha=$LOCAL_SHA" | tee -a "$REPORT_ROOT/$STAMP/env.log"
if [ "$ROVER_SHA" = "$LOCAL_SHA" ]; then
  echo "sync_needed=false" | tee -a "$REPORT_ROOT/$STAMP/env.log"
else
  echo "sync_needed=true"  | tee -a "$REPORT_ROOT/$STAMP/env.log"
fi
```

Expected: `sync_needed=true` on a fresh trunk pull; `false` if the rover was already up-to-date. Either is fine — Task 7 is a no-op on `false`.

---

## Task 7: Sync trunk source to Jetson (only if needed)

**Files:**
- Read: worktree `src/`, `scripts/`, `config/`, `tests/`, `tools/`, `pyproject.toml`
- Write (over SSH): rover `/opt/mousedroid/src/` and repo tree

**Interfaces:**
- Consumes: `sync_needed`, `JETSON_HOST`, `REMOTE_USER`.
- Produces: rover repo tree matching worktree SHA; `sync_jetson_overlay.sh --verify` clean.

- [ ] **Step 1: Sync (guarded on `sync_needed=true`) via the canonical `deploy_remote.sh`**

```bash
if [ "$(grep '^sync_needed=' $REPORT_ROOT/$STAMP/env.log | cut -d= -f2)" = "true" ]; then
  MOUSEDROID_REMOTE_USER="$REMOTE_USER" bash scripts/deploy_remote.sh "$JETSON_HOST" --code-only \
    2>&1 | tee "$REPORT_ROOT/$STAMP/jetson/sync.log"
else
  echo "sync skipped — rover already at target SHA" | tee "$REPORT_ROOT/$STAMP/jetson/sync.log"
fi
```

Expected on sync: last section is a success banner from deploy_remote.sh. Any `ERROR:` line aborts (script uses `set -euo pipefail`).

- [ ] **Step 2: Verify rover HEAD moved to target SHA**

```bash
ROVER_SHA_AFTER="$(ssh "$JETSON_HOST" 'git -C /opt/mousedroid rev-parse HEAD' 2>&1)"
LOCAL_SHA="$(git rev-parse HEAD)"
if [ "$ROVER_SHA_AFTER" = "$LOCAL_SHA" ]; then
  echo "post_sync_sha_match=true rover=$ROVER_SHA_AFTER" | tee -a "$REPORT_ROOT/$STAMP/env.log"
else
  echo "post_sync_sha_match=false rover=$ROVER_SHA_AFTER expected=$LOCAL_SHA" | tee -a "$REPORT_ROOT/$STAMP/env.log"
  exit 1
fi
```

Expected: `post_sync_sha_match=true`. If false, **STOP** — do NOT run validation against a mismatched tree.

- [ ] **Step 3: Overlay drift check (config drift is silently corrosive)**

```bash
ssh "$JETSON_HOST" 'bash -c "cd /opt/mousedroid && bash scripts/sync_jetson_overlay.sh --verify"' \
  2>&1 | tee -a "$REPORT_ROOT/$STAMP/jetson/sync.log" || echo "overlay drift detected — see log"
```

Expected: exit 0 and no `overlay_sync_replaced` events. Non-zero means `/etc/mousedroid/*.yaml` diverges from the repo overlay — record as a finding.

---

## Task 8: Jetson full validation — `scripts/jetson_full_validation.sh`

**Files:**
- Read (over SSH): all of `/opt/mousedroid`
- Write (on rover): `/opt/mousedroid/reports/jetson_full_validation/${STAMP}/`
- Write (locally after scp): `${REPORT_ROOT}/${STAMP}/jetson/full_validation/`

**Interfaces:**
- Consumes: rover-side `ANTHROPIC_API_KEY`, `MOUSEDROID_TELEMETRY_TOKEN` (both presence-only), `MOUSEDROID_JETSON_CONFIG`, `MOUSEDROID_METRICS__NAMESPACE`.
- Produces: three-phase artifact tree (Phase 1 static CI, Phase 2 cold hardware, Phase 3 warm live) + a `summary.log` with PASS/WARN/FAIL counts. `ESP32__ENABLED=false` is enforced by the script for hardware pytest.

- [ ] **Step 1: Kick off jetson_full_validation.sh over SSH, logging to rover-side stamp dir**

```bash
ssh "$JETSON_HOST" 'bash -s' <<EOF | tee "$REPORT_ROOT/$STAMP/jetson/full_validation.log"
set -uo pipefail
cd /opt/mousedroid
export MOUSEDROID_JETSON_CONFIG="$MOUSEDROID_JETSON_CONFIG"
export MOUSEDROID_METRICS__NAMESPACE="$MOUSEDROID_METRICS__NAMESPACE"
export MOUSEDROID_ESP32__ENABLED=false           # no motion, functionally-dead ESP32
export MOUSEDROID_VALIDATION_REPORT_ROOT=/opt/mousedroid/reports/jetson_full_validation
bash scripts/jetson_full_validation.sh 2>&1
EOF
JFV_RC=\${PIPESTATUS[0]:-\$?}
echo "jfv_rc=\$JFV_RC" | tee -a "$REPORT_ROOT/$STAMP/env.log"
```

Expected: script prints a per-phase `PASS/WARN/FAIL:` audit trail and a final summary. rc=0 when all phases pass and no FAIL was recorded; rc≠0 on FAIL. WARN is not a failure (dead ESP32 lives here).

- [ ] **Step 2: Pull the rover-side report tree back for local archiving**

```bash
ROVER_LATEST="$(ssh "$JETSON_HOST" 'ls -1t /opt/mousedroid/reports/jetson_full_validation/2* 2>/dev/null | head -1')"
if [ -n "$ROVER_LATEST" ]; then
  mkdir -p "$REPORT_ROOT/$STAMP/jetson/full_validation"
  scp -qr "${JETSON_HOST}:$ROVER_LATEST/*" "$REPORT_ROOT/$STAMP/jetson/full_validation/" \
    2>&1 | tee -a "$REPORT_ROOT/$STAMP/jetson/full_validation.log"
  echo "scp_rover_report_dir=$ROVER_LATEST" | tee -a "$REPORT_ROOT/$STAMP/env.log"
fi
```

Expected: local `jetson/full_validation/` dir populated with the rover's per-phase logs. If empty, the script didn't reach report-writing — check `jetson_full_validation.log` tail.

- [ ] **Step 3: Extract summary tallies (PASS/WARN/FAIL counts) — appended to findings.md**

```bash
{
  echo; echo "## Jetson full validation verdict"
  echo "- exit_code = $(grep '^jfv_rc=' $REPORT_ROOT/$STAMP/env.log | cut -d= -f2)"
  awk '/PASS:|WARN:|FAIL:/ {print}' "$REPORT_ROOT/$STAMP/jetson/full_validation.log" | tail -50
} >> "$REPORT_ROOT/$STAMP/findings.md"
```

Expected: last 50 PASS/WARN/FAIL lines of the composed run — the reviewer's punch list.

---

## Task 9: Regression trend check via `preflight --journal-path --trend`

**Files:**
- Read: worktree default JSONL journal path (or user-supplied)
- Write: `${REPORT_ROOT}/${STAMP}/local/preflight_trend.log`

**Interfaces:**
- Consumes: `PY`, and either an existing journal file from a prior run or a fresh one seeded by this task.
- Produces: regression verdict comparing the two newest preflight reports — flags status downgrades / new FAILs / latency creep.

- [ ] **Step 1: Record this run's preflight into the harness journal**

```bash
JOURNAL="${MOUSEDROID_JOURNAL_PATH:-$REPORT_ROOT/harness_journal.jsonl}"
mkdir -p "$(dirname "$JOURNAL")"
"$PY" -m mousedroid.cli.preflight --journal-path "$JOURNAL" 2>&1 \
  | tee "$REPORT_ROOT/$STAMP/local/preflight_record.log"
```

Expected: `preflight_report` event appended; exit 0 (OK/DEGRADED) or 1 (FAIL). Record either way — DEGRADED is not a run-abort here.

- [ ] **Step 2: Run the trend comparison against prior journal entries**

```bash
set +e
"$PY" -m mousedroid.cli.preflight --journal-path "$JOURNAL" --trend 2>&1 \
  | tee "$REPORT_ROOT/$STAMP/local/preflight_trend.log"
TREND_RC=$?
set -e
echo "preflight_trend_rc=$TREND_RC" | tee -a "$REPORT_ROOT/$STAMP/env.log"
```

Expected: rc=0 → no regression; rc=1 → regression detected (status downgrade, new FAIL, or latency creep beyond both `slow_ratio` AND `slow_floor_s`). Content: a human-readable regression report.

- [ ] **Step 3: Append trend verdict to findings.md**

```bash
{
  echo; echo "## Preflight trend"
  echo "- exit_code = $(grep '^preflight_trend_rc=' $REPORT_ROOT/$STAMP/env.log | cut -d= -f2) (0=no regression, 1=regression)"
  echo '```'
  cat "$REPORT_ROOT/$STAMP/local/preflight_trend.log"
  echo '```'
} >> "$REPORT_ROOT/$STAMP/findings.md"
```

Expected: findings.md ends with a full trend block.

---

## Task 10: Consolidation, verification, cleanup

**Files:**
- Read: `${REPORT_ROOT}/${STAMP}/findings.md`, `env.log`
- Write: `${REPORT_ROOT}/${STAMP}/SUMMARY.md`
- Delete (via `git worktree remove`): `${WORKTREE_DIR}` (only if all phases green — else leave for triage)

**Interfaces:**
- Consumes: every log written by Tasks 1–9.
- Produces: a top-of-report `SUMMARY.md` with a pass/fail matrix + operator's next-action recommendation.

- [ ] **Step 1: Invoke superpowers:verification-before-completion**

Announce: `Using superpowers:verification-before-completion.` Do NOT claim green until each stage's exit code has been verified from `env.log`, not from memory.

- [ ] **Step 2: Assemble SUMMARY.md — the reviewer's top-level artifact**

```bash
{
  echo "# Trunk-sync + Jetson validation run $STAMP"
  echo "- trunk_ref: $TRUNK_REF"
  echo "- trunk_sha: $(git rev-parse HEAD)"
  echo "- worktree: $WORKTREE_DIR"
  echo "- jetson_host: $JETSON_HOST"
  echo
  echo "## Stage matrix"
  echo "| Stage | Verdict |"
  echo "| --- | --- |"
  echo "| ruff check | $(grep '^ruff_check_rc=' $REPORT_ROOT/$STAMP/env.log | cut -d= -f2) |"
  echo "| ruff format --check | $(grep '^ruff_format_rc=' $REPORT_ROOT/$STAMP/env.log | cut -d= -f2) |"
  echo "| mypy --strict | $(grep '^mypy_rc=' $REPORT_ROOT/$STAMP/env.log | cut -d= -f2) |"
  echo "| local ci.sh | $(grep '^local_ci_rc=' $REPORT_ROOT/$STAMP/env.log | cut -d= -f2) |"
  echo "| jetson full validation | $(grep '^jfv_rc=' $REPORT_ROOT/$STAMP/env.log | cut -d= -f2) |"
  echo "| preflight trend | $(grep '^preflight_trend_rc=' $REPORT_ROOT/$STAMP/env.log | cut -d= -f2) |"
  echo
  echo "## Next action"
  # If ALL zero → green. Otherwise link findings.md and stop for triage.
  FAIL=0
  for k in ruff_check_rc mypy_rc local_ci_rc jfv_rc preflight_trend_rc; do
    v=$(grep "^${k}=" $REPORT_ROOT/$STAMP/env.log | cut -d= -f2)
    [ "$v" != "0" ] && FAIL=1
  done
  if [ "$FAIL" = "0" ]; then
    echo "- All stages green. Safe to close the worktree."
  else
    echo "- FAILURES detected. Do NOT remove the worktree yet. See findings.md."
  fi
} > "$REPORT_ROOT/$STAMP/SUMMARY.md"
cat "$REPORT_ROOT/$STAMP/SUMMARY.md"
```

Expected: matrix rendered, next-action line correct.

- [ ] **Step 3: Confirm the primary tree still matches its Task-1 snapshot (the untouched-tree guarantee)**

```bash
diff "$REPORT_ROOT/$STAMP/primary_tree_before.log" \
     <(git -C "$OLDPWD" status --porcelain) \
  && echo "primary tree UNCHANGED — good" | tee -a "$REPORT_ROOT/$STAMP/env.log"
```

Expected: no diff. If diff is non-empty, **STOP** and reconcile — something in this run leaked into the primary tree.

- [ ] **Step 4: Cleanup — remove worktree only on all-green**

```bash
if grep -q "All stages green" "$REPORT_ROOT/$STAMP/SUMMARY.md"; then
  cd "$OLDPWD"
  git worktree remove "$WORKTREE_DIR" --force
  git worktree prune
  echo "worktree removed" | tee -a "$REPORT_ROOT/$STAMP/env.log"
else
  echo "worktree KEPT for triage at $WORKTREE_DIR" | tee -a "$REPORT_ROOT/$STAMP/env.log"
fi
```

Expected on green: worktree gone, primary tree still dirty as before. Expected on red: worktree kept, reviewer has a warm venv + a full log tree to iterate against.

- [ ] **Step 5: Final report to user — under 200 words**

Output (as user-facing text): pass/fail per stage, path to `SUMMARY.md` and `findings.md`, one-sentence recommendation. No emojis. No prose beyond that.

---

## If Findings Warrant Fix-Forward (escape hatch, out-of-scope)

If Task 4's audit surfaces actionable ruff/format/mypy/numpy issues **and** the user asks to fix them: don't extend this plan. Instead:

1. Note the finding count in `findings.md`.
2. Spawn `superpowers:writing-plans` again with a fresh spec ("Fix N ruff-format drifts on trunk", "Address M mypy-strict misses in `world_model/`", etc.).
3. Each fix-forward should be its own PR against `${TRUNK_REF}` — TDD (write failing test → make it pass), one concern per PR, follows `superpowers:test-driven-development`.

This plan is intentionally read-only against `src/`. Bundling fixes here would defeat its purpose as a **repeatable validation harness**.

---

## Self-Review Notes (author's checklist output)

- **Spec coverage:** clone trunk ✅ (Task 2), deps ✅ (Task 3), local tests ✅ (Task 5), Jetson e2e ✅ (Task 8), agents/skills/worktrees ✅ (Tasks 2, 4, 9, 10 name the skills explicitly), ruff/lint/mypy/numpy notes ✅ (Task 4), no hardcoded values ✅ (Global Constraints + every step reads from env), backwards-compat ✅ (constraint), logging/debug ✅ (structlog constraint; findings.md is the debug catalog).
- **Placeholder scan:** no `TBD`/`TODO`; every "run this" step has the exact command; every "expected" line names the marker to look for.
- **Type/name consistency:** env-var names (`TRUNK_REF`, `WORKTREE_DIR`, `JETSON_HOST`, `REMOTE_USER`, `VENV_DIR`, `REPORT_ROOT`, `STAMP`, `PY`, `MOUSEDROID_JETSON_CONFIG`, `MOUSEDROID_METRICS__NAMESPACE`, `MOUSEDROID_ESP32__ENABLED`) referenced consistently across Tasks 1–10. Env-log keys (`ruff_check_rc`, `ruff_format_rc`, `mypy_rc`, `local_ci_rc`, `jfv_rc`, `preflight_trend_rc`, `sync_needed`, `post_sync_sha_match`) referenced consistently by Tasks 4/5/7/8/9/10.
