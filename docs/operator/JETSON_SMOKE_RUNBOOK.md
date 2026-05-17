# Jetson Rover Smoke Runbook

> Operator-facing step-by-step for running the full hardware + AI smoke pass against a live MouseDroid rover. Output of this run fills `docs/planning/SMOKE_REPORT_TEMPLATE.md`, which PR-attaches to the branch under review.

## Prerequisites

- Jetson Orin Nano reachable at `192.168.55.1` (your documented Jetson<->host USB-network bridge).
- `git`, `bash`, `python3` available on the Jetson; the deploy `.venv` is at `/opt/mousedroid/venv` (per `scripts/jetson_smoke_test.sh` defaults).
- ESP32 motor driver wired and the rover physically clear of obstacles if motor loopback will run.
- The branch under review checked out on the Jetson (`git fetch && git checkout <branch>`).
- An Ollama daemon (or other OpenAI-compatible LLM) reachable if you want the mission-lifecycle smoke to exercise the replan path — otherwise the lifecycle defaults to disabled.

## Step 1 — SSH + checkout

```bash
ssh jetson@192.168.55.1
cd /opt/mousedroid/src   # or wherever the repo is cloned
git fetch origin
git checkout <branch-under-review>
```

## Step 1b — Config-deploy verification (F-013)

After every `git pull` on the Jetson, run the overlay-sync verification so the
deployed `/etc/mousedroid/jetson_production.yaml` matches the repo. The
production restart that surfaced F-013 only failed because the deployed copy
silently drifted four days behind the repo and the silent mode of the
sync script hid the drift.

```bash
# Verify the deployed overlay matches the repo (read-only; never mutates state).
sudo bash scripts/sync_jetson_overlay.sh --verify
# Expected on a clean deploy:
#   [sync_jetson_overlay] OK overlay_sync_match src=... dst=... sha256=...
# Exit 0 = synced; exit 1 = drift (run without --verify to repair).

# Apply the sync (idempotent — logs overlay_sync_match if already current).
sudo bash scripts/sync_jetson_overlay.sh
```

Also ensure `/etc/mousedroid/docker.env` exists and matches the template at
`config/.env.jetson.example`. The compose `env_file:` directive (F-014) reads
from `/etc/mousedroid/docker.env` for `MOUSEDROID_MOCK_HARDWARE`,
`MOUSEDROID_TELEMETRY_TOKEN`, and any per-host `MOUSEDROID_LLM__*` overrides.

```bash
# First-time setup (then populate the token):
sudo cp config/.env.jetson.example /etc/mousedroid/docker.env
sudoedit /etc/mousedroid/docker.env   # set MOUSEDROID_TELEMETRY_TOKEN to a real value
```

## Step 2 — Pre-flight check (shell + Python)

The existing shell preflight stays as the bash entry point; the new programmatic API is the in-process replacement that the orchestrator + CLI consume.

```bash
# Shell preflight (legacy entry — kept for operator runbook continuity)
bash scripts/preflight_check.sh

# Programmatic preflight (new, Tier-smoke deliverable)
python -m mousedroid.cli.validate_pillars --dry-run
```

Both should exit 0. If either fails, **stop and triage** before continuing — the deeper smoke stages assume preflight is green.

## Step 3 — Pillar validation (10 pillars, ~5-30s)

```bash
python -m mousedroid.cli.validate_pillars --json | tee /tmp/pillars.json
```

- Exit 0 → all 10 pillars OK / SKIPPED. Paste `/tmp/pillars.json` into the SMOKE_REPORT "Pillar results" table.
- Exit 1 → at least one pillar FAILED. Capture the `detail` field for each FAIL entry and triage before the next stage.

## Step 4 — Hardware smoke (scripts/jetson_smoke_test.sh)

```bash
# Motor loopback gate — set when the rover is physically clear to move
export MOUSEDROID_SMOKE_ALLOW_MOTION=1

bash scripts/jetson_smoke_test.sh all | tee /tmp/jetson_smoke.log
```

Stages (numbered per the script): system, GPIO, serial, motor, camera, audio, LiDAR, speaker, voice, app health, hardware pytest, E2E 5-second run. Each emits PASS / FAIL / SKIP; final block prints the rollup tally.

## Step 5 — Re-run the Windows-skipped tests on Linux (regression net)

These tests SKIP on Windows-Git-Bash (no `python3` in subprocess PATH). On the Jetson they should actually RUN and PASS — pinning the fix from Task 4 of this sprint:

```bash
pytest tests/unit/test_jetson_smoke_orchestrator.py -v
```

Expected: 13 PASSED (not skipped).

## Step 6 — Hardware-marker pytest sweep

```bash
pytest -m hardware tests/hardware/ tests/e2e/test_jetson_hardware_e2e.py -v
```

Expected: per-sensor pillar tests pass. The new `test_face_display_smoke.py` and `test_hailo_smoke.py` (this sprint) run here too — they skip cleanly when the corresponding hardware isn't present.

## Step 7 — Optional: full live-mission lifecycle smoke (Ollama)

If an Ollama daemon is reachable from the Jetson (typically `http://192.168.55.1:11434` if running on your host PC, or `http://127.0.0.1:11434` if on the Jetson itself):

```bash
export MOUSEDROID_LLM__BACKEND=openai_compatible
export MOUSEDROID_LLM__BASE_URL=http://192.168.55.1:11434
export MOUSEDROID_LLM__MODEL_NAME=gemma-4-e4b
pytest tests/smoke/test_mission_lifecycle_smoke.py -v -m smoke
```

## Step 8 — Fill the SMOKE_REPORT

Copy `docs/planning/SMOKE_REPORT_TEMPLATE.md` to `SMOKE_REPORT.md`, fill every section from the logs above, and commit it to the same branch:

```bash
cp docs/planning/SMOKE_REPORT_TEMPLATE.md SMOKE_REPORT.md
# fill it in
git add SMOKE_REPORT.md
git commit -m "docs(smoke-pass): operator Jetson smoke run results"
git push
```

The PR review merges once the SMOKE_REPORT.md is filled with PASS-equivalent outcomes (or documented FAILs that have a follow-up issue linked).

## Troubleshooting cheatsheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `pytest tests/unit/test_jetson_smoke_orchestrator.py` still SKIPPED | `python3` not in bash PATH on Jetson | `which python3` — install / symlink |
| `bash scripts/jetson_smoke_test.sh all` motor stage skipped | `MOUSEDROID_SMOKE_ALLOW_MOTION` not set | `export MOUSEDROID_SMOKE_ALLOW_MOTION=1` (only when rover is clear) |
| `validate_pillars` continual/meta/scaling/growth FAIL | Pattern-B delegation finds a failing unit test | Re-run that unit test directly via `pytest tests/unit/test_<x>.py -v` |
| Face display test FAILS not SKIPS on Jetson | I²C bus / address mismatch in `cfg.face_display` | Set `MOUSEDROID_FACE_DISPLAY_DEV` or edit overlay; `fallback_to_mock_on_error=True` keeps the orchestrator alive |
| Hailo smoke FAILS not SKIPS | hailort installed but device not reachable | `lspci | grep -i hailo` to confirm device; check PCIe seating |
