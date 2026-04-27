# Ten Pillars Validation Plan — Jetson Orin Nano

Operator-grade plan for validating every one of the 10 Pillars on the actual
`mousedroid` Jetson Orin Nano host (`ian@mousedroid.local`). Reuses the
existing container-backed smoke harness so every check runs against the
production runtime path (factory + `validation.runtime` + Pydantic config),
not a side path.

## Goals

- Every pillar has at least one **headless unit/regression check** plus one
  **on-Nano runtime check** that exercises the same factory wiring used by
  `python -m mousedroid.main`.
- All on-Nano runtime checks run inside the `mousedroid` container via the
  smoke wrapper's `python3-in-container` shim so config and devices match
  production.
- Pass criteria are **observable**, not subjective: exit code, structured
  log keys, telemetry counters, or recorded artefacts.
- Failures must be **playbook-routable** (point at the right file under
  [docs/playbooks/](../playbooks)).

## Pre-conditions on the Nano

Run once before the campaign and reuse the same run directory throughout:

1. SSH reachable: `ssh ian@mousedroid.local 'hostname; uname -a'`
2. Container up and healthy:
   `docker ps --filter name=^/mousedroid$ --format '{{.Names}} {{.Status}}'`
3. Service unit clean:
   `sudo systemd-analyze verify /etc/systemd/system/mousedroid-docker.service`
4. Production overlay synced:
   `sudo /opt/mousedroid/scripts/sync_jetson_overlay.sh`
5. Smoke wrapper warm-up (creates `python3-in-container` and run dir):
   `cd /opt/mousedroid && bash scripts/jetson_full_smoke_run.sh`
6. Capture latest run dir for reuse:
   `LATEST=$(ls -1dt /opt/mousedroid/reports/jetson_smoke/* | head -n 1)`
   `export MOUSEDROID_SMOKE_PYTHON="$LATEST/python3-in-container"`

All "On-Nano" commands below assume `MOUSEDROID_SMOKE_PYTHON` is exported.

## Execution Order

Run pillars in dependency order so a failure stops the campaign at the
right place:

`Safety -> World Model -> Memory -> Cognitive -> Reward -> Curiosity ->`
`Continual -> Meta -> Scaling -> Growth`.

Safety runs first because every later pillar depends on the runtime
safety monitor not vetoing actions during validation drives.

## Per-Pillar Validation

### Pillar 10 — Safety & Alignment (`src/mousedroid/safety/`)

- **Headless:** `pytest tests/unit/test_safety_monitor.py tests/unit/test_safety_context.py tests/unit/test_three_laws.py -q`
- **On-Nano runtime:**
  - Constitutional veto path:
    `"$MOUSEDROID_SMOKE_PYTHON" -m pytest -m hardware tests/hardware/test_esp32_loopback.py::test_emergency_stop_latency -q`
  - Runtime safety monitor wiring:
    `"$MOUSEDROID_SMOKE_PYTHON" -c "from mousedroid.config.loader import load_settings; from mousedroid.factory import build_safety_monitor; from mousedroid.validation.runtime import resolve_runtime_config_paths; cfg = load_settings(*resolve_runtime_config_paths()); m = build_safety_monitor(cfg); print('safety_monitor_ok', type(m).__name__)"`
- **Pass:** unit suite green; emergency-stop latency under configured budget;
  print line `safety_monitor_ok ...`.
- **Telemetry:** `safety_violations_total` counter visible at
  `http://mousedroid.local:8080/metrics`.

### Pillar 1 — World Model (`src/mousedroid/world_model/`)

- **Headless:** `pytest tests/unit/test_rssm.py tests/unit/test_dual_stream_rssm.py tests/unit/test_mcts.py -q`
- **On-Nano runtime:**
  - Build dual-stream RSSM with the active config:
    `"$MOUSEDROID_SMOKE_PYTHON" -c "import torch; from mousedroid.config.loader import load_settings; from mousedroid.factory import build_world_model; from mousedroid.validation.runtime import resolve_runtime_config_paths; cfg = load_settings(*resolve_runtime_config_paths()); wm = build_world_model(cfg); print('world_model_ok', type(wm).__name__, sum(p.numel() for p in wm.parameters() if hasattr(wm,'parameters')))"`
  - End-to-end tick (already exercised by `bash scripts/jetson_smoke_test.sh e2e`).
- **Pass:** unit suite green; build line printed; e2e stage produces at
  least one orchestrator tick with `world_model.predict` log key.

### Pillar 4 — Memory Systems (`src/mousedroid/memory/`)

- **Headless:** `pytest tests/unit/test_memory_working.py tests/unit/test_memory_episodic.py tests/unit/test_memory_semantic.py tests/unit/test_memory_tier.py tests/unit/test_consolidation.py -q`
- **On-Nano runtime:** verify the LMDB experience store the orchestrator
  writes to is non-empty after a 30 s drive:
  - `timeout 30 "$MOUSEDROID_SMOKE_PYTHON" -m mousedroid.main --config /etc/mousedroid/default.yaml /etc/mousedroid/jetson_production.yaml || true`
  - `"$MOUSEDROID_SMOKE_PYTHON" -c "from mousedroid.experience.dataset import OfflineRLDataset; from mousedroid.config.loader import load_settings; from mousedroid.validation.runtime import resolve_runtime_config_paths; cfg=load_settings(*resolve_runtime_config_paths()); ds=OfflineRLDataset(cfg.experience.lmdb_path); print('lmdb_records', len(ds))"`
- **Pass:** unit suite green; `lmdb_records` strictly greater than the
  pre-drive baseline.

### Pillar 2 — Cognitive Architecture (`src/mousedroid/cognitive/`, `orchestrator/`)

- **Headless:** `pytest tests/unit/test_cognitive_core.py tests/unit/test_metacognitive.py tests/unit/test_orchestrator.py tests/unit/test_orchestrator_telemetry.py -q`
- **On-Nano runtime:** confirm dual-cadence ticks at the configured rate
  during a 10 s run:
  - `timeout 10 "$MOUSEDROID_SMOKE_PYTHON" -m mousedroid.main --config /etc/mousedroid/default.yaml /etc/mousedroid/jetson_production.yaml 2>&1 | tee /tmp/cog_run.log || true`
  - `grep -c '"event": "tick"' /tmp/cog_run.log`
- **Pass:** unit suite green; tick count consistent with
  `orchestrator.tick_hz` from production config (within 20 percent).

### Pillar 8 — Reward Modelling (`src/mousedroid/reward/`)

- **Headless:** `pytest tests/unit/test_reward_model.py tests/unit/test_reward_aggregator.py tests/unit/test_three_laws_reward.py -q`
- **On-Nano runtime:** scrape per-objective reward components from `/metrics`
  during the same 10 s run as Pillar 2:
  - `curl -s http://mousedroid.local:8080/metrics | grep -E '^mousedroid_reward_(progress|safety|battery|comfort)'`
- **Pass:** unit suite green; all four configured reward objectives
  appear at least once with finite values.

### Pillar 6 — Curiosity & Exploration (`src/mousedroid/curiosity/`)

- **Headless:** `pytest tests/unit/test_icm.py tests/unit/test_curiosity_wiring.py tests/unit/test_novelty_decay.py -q`
- **On-Nano runtime:** confirm ICM intrinsic reward is wired into the
  reward aggregator and produces non-zero novelty during exploration:
  - reuse the 10 s run; `curl -s http://mousedroid.local:8080/metrics | grep '^mousedroid_curiosity_intrinsic_reward'`
- **Pass:** unit suite green; intrinsic reward sample count strictly
  positive.

### Pillar 3 — Continual Learning (`src/mousedroid/learning/`)

- **Headless:** `pytest tests/unit/test_ewc.py tests/unit/test_progressive.py -q`
- **On-Nano runtime:** load existing weights and verify EWC penalty path
  is reachable without GPU OOM at the production batch size:
  - `"$MOUSEDROID_SMOKE_PYTHON" -m mousedroid.learning.ewc --selfcheck` (if
    a CLI exists); otherwise a 1-step train smoke from
    `training/run_pipeline.py --phase 1 --steps 1 --replay false`.
- **Pass:** unit suite green; on-Nano selfcheck or 1-step pipeline
  returns rc 0 within configured `learning.timeout_s`.

### Pillar 5 — Meta-Learning (`src/mousedroid/meta/`)

- **Headless:** `pytest tests/unit/test_maml.py tests/unit/test_in_context.py -q`
- **On-Nano runtime:** in-context adaptation smoke:
  `"$MOUSEDROID_SMOKE_PYTHON" -c "from mousedroid.config.loader import load_settings; from mousedroid.factory import build_meta_learner; from mousedroid.validation.runtime import resolve_runtime_config_paths; cfg=load_settings(*resolve_runtime_config_paths()); m=build_meta_learner(cfg); print('meta_ok', type(m).__name__)"`
- **Pass:** unit suite green; meta learner builds from the production
  config without a fallback path being taken.

### Pillar 9 — Scaling (`src/mousedroid/scaling/`)

- **Headless:** `pytest tests/unit/test_moe.py tests/unit/test_adaptive_compute.py tests/unit/test_batch_tuner.py -q`
- **On-Nano runtime:** confirm adaptive compute selects the configured
  Jetson tier during the 10 s run:
  - `grep '"event": "adaptive_compute_decision"' /tmp/cog_run.log | head -3`
- **Pass:** unit suite green; at least one decision log entry, with
  `tier` matching `scaling.jetson_tier`.

### Pillar 7 — Growth & Distillation (`src/mousedroid/growth/`)

- **Headless:** `pytest tests/unit/test_distillation.py -q`
- **On-Nano runtime:** validate the Jetson efficiency layer (TensorRT +
  profiler are tagged Pillar 10 in code but feed Pillar 7 deployment):
  - `"$MOUSEDROID_SMOKE_PYTHON" -c "import tensorrt; print('trt', tensorrt.__version__)"`
  - `"$MOUSEDROID_SMOKE_PYTHON" -m mousedroid.efficiency.profiler --selfcheck` (if implemented; otherwise skip).
- **Pass:** unit suite green; TensorRT import succeeds; profiler
  selfcheck returns rc 0 or is explicitly skipped.

## Aggregated Run

A single shell loop should drive the full campaign and produce one
artefact directory:

```bash
ssh ian@mousedroid.local 'cd /opt/mousedroid && bash scripts/jetson_full_smoke_run.sh'
LATEST=$(ssh ian@mousedroid.local 'ls -1dt /opt/mousedroid/reports/jetson_smoke/* | head -n 1')
ssh ian@mousedroid.local "export MOUSEDROID_SMOKE_PYTHON=$LATEST/python3-in-container; \
  for p in safety world_model memory cognitive reward curiosity continual meta scaling growth; do \
    echo === pillar:$p ===; \
    bash /opt/mousedroid/scripts/validate_pillar.sh $p || echo PILLAR_FAIL:$p; \
  done | tee $LATEST/ten_pillars.log"
```

`scripts/validate_pillar.sh` is present in the repository and implements
this plan. It dispatches each pillar's commands using the same env contract
as `jetson_full_smoke_run.sh`: reads `MOUSEDROID_SMOKE_PYTHON`, writes
per-pillar log files to `$REPORT_DIR/pillar_<name>_<kind>.log`, and writes
the campaign summary table to `$REPORT_DIR/ten_pillars.log`.
Use `bash scripts/validate_pillar.sh all` to run all ten pillars, or
`bash scripts/validate_pillar.sh <pillar>` to run a single pillar.

## Reporting

Each campaign produces:

- `reports/jetson_smoke/<stamp>/ten_pillars.log` — top-level pass/fail
  per pillar.
- `reports/jetson_smoke/<stamp>/pillar_<name>.log` — raw stdout/stderr
  per pillar.
- `reports/jetson_smoke/<stamp>/SUMMARY.md` — extended with a
  "Ten Pillars" table when `ten_pillars.log` is present.

## Exit Criteria

- All 10 pillars green on `mousedroid.local` against the active
  production overlay.
- Telemetry counters listed above are observable from the Prometheus
  endpoint during the campaign window.
- No skipped pillar without an explicit, recorded reason in
  `ten_pillars.log` (for example: "Pillar 7 profiler selfcheck not yet
  implemented").

## Out of Scope

- Robot arm pillars (parked outside active delivery).
- HC-SR04 ultrasonic path (parked).
- Cloud digital-twin sync; covered by separate cloud test suite.
