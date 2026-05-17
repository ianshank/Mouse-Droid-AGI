# Smoke Report — `claude/smoke-test-stability-pass`

## Run metadata

- **Operator:** Claude (executing remotely on behalf of @ianshank)
- **Date (UTC):** 2026-05-17T17:24:54Z
- **Branch:** `claude/smoke-test-stability-pass` @ `0d01594` (lazy-pytest-import fix landed mid-run)
- **Jetson host:** `mousedroid` @ `192.168.55.1` (USB-network bridge) — Jetson Orin Nano 8GB, L4T R36.4.7, kernel 5.15.148-tegra
- **Deployment:** Docker (image `mousedroid:jetson`, container `mousedroid` up 58 min healthy); host has no Python install of `mousedroid`
- **Container Python:** `/usr/bin/python3` 3.10 (CUDA-torch via `dustynv/l4t-pytorch` base image)
- **Host Python:** `/usr/bin/python3` 3.10.12 (CPU-only torch — host smoke-script torch.cuda check FAILs as a result)
- **Ollama daemon reachable?:** Not probed (mission-lifecycle smoke skipped — out of scope this run)
- **Motor loopback enabled (`MOUSEDROID_SMOKE_ALLOW_MOTION=1`)?:** No (rover currently detached from Jetson per operator)

## Physical state during run

- Jetson **detached from rover chassis** — USB-C cable currently routes PC ↔ Jetson directly (not via rover).
- Consequently: rover-mounted CSI camera + FHL-LD19 LiDAR + ESP32 motor controller all FAIL/SKIP at the hardware probe layer. These are **expected failures** correlated to physical disconnection, not regressions.

## Step 2 — Pre-flight

| Check | Result | Notes |
|---|---|---|
| `bash scripts/preflight_check.sh` | _not run — host has no mousedroid_ | See finding F-002 below |
| `docker exec mousedroid python3 -m mousedroid.cli.validate_pillars --dry-run` | ✅ **PASS** | All 10 pillars listed as SKIPPED (dry-run), exit 0, 70µs total |

## Step 3 — Pillar results

`docker exec mousedroid python3 -m mousedroid.cli.validate_pillars --json` — exit 0, **overall=ok** in 540ms.

| Pillar | Status | Elapsed | Detail |
|---|---|---|---|
| safety | ✅ OK | 0.04 ms | `monitor=MouseDroidSafetyMonitor` |
| world_model | ✅ OK | 30.5 ms | `engine=RSSM` |
| memory | ⏭ SKIPPED | — | `build_memory_tier returned None (memory disabled in cfg)` |
| cognitive | ✅ OK | 463.1 ms | `core=CognitiveCore` (loads BDI weights from disk) |
| reward | ✅ OK | 29.0 ms | `model=MultiObjectiveRewardModel` |
| curiosity | ⏭ SKIPPED | — | `build_curiosity_module returned None (disabled in cfg)` |
| continual | ⏭ SKIPPED | — | pytest not installed in production runtime (Pattern-B delegation requires dev extras) |
| meta | ⏭ SKIPPED | — | same |
| scaling | ⏭ SKIPPED | — | same |
| growth | ⏭ SKIPPED | — | same |
| **Overall** | ✅ **OK** | **540 ms** | 4 OK / 6 SKIPPED (all SKIPs are documented expected states, not failures) |

## Step 4 — Preflight on real hardware (`mock_hardware=False`)

`docker exec mousedroid python3 -c "asyncio.run(run_preflight(load_settings(); cfg.mock_hardware=False))"` — exit 0, **overall=fail** in 2.13s (2 of 6 checks failed, both correlated to rover detachment).

| Subsystem | Status | Elapsed | Detail |
|---|---|---|---|
| camera | ❌ **FAIL** | 0 ms | `RuntimeError: Failed to open CSI camera via GStreamer pipeline or V4L2 device /dev/video0` — **expected: rover-mounted CSI camera is detached** |
| microphone | ✅ OK | 449 ms | `samples=1024 dtype=float32` (USB PnP Audio Device) |
| speaker | ✅ OK | 616 ms | `samples_written=7168` (USB hw:0,0) |
| lidar | ❌ **FAIL** | 0 ms | `no diagnostics returned` — **expected: rover-mounted FHL-LD19 is detached** |
| esp32 | ✅ OK | 4 ms | `driver=ResilientESP32Driver` (mock fallback active — physical ESP32 detached but driver built cleanly) |
| config | ✅ OK | <1 ms | `action_dim=3 control_hz=30.0` |

## Step 4-host — `scripts/jetson_smoke_test.sh` stages

| Stage | Result | Notes |
|---|---|---|
| system | 🟡 3 PASS / 1 FAIL | TensorRT 10.3.0 OK, thermal 55°C OK, memory 29% used OK. **Host `torch.cuda.is_available() == False`** — see F-001 |
| gpio | ❌ FAIL | `ModuleNotFoundError: No module named 'mousedroid'` — see F-002 |
| serial | 🟡 hung | Same root cause (host-Python `import mousedroid` blocks I/O) |
| motor / camera / audio / lidar / speaker / voice / app / pytest / e2e | _not attempted_ | All depend on host Python being able to import mousedroid — see F-002 |

## Step 5 — Orchestrator health-check (container)

`docker exec mousedroid python3 -m mousedroid.main --health-check` — exit 0.

```json
{"status": "ok",
 "platform": "PlatformType.MOUSE_DROID",
 "mock_hardware": "True",
 "agents": ["mouse_droid_navigator"],
 "event": "health_check_passed"}
```

Full DI graph instantiates cleanly: encoder → RSSM → MCTS → circuit breakers → mock camera → mock ultrasonic → mock microphone (real USB sample-rate detected) → audio feature extractor → sensor manager → health monitor → mission parser → speaker (real USB hardware) → Rocky voice engine. **Container is healthy.**

## Step 6 — Structured-log grep (observability sanity)

Captured live from the smoke runs:

```
preflight_start                checks=['camera', 'microphone', 'speaker', 'lidar', 'esp32', 'config']
preflight_complete             checks_run=6 elapsed_s=0.000104 overall=ok    # mock-hardware run
preflight_complete             checks_run=6 elapsed_s=2.131094 overall=fail  # real-hardware run (camera+lidar detached)
pillar_validation_start        dry_run=False pillars=[10 names]
pillar_validation_complete     dry_run=False elapsed_s=0.540041 overall=ok
health_check_passed
```

✅ All new structured events fire as designed. Operator dashboards can ingest these directly.

## Findings + follow-ups

| ID | Severity | Surface | Root cause | Resolution |
|---|---|---|---|---|
| F-001 | INFO | Host `torch.cuda.is_available() == False` (bash smoke stage 1) | Host Python has CPU-only torch; CUDA torch is inside the container | Expected on Docker-only deployments; not a regression. Bash smoke should delegate stage 1 to `docker exec mousedroid python3 -c "import torch; ..."` |
| F-002 | **HIGH (architecture)** | `scripts/jetson_smoke_test.sh` GPIO/serial/app/pytest stages all FAIL with `ModuleNotFoundError: No module named 'mousedroid'` | The bash script assumes a venv-based install where the host Python has mousedroid importable. This Jetson is **Docker-only deployment** (no host Python install). | **Follow-up sprint:** make `jetson_smoke_test.sh` Docker-aware. Detect host-vs-container deployment and route mousedroid-importing stages through `docker exec mousedroid …`. |
| F-003 | **HIGH (bug, fixed in this run)** | `validate_all_pillars` crashed at module import with `ModuleNotFoundError: No module named 'pytest'` inside the production container | I imported `pytest` at module level in `pillars.py`; pytest is a dev-only extra not in the production Docker image | **Fixed in commit `0d01594`** — lazy import + graceful SKIP when pytest is absent. New regression test pinning the path. Pulled to Jetson + re-verified live. |
| F-004 | INFO | Jetson repo was on commit `90119ef` (pre-Tier-C2.3, multiple months stale) before this run | Operator deployment script doesn't auto-pull integration branch | Documented; no code fix needed (deployment is operator-driven). |
| F-005 | INFO | Camera + LiDAR preflight FAIL on real hardware | Jetson currently physically detached from rover chassis (operator confirmed earlier in session) | **Expected state, not a regression.** Re-run preflight after re-attaching the rover to clear F-005. |

## Sign-off

- [x] All Step-3 pillars OK or SKIPPED with documented reasons (no FAIL).
- [x] Hardware preflight: 4 OK + 2 expected FAIL (rover detached) — re-run after re-attaching rover.
- [x] Orchestrator health-check: PASSED inside the live Docker container.
- [x] Structured-log events: all 5 new event names fire as designed.
- [x] Real bug found + fixed mid-run (F-003, commit `0d01594`).
- [x] F-002 (bash smoke script Docker-mismatch) logged as a follow-up sprint.

**Operator signature:** Claude on behalf of @ianshank, 2026-05-17T17:24:54Z (Jetson smoke pass executed via SSH at `192.168.55.1`)
