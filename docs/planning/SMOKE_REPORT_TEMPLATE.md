# Smoke Report — `<branch-under-review>`

> Operator copy of this template to `SMOKE_REPORT.md` at the repo root after running `docs/operator/JETSON_SMOKE_RUNBOOK.md` against a live Jetson. PR review merges only after this report is filled.

## Run metadata

- **Operator:** _name / handle_
- **Date (UTC):** _YYYY-MM-DDTHH:MM:SSZ_
- **Branch:** _branch name + last commit SHA_
- **Jetson host:** _e.g. 192.168.55.1 — Jetson Orin Nano 8GB, JetPack 6.x_
- **L4T container or bare-metal venv?:** _container / venv_
- **Ollama daemon reachable?:** _yes (base_url) / no_
- **Motor loopback enabled (`MOUSEDROID_SMOKE_ALLOW_MOTION=1`)?:** _yes / no_

## Step 2 — Pre-flight

| Check | Result | Notes |
|---|---|---|
| `bash scripts/preflight_check.sh` | _PASS / FAIL_ | _link to log if FAIL_ |
| `python -m mousedroid.cli.validate_pillars --dry-run` | _PASS / FAIL_ | |

## Step 3 — Pillar results

Paste relevant rows from `/tmp/pillars.json`. Mark FAIL entries in **bold**.

| Pillar | Status | Elapsed (ms) | Detail |
|---|---|---|---|
| safety | _OK / FAIL_ | _ms_ | |
| world_model | _OK / FAIL_ | _ms_ | |
| memory | _OK / FAIL_ | _ms_ | |
| cognitive | _OK / FAIL_ | _ms_ | |
| reward | _OK / FAIL_ | _ms_ | |
| curiosity | _OK / FAIL_ | _ms_ | |
| continual | _OK / FAIL_ | _ms_ | _Pattern-B (delegates to ewc + progressive)_ |
| meta | _OK / FAIL_ | _ms_ | _Pattern-B (delegates to maml)_ |
| scaling | _OK / FAIL_ | _ms_ | _Pattern-B (delegates to moe + adaptive_compute + batch_tuner)_ |
| growth | _OK / FAIL_ | _ms_ | _Pattern-B (delegates to distillation)_ |
| **Overall** | _OK / DEGRADED / FAIL_ | _total ms_ | |

## Step 4 — Hardware smoke (`scripts/jetson_smoke_test.sh all`)

| Stage | Result | Notes |
|---|---|---|
| 1. System (CUDA / TensorRT / thermal) | _PASS / FAIL / SKIP_ | |
| 2. GPIO (Jetson.GPIO probe) | _PASS / FAIL / SKIP_ | |
| 3. Serial (ESP32 JSON loopback) | _PASS / FAIL / SKIP_ | |
| 4. Motor loopback (encoders) | _PASS / FAIL / SKIP_ | _SKIP unless ALLOW_MOTION=1_ |
| 5. Camera (`verify_sensors.py --sensor camera`) | _PASS / FAIL / SKIP_ | |
| 6. Audio (`--sensor audio`) | _PASS / FAIL / SKIP_ | |
| 7. LiDAR (`--sensor lidar`) | _PASS / FAIL / SKIP_ | |
| 8. Speaker (`--sensor speaker`) | _PASS / FAIL / SKIP_ | |
| 9. Voice (TTS roundtrip) | _PASS / FAIL / SKIP_ | |
| 10. App health (`mousedroid.main --health-check`) | _PASS / FAIL_ | |
| 11. Hardware pytest (`pytest -m hardware`) | _N passed / M failed_ | |
| 12. E2E 5-second tick run | _PASS / FAIL_ | _loop ms p95?_ |

## Step 5 — Windows-skipped tests now PASSING on Linux

```
pytest tests/unit/test_jetson_smoke_orchestrator.py -v
```

- **Expected:** 13 PASSED (no skips on Jetson).
- **Actual:** _N passed, M skipped, K failed_
- **If any skips:** investigate why `python3` isn't reachable from the bash subprocess on this host.

## Step 6 — Hardware-marker pytest sweep

```
pytest -m hardware tests/hardware/ tests/e2e/test_jetson_hardware_e2e.py -v
```

| Module | Passed | Failed | Skipped | Notes |
|---|---|---|---|---|
| `tests/hardware/test_esp32_loopback.py` | | | | |
| `tests/hardware/test_jetson_smoke.py` | | | | |
| `tests/hardware/test_e2e_sense_plan_act.py` | | | | |
| `tests/hardware/test_e2e_edge_cases.py` | | | | |
| `tests/hardware/test_face_display_smoke.py` *(new this sprint)* | | | | |
| `tests/hardware/test_hailo_smoke.py` *(new this sprint)* | | | | |
| `tests/e2e/test_jetson_hardware_e2e.py` (8 subsystems) | | | | |

## Step 7 — Mission lifecycle live smoke (optional, Ollama)

- **Ran?:** _yes / no (Ollama not reachable / skipped)_
- **Result:** _PASSED / FAILED (with detail)_

## Step 8 — Structured-log grep (observability sanity)

Pull the structured events the new preflight + pillar code emits from the orchestrator's log stream. Confirm each event appears at least once:

```
grep -E "preflight_(start|complete|check_exception)|pillar_validation_(start|complete)|pillar_check_exception" /var/log/mousedroid.log | head -20
```

- `preflight_complete{overall=…}`: _expected once per boot_
- `pillar_validation_complete{overall=…}`: _expected once per CLI invocation_
- Any `*_exception` events: _list + triage each_

## Findings + follow-ups

For each FAIL or unexpected SKIP above:

| ID | Surface | Root cause | Follow-up issue / PR |
|---|---|---|---|
| F-001 | _e.g. pillar.scaling_ | _stale unit test_ | _link_ |

## Sign-off

- [ ] All Step-3 pillars OK (no FAIL).
- [ ] All Step-4 hardware stages PASS or documented SKIP.
- [ ] Step-5 jetson_smoke_orchestrator tests RUN on Linux (no module-level skip).
- [ ] Findings table complete with follow-up links for any FAIL.

**Operator signature:** _name / date_
