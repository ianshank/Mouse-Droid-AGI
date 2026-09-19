# Runbook — Jetson ONNX observe_step benchmark

Measure the world model's `observe_step` stage on the rover, across engines, and
record the result as evidence. Feature: **F-050** (runtime + performance
evidence). Change bundle:
`openspec/changes/mouse-droid-jetson-onnx-delivery/`.

> **Read this first — the decision this runbook exists to keep honest.**
> The change's task-2.1 decision gate **closed against** building an ONNX/TensorRT
> fast path: the computed end-to-end ceiling for replacing the whole stage is
> **1.0016x–1.0039x**. So `world_model.engine` stays `"torch"`, and no I/O
> binding, FP16 profile or TensorRT engine cache is enabled. Benchmarking here is
> for *evidence about the stage*, not for shipping a switch. If a number in this
> runbook ever argues for flipping the engine, it has to beat that ceiling
> first — and the ceiling is a property of Amdahl arithmetic over the tick, not
> of the kernel.

## Scope and non-scope

| In scope | Not in scope |
| --- | --- |
| Timing `observe_step` on the rover, torch engine | Enabling `engine: onnx_trt` in any tracked overlay |
| Recording provider availability | Building a TensorRT engine on-device |
| Cold-vs-warm cache behaviour of the new cache volume | FP16 / INT8 / I/O binding / CUDA graphs |

## Preconditions

- The container is running and healthy
  (`docker ps --filter name=mousedroid`; the image `HEALTHCHECK` runs
  `scripts/mousedroid_healthcheck.sh`, which only checks heartbeat freshness).
- **Disk:** `df -h /` plus `docker system df -v`, **>= 8 GiB free**. Reclaim with
  `scripts/jetson_disk_cleanup.sh`.
- **Memory:** record `swapon --show` and `zramctl`. The container is capped at
  6 GiB of ~7.4 GiB usable RAM and swap is deliberately unbounded — see the
  `RESOURCE BUDGET DECISION` note in `docker-compose.jetson.yml` for why, and
  for the three knobs to set together if engine builds are ever enabled.
- Bench time: the on-rover phase is scheduled **behind F-008** ("USB-C rover
  smoke passes on the physical Jetson") per the F-024 hardware-priority rule.
  It competes for the same bench.

## 1. Establish the arithmetic before touching the rover

The ceiling is computed on the PC, from the observed share of the tick that
`observe_step` occupies:

```bash
python scripts/analyze_observe_step_ceiling.py --justify-ceiling
```

This is the tool that produced the 1.0016x–1.0039x figure. Run it first so the
rover measurement is interpreted against a stated ceiling rather than admired
on its own.

## 2. Measure on the rover

```bash
docker exec mousedroid python3 scripts/benchmark_latency.py \
  --config /etc/mousedroid/jetson_production.yaml \
  --checkpoint weights/rssm/final.pt
```

`benchmark_latency.py` today covers `RSSM.imagine_step` and
`MCTSPlanner.plan`, with p95 targets exposed as `--rssm-target-ms` and
`--mcts-target-ms` and an exit code that reflects them. **The dedicated
`observe_step` mode is task 7.5 of this bundle and is not present yet** — until
it lands, the per-stage number comes from the histogram below rather than from a
flag. Do not add a second benchmark script; there is one source of truth for the
10 ms / 33 ms numbers.

The stage's own instrumentation is the Prometheus family
`mousedroid_world_model_observe_step_seconds` (F-050). Read it from the
telemetry endpoint rather than re-timing by hand:

```bash
curl -sf "http://127.0.0.1:${MOUSEDROID_TELEMETRY_PORT:-8080}/metrics" \
  | grep mousedroid_world_model_observe_step_seconds
```

The performance-tier budget for the same stage is
`tests/performance/test_observe_step_budget.py`, whose ceiling comes from
`MOUSEDROID_OBSERVE_STEP_BUDGET_MS`. Keep the rover number and that budget in
the same units and do not restate either as a literal in a new place.

## 3. Cold versus warm cache

The TensorRT engine/timing cache now persists in the named volume
`mousedroid_tensorrt_cache`, mounted at `cfg.jetson.tensorrt_cache_dir` (set
once via `MOUSEDROID_JETSON__TENSORRT_CACHE_DIR` in
`/etc/mousedroid/docker.env` to relocate it). Nothing writes to it while the
engine is `torch`, which is the expected state — so the useful check today is
that recreation does not *lose* it:

```bash
docker volume inspect mousedroid_tensorrt_cache          # exists, driver local
docker compose -f docker-compose.jetson.yml up -d --force-recreate
docker exec mousedroid ls -la "${MOUSEDROID_JETSON__TENSORRT_CACHE_DIR:-/opt/mousedroid/tensorrt_cache}"
```

A warm run must be recorded **as** a warm run. An unlabelled warm number
compared against a cold one is the most common way this measurement lies.

## 4. Prove the provider, do not assume it

```bash
bash scripts/docker_deploy.sh --health-only --strict-health
```

Under the default `engine: torch` the provider and digest legs report `n/a` and
pass — there is no ONNX artifact in the 30 Hz path to prove. Under
`engine: onnx_trt` the same command **fails closed**: an observed provider other
than the configured primary, an absent `model_sha256` in
`deployments/jetson-image.json`, or a digest mismatch all abort. That is the
intended asymmetry; a green `--strict-health` on the torch engine is not
evidence about ORT.

## 5. Record the evidence

- Provider chain, warm/cold status, and the tick-share arithmetic go together —
  a latency number without its provider and cache state is not reproducible.
- Redact hostnames, usernames, paths and secrets from anything published as a CI
  artifact.
- Feature evidence belongs under `reports/` and is linked from `features.yaml`
  (see the `evidence-commit` skill); `scripts/validations/F-050.sh` is the
  always-on fixture half and needs no rover.

## Related

- `docs/runbooks/pc-to-jetson-promotion.md` — how a measured change actually
  reaches the rover, and how to roll it back.
- `docs/architecture/ADR-008-world-model-onnx-engine.md` — the engine decision
  this bundle revisits.
- `docs/runbooks/jetson-full-validation.md` — the full on-device validation
  pipeline this benchmark sits inside.
