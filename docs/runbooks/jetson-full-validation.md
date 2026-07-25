# Jetson Full-Validation Runbook

Operator-facing companion to `scripts/jetson_full_validation.sh` — the single
entry point that runs **static CI → cold-hardware smoke → warm-live e2e** in one
ordered, artifact-producing pass and validates every recently-merged surface
(USB-C smoke, the Claude LLM gateway, the CI gates, and the `/metrics`
observability families). Every threshold/port/path referenced below comes from
`config/jetson_production.yaml`, the Pydantic `Settings` schema, or an env
override — **no values are hardcoded** in this document or the script.

This wrapper **composes** the existing tooling (it does not re-implement it): it
delegates per-sensor probing to `scripts/verify_sensors.py`, smoke stages to
`scripts/jetson_smoke_test.sh`, the deliberative dry-run to
`scripts/translate_mission.py`, the LiDAR→WS check to
`tools/lidar_telemetry_probe.py`, and the runtime checks to the
`mousedroid.cli.preflight` / `mousedroid.cli.validate_pillars` CLIs. For a smoke
pass only, see `jetson-rover-smoke.md`.

## Prerequisites

- Repository checked out at `/opt/mousedroid` (or set `MOUSEDROID_VALIDATION_REPORT_ROOT`).
- Docker container `mousedroid` running (the script stops it for the cold phase
  and **always restarts it on exit**, even on failure).
- Host venv at `/opt/mousedroid/venv` (or set `VENV_DIR`) for the cold phase
  (the stopped container cannot be `docker exec`-ed into).
- For the cloud-tier metric assertions: `ANTHROPIC_API_KEY` exported. Absent →
  the gateway falls back to local Phi-3 and Test B asserts served-only.
- For the authenticated HTTP checks: `MOUSEDROID_TELEMETRY_TOKEN` exported.
  Absent → only the optional HTTP-ingress check (Test C) skips; `/metrics` is
  auth-exempt and needs no token.

## Quick start

```bash
bash scripts/jetson_full_validation.sh
```

Writes a stamped report directory under
`reports/jetson_full_validation/<UTC-timestamp>/` with one log per step plus
`SUMMARY.md`. Exit code is non-zero iff a **blocking** step failed (the ESP32
serial/motor/power steps are non-blocking — see below).

Useful selectors:

```bash
bash scripts/jetson_full_validation.sh --phase 1     # one phase (0-4)
bash scripts/jetson_full_validation.sh --pytest-only # hardware pytest tier only
bash scripts/jetson_full_validation.sh --dry-run     # print the plan, run nothing
bash scripts/jetson_full_validation.sh --help
```

## Phases

| Phase | What runs | Blocking? |
|-------|-----------|-----------|
| 0 — preconditions | host check, config present, key/token presence (presence only — never echoed) | yes |
| 1 — static CI (mock) | `scripts/ci.sh` (ruff/mypy/`--cov-fail-under=85`/e2e/regression/branch-cov) + `preflight --mock-hardware` + `validate_pillars --dry-run` | yes |
| 2 — cold hardware | `docker stop` → real `preflight` (appends to the trend journal: `--journal-path … --trend --journal-max-bytes …`, F-018) + `verify_sensors --sensor all` + per-stage `jetson_smoke_test.sh` + `pytest -m hardware` + real `validate_pillars` → `docker start` | mixed |
| 3 — warm live | `/api/v1/health` → `translate_mission` → live `/metrics` scrape → `lidar_telemetry_probe` → structlog greps | mixed |
| 4 — report + gate | aggregate counts → `SUMMARY.md` (rendered by `scripts/render_validation_summary.py` with a **Trend** section mined from the Phase-2 `--trend` output; inline-bash fallback on python-less hosts); exit non-zero iff any blocking failure | — |

## Cold-then-warm discipline

LiDAR / camera / GPIO need **exclusive** device access — they show false
negatives while the orchestrator owns them. Phase 2 therefore stops the
container before those checks and restarts it afterward (a `trap` guarantees the
restart even if the pass aborts). Phase 3 then runs the **warm** checks that
require the live server (`/api/v1/health`, the live `/metrics` scrape, the
LiDAR→WS probe).

## Phase-1 ci.sh OOM guard (PR #161)

The Jetson has ~7.4 GB RAM. A running `mousedroid` daemon plus the container
`ci.sh` invocation (pytest + coverage + torch + LMDB loaded together) routinely
overshoots memory and the kernel OOM killer SIGKILLs `ci.sh` with `rc=137`.
Phase 1 protects against this with a two-tier guard implemented in
`run_phase1_ci_container`:

| Attempt | Under | ci.sh mode | On success | On rc=137 |
|---------|-------|------------|------------|-----------|
| 1st | `ulimit -v ${PHASE1_CI_ULIMIT_KB}` (default 6 GB) | full | record PASS | retry (if enabled) |
| 2nd (retry) | `ulimit -v ${PHASE1_CI_RETRY_ULIMIT_KB}` (default 5 GB) + `MOUSEDROID_CI_SLIM=1` | slim: skip Perf/Regression/E2E | record **WARN** ("OOM on first attempt; passed on slim-mode retry") | record FAIL |

The retry attempt drops the memory-heaviest pytest stages via the
`MOUSEDROID_CI_SLIM=1` env var — `ci.sh` gates them behind a conditional.
Unit + Property + Integration + coverage (the core signal) always runs.
Perf/Regression/E2E coverage is NOT lost repository-wide: Phase 2's
`hardware pytest (-m hardware)` step runs those tiers in a hardware-owning
environment where memory pressure is different.

Tunables (all env-overridable — no hardcoded values):

| Env var | Default | Purpose |
| --- | --- | --- |
| `MOUSEDROID_VALIDATION_PHASE1_CI_ULIMIT_KB` | 6291456 (6 GB) | vmem cap on first attempt |
| `MOUSEDROID_VALIDATION_PHASE1_CI_RETRY_ULIMIT_KB` | 5242880 (5 GB) | vmem cap on retry |
| `MOUSEDROID_VALIDATION_PHASE1_CI_OOM_RETRY` | 1 | 1=retry on rc=137, 0=don't (operator kill-switch) |

The contract is pinned by `tests/regression/test_jetson_phase1_oom_guard.py`
(17 source-text tests) — a future edit that drops the ulimit, unlocks the
retry from rc==137, or unwraps the core Unit+Property+Integration+coverage
stage from the mandatory path will fail those pins.

Both in-container `ci.sh` invocations also pass `-e PYTHONOPTIMIZE=0`. The
image now ships `PYTHONOPTIMIZE=1` (the documented Jetson runtime contract),
but the pytest suite has only ever run with `assert` semantics intact — the
override keeps the test posture unchanged while the runtime gets `-O`.

## Strict pillar-skip gate

The Phase-2 real pillar run passes `--strict-skips` by default, so a pillar
that SKIPs because the **runtime cannot exercise it** fails the campaign
instead of reading as a 10/10 pass:

| Env var | Default | Purpose |
| --- | --- | --- |
| `MOUSEDROID_VALIDATION_PILLARS_STRICT_SKIPS` | 1 | 1=pass `--strict-skips`, 0=legacy lenient |

`PillarResult.skip_reason` discriminates the three skip classes:

- `environment` — Pattern-B delegation found no `pytest` in the runtime
  (`continual`, `meta`, `scaling`, `growth` silently unexercised). **Fails**
  under strict mode; the fix is `pip install -e ".[dev]"` in the host venv.
- `config_disabled` — the subsystem is deliberately off in cfg (`memory` and
  `curiosity` on the production overlay, which has no `memory:` block).
  **Passes** — a legitimate skip, not a gap.
- `dry_run` — the dispatcher never ran the check. `--strict-skips` is rejected
  with `--dry-run` (argparse error), so Phase 1's dry-run step never gets it.

Without the flag the CLI keeps its original contract: exit 0 on OK/DEGRADED,
1 only on FAIL.

## Config resolution matrix

Four launchers resolve four different `Settings`. The **compose path is
canonical for production**; know which one you are validating:

| Launcher | Config it loads | Notes |
|----------|-----------------|-------|
| `docker-compose.jetson.yml` (+ `mousedroid-docker.service`) | `/etc/mousedroid/jetson_production.yaml` | **Canonical production.** Synced from the repo by `scripts/sync_jetson_overlay.sh`. |
| `scripts/jetson_full_validation.sh` | repo `config/jetson_production.yaml` (override: `MOUSEDROID_JETSON_CONFIG`) | Same bytes as the canonical file **only after** an overlay sync — treat "overlay is current" as a Phase-0 precondition. |
| `.github/workflows/jetson-nightly.yml` | `/etc/mousedroid/default.yaml,/etc/mousedroid/jetson_production.yaml` | Two-overlay **stack** (deep-merged), not the prod file alone. |
| `scripts/mousedroid.service` | `/etc/mousedroid/jetson_sdcard_64gb.yaml` | **Legacy bare-metal venv path**, superseded by the Docker spine. Not exercised by this campaign. |

Env vars beat YAML in all four (`MOUSEDROID_*` with `__` nesting), which is why
per-host overrides live in `/etc/mousedroid/docker.env` and never in a commit.

## Validate-around the dead ESP32

The rover ESP32 is currently non-functional. The pass tolerates this:

- `config/jetson_production.yaml` ships **`esp32.enabled: false`** so the
  container cannot crash-loop on a dead board (`orchestrator.start()` →
  `connect()` retries then raises). The schema default stays `True`; only this
  overlay is safe-by-default. After a bench repair, lift it with
  `MOUSEDROID_ESP32__ENABLED=true` in `/etc/mousedroid/docker.env` — see the
  probe-first flow in `docs/runbooks/jetson-full-bringup.md`.
- **Preflight's `esp32` check is construction-only** — it never opens the serial
  port, so with the driver mocked it reports a vacuous OK. Do **not** read that
  line as ESP32 health; the real probe is `assert_power_chain` (post-F-008).
- The board must still **enumerate on USB**: `smoke:usbc` is blocking and
  `usbc_discovery` declares `rover_esp32` as `required: true`, and compose maps
  `${MOUSEDROID_ESP32_DEV:-/dev/ttyUSB0}` as a device. The CP2102N bridge is
  bus-powered and enumerates independently of ESP32 firmware health, so this
  normally holds. If the bridge itself drops off the bus, flip the endpoint's
  `required` and fix **both** `MOUSEDROID_ESP32_DEV` and `MOUSEDROID_LIDAR_DEV`
  (losing one CP210x renumbers `/dev/ttyUSB*`; by-id paths are unaffected).
- The hardware pytest tier and any orchestrator boot run with
  `MOUSEDROID_ESP32__ENABLED=false` → `MockESP32Driver` (resilience wrapper
  intact), so the loop ticks without the drivetrain.
- The `smoke:serial`, `smoke:motor`, and `smoke:power` steps are **non-blocking**
  (WARN, not FAIL) — they capture repair diagnostics without failing the gate.
- **No motion is ever armed.** The motion double-gate
  (`MOUSEDROID_SMOKE_ALLOW_MOTION` + `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION`)
  stays off.

## Where the #115 `/metrics` families are actually proven

`config/jetson_production.yaml` has no `openclaw:` block, so the
`POST /api/v1/mission` HTTP ingress is **not registered** — nothing drives the
orchestrator's gateway over the wire on production. Therefore:

- **Phase 3 `/metrics` scrape** only proves the endpoint is a healthy Prometheus
  surface (auth-exempt; `mousedroid_` namespace present). It does **not** assert
  the families are populated.
- **Population is proven in Phase 2** by the in-process hardware test
  `tests/hardware/test_llm_gateway_metrics_live_jetson.py::test_inprocess_mission_populates_metric_families`,
  which builds the real orchestrator, drives a **guaranteed-UNKNOWN** command
  (`"navigate to the cantina"` — rule-parsed commands like `"go forward"` /
  `"patrol …"` never reach the LLM) through `process_mission`, and asserts the
  orchestrator's shared registry renders the four families. This validates the
  exact `build_orchestrator → build_llm_gateway(metrics=…)` wiring on live Claude
  without needing `openclaw` or the HTTP server.
- The optional HTTP-ingress check (Test C / Phase-3 POST) **skips** on production
  with a logged reason. To assert HTTP-driven population, add an `openclaw:`
  block with `enabled: true` to the overlay (additive, backwards-compatible).

### Live `/metrics` grep recipe (the four #115 families)

```bash
curl -fsS "${MOUSEDROID_TELEMETRY_URL:-http://127.0.0.1:8080}/metrics" \
  | grep -E 'mousedroid_llm_(tokens_total|gateway_latency_ms|gateway_served_total|latency_budget_exceeded_total)'
```

## Triage

The `SUMMARY.md` table lists every step as PASS / WARN / FAIL with a note
pointing at its per-step log. Structured-log evidence captured in
`phase3_structlog.log` covers `usbc_endpoint_*`, `esp32_serial_port_overridden`,
`power_chain_probe_complete`, `esp32_raw_line`, `anthropic_gateway_*`, and
`fallback_gateway_started`. For per-stage smoke triage (cable reseating, by-id
drift, warm-vs-cold), see `jetson-rover-smoke.md`.

## Continuous trend sampling between runs (F-018)

The one-shot trend journal above only grows when an operator runs the full
harness. For continuous degradation monitoring, install the hourly timer:

```bash
sudo bash scripts/host_bootstrap.sh --with-trend-timer   # --dry-run first
journalctl -u mousedroid-trend -f                        # watch samples
```

Contract (pinned by `tests/regression/test_trend_timer_units.py`):

- The timer runs **non-exclusive checks only** (`config,host_env_keys` via
  `MOUSEDROID_TREND_CHECKS`) — the orchestrator container owns camera /
  LiDAR / ESP32 / audio, and a concurrent open corrupts both readers. Full
  device trends come only from this harness's Phase 2 (which stops the
  container first).
- The timer journals to a **separate path**
  (`/var/lib/mousedroid/trend/preflight.jsonl` by default) so 2-check timer
  runs never poison the full-run latency trend with bogus elapsed-time
  comparisons.
- Journal growth is capped via `--journal-max-bytes` (single-generation
  rotation to `<path>.1`; the run after a rotation reports "insufficient
  history" once, by design).
