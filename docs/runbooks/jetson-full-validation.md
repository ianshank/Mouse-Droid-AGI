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
| 2 — cold hardware | `docker stop` → real `preflight` + `verify_sensors --sensor all` + per-stage `jetson_smoke_test.sh` + `pytest -m hardware` + real `validate_pillars` → `docker start` | mixed |
| 3 — warm live | `/api/v1/health` → `translate_mission` → live `/metrics` scrape → `lidar_telemetry_probe` → structlog greps | mixed |
| 4 — report + gate | aggregate counts → `SUMMARY.md`; exit non-zero iff any blocking failure | — |

## Cold-then-warm discipline

LiDAR / camera / GPIO need **exclusive** device access — they show false
negatives while the orchestrator owns them. Phase 2 therefore stops the
container before those checks and restarts it afterward (a `trap` guarantees the
restart even if the pass aborts). Phase 3 then runs the **warm** checks that
require the live server (`/api/v1/health`, the live `/metrics` scrape, the
LiDAR→WS probe).

## Validate-around the dead ESP32

The rover ESP32 is currently non-functional. The pass tolerates this:

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
