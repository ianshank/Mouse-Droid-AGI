---
description: Tiered Jetson smoke testing discipline with operator consent gates for live actuation
status: active
---

# Jetson Smoke Testing

Tiered smoke discipline for on-device validation. Documents both the operator
consent mechanism and the mechanical environment-variable gates.

## Tier 1 — Passive Probes (always safe)

System health, sensor enumeration, config validation. No actuation.

```bash
bash scripts/jetson_smoke_test.sh system
bash scripts/jetson_smoke_test.sh camera
bash scripts/jetson_smoke_test.sh lidar
```

## Tier 2 — Active Sensors (safe, no motion)

Speaker playback, microphone capture, GPIO reads. Non-destructive.

```bash
bash scripts/jetson_smoke_test.sh speaker
bash scripts/jetson_smoke_test.sh gpio
bash scripts/jetson_smoke_test.sh serial
```

## Tier 3 — Live Actuation (requires consent)

Motor motion, wheel drive, arm actuation. Requires BOTH:
1. **Typed consent phrase:** operator types `RUN-MOTION` at the prompt
2. **Environment gates** (mechanical, not honor-system):
   - `MOUSEDROID_SMOKE_ALLOW_MOTION=1`
   - `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=1`

Physical prerequisites: chassis raised off ground, arm clear of obstacles.

## Full Validation Suite

The consolidated entry point composes all tiers:

```bash
bash scripts/jetson_full_validation.sh
```

This runs `scripts/ci.sh`, `scripts/verify_sensors.py`,
`scripts/jetson_smoke_test.sh`, and the preflight/validate_pillars checks.

## Guardrails

- Steps depending on broken hardware (ESP32 today) are non-blocking WARNs
- No hardcoded values in scripts — ports, timeouts, and retries are env-overridable
- No `assert` in inline shell-python under `PYTHONOPTIMIZE=1`
- Reports go to `smoke-reports/` (evidence.tracked_roots)
