# Test Validation & Jetson Deploy Prep — Reference

> **Branch:** `feat/test-validation-and-jetson-deploy-prep`
> **Date:** 2026-07-24
> **Base:** `origin/HEAD` (commit `e3a834e`)

## What this branch is for

This branch tracks the work of running the complete MouseDroid test pyramid
locally on a Windows dev host and preparing deployment artifacts for the
Jetson Orin Nano (not yet attached).

## Key files

| File | Purpose |
|---|---|
| `pyproject.toml` | Package deps, pinned ruff/mypy, test config |
| `.github/workflows/ci.yml` | 14-job CI pipeline definition |
| `scripts/ci.sh` | Local CI script (bash — not directly runnable on Windows) |
| `deployments/jetson-image.json` | Deploy record — SHA pin for config-compat gate |
| `config/jetson_production.yaml` | Active Jetson config overlay |
| `Dockerfile.jetson` | Multi-stage Jetson L4T container build |
| `docker-compose.jetson.yml` | Compose service with hardware passthrough |
| `scripts/jetson_full_validation.sh` | 3-phase on-device validation |
| `tests/conftest.py` | Root conftest — forces `MOUSEDROID_MOCK_HARDWARE=true` |

## Test pyramid

| Tier | Directory | Count (approx) | Gating |
|---|---|---|---|
| Unit | `tests/unit/` | ~170 files | 85% line coverage |
| Property | `tests/property/` | ~20 files | Part of 85% gate |
| Integration | `tests/integration/` | ~61 files | Part of 85% gate |
| Regression | `tests/regression/` | ~70 files | No coverage gate |
| E2E | `tests/e2e/` | ~10 files | No coverage gate |
| Performance | `tests/performance/` | ~8 files | Latency budgets |
| Smoke | `tests/smoke/` | ~16 files | Sub-second sanity |
| Hardware | `tests/hardware/` | ~25 files | Jetson-only (`@pytest.mark.hardware`) |

## Current blockers (from NEXT_STEPS.md)

1. **P0:** ESP32 physically dead — blocks F-008 / autonomous motion
2. **P0:** ANTHROPIC_API_KEY exposed — needs rotation
3. **P1:** `/opt/mousedroid` ownership drift on Jetson
