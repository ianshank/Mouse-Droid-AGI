# Enterprise-Hardening Refactor — Notes & Deliverable Mapping

This pass applied an "enterprise production standards" refactoring brief to a
codebase that already met most of that bar. Rather than a broad rewrite (which
would have been high-risk churn against 62k LOC and would have violated the
project's own invariants), the work was scoped to the genuine gaps. Full
rationale and the rejected alternatives live in
[`ADR-014`](../architecture/ADR-014-cyclomatic-complexity-gate.md).

## What changed

| Area | Change | Behaviour impact |
|------|--------|------------------|
| Complexity gate | `C901` + `max-complexity = 15` in `pyproject.toml` | none (lint-only) |
| `render_prometheus` | 624-line method → 13 `_families_*` helpers | **byte-identical** output (golden-pinned) |
| `telemetry/server.py` | `_handle_mission_post` / `_broadcast_loop` decomposed | none (164 tests pass) |
| `orchestrator.py` | `start` / `stop` decomposed by lifecycle phase | none (239 tests pass) |
| Thin modules | dedicated unit suites (`growth`/`meta`/`scaling`/`efficiency`/`logging`) | additive; 96–100 % module coverage |

There are **no breaking changes** — no public signature, config field, metric
name, structlog event, or YAML contract was altered. Therefore **no migration
guide is required**.

## Mapping the brief's literal deliverables onto this codebase

The brief asked for several artefacts that assume a greenfield/legacy layout.
This project's established patterns already fulfil their *intent*; creating the
literal files would duplicate or undercut existing invariants.

| Brief deliverable | This codebase's equivalent | Why not the literal file |
|-------------------|----------------------------|--------------------------|
| `config/defaults.py` | `src/mousedroid/config/schema.py` — Pydantic `BaseSettings`, defaults on every field | Architecture Invariant #3: config is single-source Pydantic + YAML. A parallel `.py` defaults module would fork the source of truth. |
| `config/development.py`, `config/production.py` | `config/*.yaml` overlays (`default.yaml`, `jetson_production.yaml`, `mock_hardware.yaml`, …) selected by `platform:` + env | Env-specific config is data (YAML), not code, and is validated against the schema in CI (`scripts/validate_configs.py`). |
| `.env.example` | `config/docker.env.example`, `config/.env.jetson.example` | Already present; documents the secret surface (incl. `ANTHROPIC_API_KEY`) without live values. |
| Lower coverage to 70 % | Kept at **85 %** (dual gate: total line + per-changed-file branch) | The existing gate already exceeds the 70 % ask; lowering it would be a regression. |
| Custom exception hierarchy / structured logging / no bare excepts | Already in place (`structlog`, zero bare `except:`, module exception types) | Verified by audit; no gap to fill. |
| Cyclomatic complexity < 15 | **Newly enforced** via `C901` (this pass) | This was the one genuinely missing quality gate. |

## Verifying the changes

```bash
# Lint incl. the new complexity gate, across all CI lint scopes
python -m ruff check src/ tests/ scripts/ && python -m ruff format --check src/ tests/ tools/

# Type check
python -m mypy --strict src/mousedroid/

# The byte-identical Prometheus guarantee + the new module suites
python -m pytest tests/regression/test_render_prometheus_golden.py \
  tests/unit/{logging,meta,scaling,growth,efficiency} \
  --no-cov --import-mode=importlib

# Coverage-gated core suite
python -m pytest tests/unit tests/property tests/integration \
  --cov=src/mousedroid --cov-fail-under=85 --import-mode=importlib
```

## Follow-ups (deliberately not done)

* **No factory `_build_optional_driver` helper.** Only `build_microphone` /
  `build_speaker` share the shape and they diverge in logging — a helper would
  be a net-negative abstraction. See ADR-014 § Rejected alternatives.
* **MAML left first-order.** `meta_step` does not propagate gradients to the base
  model; a second-order fix is a behaviour change out of scope here and is
  characterized by a test instead.
* **Large-file splits (`factory.py`, `schema.py`) not attempted in this pass** —
  chosen scope excluded them to avoid large low-value diffs. Both were later
  split: `schema.py` in commit `4646d80` (PR #191), `factory.py` in the
  god-files decomposition (see ADR-017).
