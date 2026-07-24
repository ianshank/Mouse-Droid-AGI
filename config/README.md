# Configuration

Pydantic-validated YAML. `default.yaml` holds all defaults; platform overlays layer on top
(`platform: mouse_droid | robot_arm`). Environment overrides use the `MOUSEDROID_` prefix with `__` nesting
(e.g. `MOUSEDROID_ESP32__ENABLED=false`).

- `default.yaml` — base (mock hardware, safe thresholds)
- `jetson_*.yaml` — Jetson Orin Nano production / pilot overlays
- `robot_arm_*.yaml` — the parked arm platform
- `*_training.yaml` — offline training overlays
- `*.example` — env-file / config templates (never commit real secrets; see `docker.env.example`)
- `prometheus/`, `grafana/`, `loki/` — monitoring-stack configs

The schema is `src/mousedroid/config/schema.py`. **New config fields MUST carry a default** so existing YAML
loads unchanged (enforced by the `config-compat` CI gate).
