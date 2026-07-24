# Developer Tools

Probes and dev utilities (not shipped in the runtime path):

- `dashboard_proxy.py` — workstation → Jetson auth-gated telemetry reverse proxy
- `doc_hygiene.py` — `NEXT_STEPS.md` budget guard
- `validate_skill_commands.py` — skill-doc path / host hygiene (CLI + importable library)
- `llm_latency_probe.py`, `lidar_telemetry_probe.py`, `jetson_remote_llm_probe.py` — latency / telemetry probes
- `spikes/` — throwaway investigation spikes

`tools/` is inside the `ruff` lint + format scope (see `pyproject.toml`).
