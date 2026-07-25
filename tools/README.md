# Developer Tools

Probes and dev utilities (not shipped in the runtime path):

- `claude_hooks/` — Claude Code workforce governance (F-024): edit-time secret
  scan, capability freeze gate, and advisory post-edit checks, plus the reusable
  primitives they share (config, paths/globs, hook I/O, stderr logging,
  portability). Configured by `.claude/workforce.yaml`; operator guide in
  `docs/runbooks/claude-workforce-hooks.md`; architecture in
  `docs/architecture/c4-claude-workforce.md`.
- `dashboard_proxy.py` — workstation → Jetson auth-gated telemetry reverse proxy
- `doc_hygiene.py` — `NEXT_STEPS.md` budget guard
- `validate_skill_commands.py` — skill-doc path / host hygiene (CLI + importable library)
- `llm_latency_probe.py`, `lidar_telemetry_probe.py`, `jetson_remote_llm_probe.py` — latency / telemetry probes
- `spikes/` — throwaway investigation spikes

`tools/` is inside the `ruff` lint + format scope — in `pyproject.toml`, in
`scripts/ci.sh`, and (since F-024) in `.github/workflows/ci.yml`, which had
silently omitted it.

`tools/claude_hooks/` additionally carries `mypy --strict` and a dedicated
coverage gate (`--cov=tools/claude_hooks --cov-branch`, threshold read from
`coverage.tools_line_min`), running in both `scripts/ci.sh` and the
`local-gates` job in `.github/workflows/ci.yml`, because the repository-wide
coverage gate measures `src/mousedroid` only and cannot see this directory.
Neither applies to the rest of `tools/` yet.

**Import discipline:** `claude_hooks/` must never import `mousedroid` — a hook
runs on every Write/Edit, and pulling in the runtime package would put
torch/faiss/lmdb on the edit path. Pinned by
`tests/regression/test_claude_workforce_aqa.py`.
