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
  (`llm_latency_probe.py --report-json PATH` also writes the model-retarget evidence: which tier served)
- `credit_ledger.py` — F-059 credit workstreams: `charge(ws, estimate_usd)` refuses an API call before it is
  made when the workstream's recorded spend would pass its cap, `record(entry)` appends one JSON line per run;
  caps and the ledger path come from `credit:` in `.claude/workforce.yaml`
- `agent_reviewer/` — F-059 WS-B / WS-D: a read-only Agent-SDK peer reviewer
  (`python -m tools.agent_reviewer.run_review --base REF --head REF --cap-usd N [--dry-run]`). The head
  under review is treated as hostile: the trust boundary is the base ref (the reviewer's configuration,
  instructions and MCP declaration are read from it), and the tool allowlist, the git-only Bash guard and the
  mock-only MCP entry (built in code) are `policy.py`; every run is pre-checked against and recorded in the
  credit ledger. Needs the `agent_reviewer` extra; see its `README.md` for the safe setup
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
