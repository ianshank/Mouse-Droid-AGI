# Spec — credit workstreams

*(reconstructed 2026-10-19 — original in the drafting-session zip)*

Delta for WS-A…WS-D (Phase 5). Every workstream is gated on task 0.2 (key rotation). No cloud agent
ever touches the rover; no agent toolset carries an actuation tool (CHARTER §3).

## Requirement: every credit-spending run MUST pre-check its cap and append a ledger row

### Scenario: over cap refuses before any API call
- **GIVEN** `.claude/workforce.yaml` `credit.caps_usd` and a ledger whose cumulative spend for the
  workstream plus the run's estimate exceeds its cap
- **WHEN** `charge(ws, estimate_usd)` is called
- **THEN** it raises `CapExceeded`
- **AND** no client call has been made

### Scenario: the row shape
- **GIVEN** a completed run
- **WHEN** `record(entry)` appends to `reports/credit/{ledger}.jsonl`
- **THEN** the row carries `ts, ws, run_id, target, model_requested, model_resolved, resolved_via,
  input_tokens, output_tokens, cache_read_tokens, cost_usd, cost_source, cap_usd,
  cumulative_ws_usd, artifact, status`
- **AND** the API key never appears in any row or artifact

### Scenario: caps are configuration
- **GIVEN** `tools/{credit_ledger}.py`
- **WHEN** its caps are read
- **THEN** they come from `.claude/workforce.yaml` (`caps_usd: {A: 10, B: 60, C: 15}` and the
  `ledger:` path), never from a code literal

## Requirement: model ids MUST be resolved at run time, never asserted

### Scenario: alias to id
- **GIVEN** a requested alias (`haiku`, `sonnet`, `opus`)
- **WHEN** a tool starts
- **THEN** it resolves the id via `client.models.retrieve()` and writes both the alias and the
  resolved id into its artifact and ledger row (`resolved_via: models.retrieve`)

### Scenario: no `cache_control`
- **GIVEN** any Phase-5 request builder
- **WHEN** its kwargs are inspected
- **THEN** no `cache_control` block is present (R-10: a no-op on a ~70-token system prompt)

## Requirement: WS-A — the Anthropic request profile MUST be byte-identical when absent and MUST NOT degrade silently when present

### Scenario: absent block, unchanged request
- **GIVEN** `LLMConfig.anthropic is None`
- **WHEN** `_create_message` runs through the `sdk=` seam of
  `tests/unit/llm_gateway/test_anthropic_gateway.py`
- **THEN** `temperature` is sent and neither `thinking` nor `output_config` is — exactly today's
  request

### Scenario: present block omits sampling params
- **GIVEN** `anthropic: {omit_sampling_params: true, thinking: "disabled"}`
- **WHEN** `_create_message` runs
- **THEN** `temperature` is absent and `thinking == {"type": "disabled"}`

### Scenario: the probe proves the primary served
- **GIVEN** the retargeted overlay on the Jetson
- **WHEN** `python tools/llm_latency_probe.py --report-json <path>` runs
- **THEN** the report carries `model_requested`, `model_resolved`, `served`, `cold_start_ms`,
  `p50_ms`, `p95_ms`, `p99_ms`, `iterations`, `passed`
- **AND** `served == "primary"`; a degraded primary (HTTP 400 on `temperature` [Likely — per the
  `claude-api` skill's model table, cached 2026-10-06; not re-verified against the live API]) is
  reported as `fallback`, never as a pass

### Scenario: the overlays and the golden move together
- **GIVEN** `config/jetson_production.yaml` and `config/jetson_claude_pilot.yaml` retargeted
- **WHEN** the regression suite runs
- **THEN** the metrics sample seeds and `tests/regression/fixtures/render_prometheus_golden.txt`
  name the same id
- **AND** legacy `llm:` YAML without the `anthropic` key still loads, and
  `AnthropicRequestConfig().omit_sampling_params is True`

## Requirement: WS-B / WS-D — a cloud agent MUST NOT hold an actuation or write tool

### Scenario: the allowlist is read-only
- **GIVEN** `tools/agent_reviewer/{__init__,run_review,prompt,policy,ledger}.py`
- **WHEN** its allowlist is checked against the `MCPConfig.actuation_tools` defaults and the
  literal denylist (`set_velocity`, `emergency_stop`, `translate_nl_mission`, `export_experience`,
  `tensorrt_compile`, `calibrate_ultrasonic`)
- **THEN** no actuation tool, no `Write`/`Edit`/`MultiEdit`/`NotebookEdit`, and no
  `WebFetch`/`WebSearch` is allowed
- **AND** `Bash` is limited to `git diff|log|show` and `pytest --collect-only`

### Scenario: the bash hook denies mutation
- **GIVEN** the reviewer's PreToolUse hook
- **WHEN** `rm`, `git push` or a bare `pytest` is requested
- **THEN** it is denied; `git diff` is allowed

### Scenario: the MCP is mock-only
- **GIVEN** the reviewer's `mcp_servers` built from `.mcp.json`
- **WHEN** inspected
- **THEN** `MOUSEDROID_MOCK_HARDWARE=true` is forced and no rover host is configured

### Scenario: the prompt builder refuses secrets
- **GIVEN** a repo path matching `*.env`, `docker.env*` or a `.gitleaks.toml` pattern
- **WHEN** the prompt builder is asked to include it
- **THEN** it refuses

### Scenario: the prompt is the house reviewer
- **GIVEN** `.claude/agents/peer-reviewer.md`
- **WHEN** `prompt.build()` runs
- **THEN** the front-matter is stripped and the house output shape (verdict table,
  severity-ordered defects with `[Certain]/[Likely]/[Guessing]`, survives, load-bearing pins,
  verdict) is appended
- **AND** the sentence "`esp32_diagnostics` is a stub until task 4.3 — never cite it as evidence"
  is present

### Scenario: cost from the SDK, into the ledger
- **GIVEN** a finished review
- **WHEN** the artifact is written under `reports/agent_reviewer/<date>/`
- **THEN** cost comes from `ResultMessage.total_cost_usd`
- **AND** the ledger row is ≤ the cap, and `max_turns` came from `.claude/workforce.yaml`

## Requirement: WS-C — Batch triage MUST propose, never delete

### Scenario: chunking and request shape
- **GIVEN** a vulture findings file
- **WHEN** `chunk_by_module` and `build_request` run
- **THEN** findings are grouped by `src/mousedroid/<pkg>`, each request carries ±20-line excerpts
  and the package's Protocol/validator names, and the output schema is
  `{name, filename, lineno, verdict: keep|delete|verify, reason}`

### Scenario: results keyed by `custom_id`
- **GIVEN** a completed batch
- **WHEN** `collect` runs
- **THEN** results are matched by `custom_id`, never by position

### Scenario: the document ranks and deletes nothing
- **GIVEN** `docs/analysis/dead-code-triage-<date>.md`
- **WHEN** rendered
- **THEN** `delete` verdicts are ranked by confidence, then `verify`, then `keep`
- **AND** the PR contains zero `src/` deletions; any later deletion PR runs
  `bash scripts/prove_pin_fails.sh` first

### Scenario: the local-only input chain is declared
- **GIVEN** `reports/dead_code/` is gitignored
- **WHEN** the document is written
- **THEN** it records the findings file's sha256 and the vulture version
- **AND** names `python scripts/dead_code_audit.py` as the regeneration command

## Requirement: the hot loop MUST stay free of the cloud SDKs (CHARTER §4 invariant 10; Ruling 22, D-14)

### Scenario: the purity pin names both SDKs
- **GIVEN** `tests/regression/test_hot_loop_import_purity.py::_FORBIDDEN_MODULES`
- **WHEN** task 5.4 lands
- **THEN** the tuple contains `"anthropic"` and `"claude_agent_sdk"` beside the #242 names
  (`google.adk`, `honcho`, `composio`, `composio_core`)
- **AND** importing `mousedroid.orchestrator.autonomous` and `mousedroid.factory.mission` loads
  neither — the gateway imports the SDK lazily in `start()`, and the reviewer lives under `tools/`
