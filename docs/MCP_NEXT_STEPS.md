# MCP Module — Next Steps

This document tracks follow-up work for the optional MCP server introduced in
`feat(mcp): add Model Context Protocol server module` (PR #52). The first PR
landed the bridge, resource providers, prompts, auth, lifecycle, factory wiring,
and metrics — everything except the actual transport bind-up to the upstream
`mcp` SDK is in place and reaches **99.40% coverage**.

The bridge and providers are SDK-agnostic and exercised end-to-end by the
existing tests. Items below are the remaining work to make the server reachable
from external MCP clients and to harden it for field operation.

## P0 — SDK adapter and transport bind-up

- [ ] Wire `mcp.server.Server` into `MouseDroidMCPServer._serve_loop` for the
      `stdio` transport. Map:
  - `list_tools()` → `MouseDroidMCPServer.list_tool_names()` (with descriptions
    pulled from `ToolRegistry.get(name).description`)
  - `call_tool(name, args)` → `MouseDroidMCPServer.call_tool(name, args, peer="stdio")`
  - `list_resources()` → `MouseDroidMCPServer.list_resource_uris()`
  - `read_resource(uri)` → `MouseDroidMCPServer.read_resource(uri, peer="stdio")`
  - `list_prompts()` / `get_prompt(name)` → `default_prompts()`
- [ ] Add the `sse` transport adapter using `mcp.server.sse.SseServerTransport`.
      Bind to `cfg.host` / `cfg.port`; refuse non-loopback bind without a token
      (already enforced by `MCPConfig` validators at config load).
- [ ] Add the `streamable_http` transport adapter; share the auth check above.
- [ ] Once a real adapter exists, drop the `# pragma: no cover` markers in
      `server.py:_serve_loop` and add an integration test using
      `mcp.client.stdio.stdio_client` against the actual server.

## P0 — Hardware smoke

- [ ] Run a smoke pass on a Jetson Orin Nano with `MOUSEDROID_MCP__ENABLED=true`
      enabled alongside the standard mock-hardware path:
  ```bash
  MOUSEDROID_MOCK_HARDWARE=true \
  MOUSEDROID_MCP__ENABLED=true \
      python -m mousedroid --config config/default.yaml
  ```
- [ ] Verify the orchestrator's 30 Hz tick stays inside its `tick_timeout_s`
      budget while a client polls `mousedroid://telemetry/recent` at 5 RPS over
      stdio.
- [ ] Confirm `mcp_request_latency_ms` histogram shows expected p50 (<10 ms for
      `health_check`).

## P1 — Operator UI

- [x] Ship a minimal connection recipe for **Claude Desktop**: a JSON snippet
      for `mcp.servers.mousedroid` pointing at `python -m mousedroid` with the
      stdio transport. Include in `docs/MCP_OPERATOR_GUIDE.md`.
- [x] Same for **Claude Code** (`.mcp.json` template).

## P1 — Telemetry / dashboards

- [ ] Add a Grafana panel to `docs/grafana_dashboard.json` for the new
      `mcp_requests_total`, `mcp_tool_calls_total{tool,result}`, and
      `mcp_request_latency_ms` series. Pin the breaker to
      `tool_bridge.BREAKER_NAME = "mcp_tool_call"`.
- [ ] Wire alerts: e.g. p95 of `mcp_request_latency_ms` > 500 ms over 5 min →
      warn; `mcp_tool_calls_total{result="circuit_open"}` rate > 1 / min →
      page (only when MCP is enabled).

## P2 — Hardening

- [ ] Per-tool timeout overrides in `MCPConfig` — current scalar
      `request_timeout_s` is sufficient for diagnostics but coarse for
      `tensorrt_compile` (long-running). Add `tool_timeouts: dict[str, float]`
      with fallback to the global.
- [ ] `MCPConfig.actuation_tools` is currently a flat list of names. Promote to
      a dict so each entry can carry an explicit "requires safety re-check"
      flag, defaulting to `True`.
- [ ] Scope tokens to specific tool sets (multi-tenant / multi-client). Today
      auth is binary; a richer model (`{token: {tools: [...], resources: [...]}}`)
      is a follow-up.
- [ ] Add a config flag to expose the per-session rate-limit budget as a
      header on every response so well-behaved clients can throttle preemptively.

## P2 — Test hardening

- [ ] Hypothesis property: dispatch ↔ visibility — a tool that's visible in
      `MCPConfig` X is dispatchable, and vice versa, across random
      allow/deny/actuation combinations. (Partial coverage exists; the property
      does not yet model the actuation toggle.)
- [ ] Slow-test marker covering a 60-second sustained load (5 RPS) measuring
      p99 request latency and zero `mcp_tool_calls_total{result="circuit_open"}`.
- [ ] Add a test that confirms the `_DoubleLabeledCounter` Prometheus rendering
      stays stable under concurrent updates from an `asyncio.gather` of bridge
      calls.

## P3 — Future surfaces

- [ ] Expose the **arm task** (Tower of Hanoi / laundry sorting) state machine
      as MCP resources when `cfg.platform == robot_arm`:
      `mousedroid://arm/task/state`, `mousedroid://arm/plan/current`. Implementation
      can mirror `MemoryResourceProvider` against the planner state object.
- [ ] First-class **subscription** (server-push) for `mousedroid://logs/tail` so
      MCP clients can stream new entries instead of polling.
- [ ] Replace the regex redactor with a typed JSON Schema → Pydantic-driven
      redactor once `Settings.model_json_schema()` exposes `secret: true`
      annotations.

## Tracking

- PR introducing this module: [#52](https://github.com/ianshank/Mouse-Droid-AGI/pull/52)
- Architecture diagram (Level 3f): `docs/architecture.md`
- Module entry point: `src/mousedroid/mcp/`
- Plan that bootstrapped the work: `/root/.claude/plans/create-a-plan-for-replicated-squid.md`
