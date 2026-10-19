# MouseDroid MCP — Operator Guide

This document shows external Model Context Protocol (MCP) clients how to
connect to a running MouseDroid instance.

The MCP server is **disabled by default**. Two settings need to flip:

* `mcp.enabled: true` — instantiate the server (otherwise
  `build_mcp_server` returns `None`)
* `mcp.bind_transport: true` — actually bind the configured transport.
  When `false` (default) the server idles in-process; useful for tests
  and embedded integrations, but external clients won't reach it.

Both can be set in YAML or via environment variables:

```bash
export MOUSEDROID_MCP__ENABLED=true
export MOUSEDROID_MCP__BIND_TRANSPORT=true
```

---

## Claude Desktop

Add a `mcpServers` block to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "mousedroid": {
      "command": "python",
      "args": [
        "-m", "mousedroid",
        "--config", "/absolute/path/to/config/default.yaml"
      ],
      "env": {
        "MOUSEDROID_MOCK_HARDWARE": "true",
        "MOUSEDROID_MCP__ENABLED": "true",
        "MOUSEDROID_MCP__BIND_TRANSPORT": "true",
        "MOUSEDROID_MCP__TRANSPORT": "stdio"
      }
    }
  }
}
```

Restart Claude Desktop after editing. The server appears in the
"Connected" list when the next conversation starts.

## Claude Code

Add a `.mcp.json` at the project root (or the configured config dir):

```json
{
  "mcpServers": {
    "mousedroid": {
      "command": "python",
      "args": [
        "-m", "mousedroid",
        "--config", "config/default.yaml"
      ],
      "env": {
        "MOUSEDROID_MCP__ENABLED": "true",
        "MOUSEDROID_MCP__BIND_TRANSPORT": "true",
        "MOUSEDROID_MCP__TRANSPORT": "stdio"
      }
    }
  }
}
```

## Networked transports (SSE / streamable HTTP) — not yet supported

The `sse` and `streamable_http` transports are **deferred to a follow-up
PR**. Setting `mcp.transport=sse` (or `streamable_http`) with
`mcp.bind_transport=true` will currently raise `NotImplementedError`
at startup with a pointer to this section.

Why deferred:

* `mcp.server.sse.SseServerTransport.connect_sse` is an
  `@asynccontextmanager`, not an ASGI app — proper integration needs a
  Starlette `Route` that enters the context inside its handler and
  separately mounts `transport.handle_post_message` for client POSTs.
* The bearer token validated by `MCPConfig` is not yet propagated into
  per-request `call_tool(token=...)`, so any SSE/HTTP bind would
  bypass the existing auth check on the bridge.

Tracked in [docs/MCP_NEXT_STEPS.md](MCP_NEXT_STEPS.md) under the P0
"transport bind-up" follow-ups. Until then, **stdio is the only
production-supported transport** — it covers Claude Desktop, Claude Code,
and any subprocess launcher (systemd, container ENTRYPOINT).

---

## Available tools (default configuration)

| Tool                  | Type        | Notes                                                    |
|-----------------------|-------------|----------------------------------------------------------|
| `health_check`        | non-actuation | Always available. Used by clients to confirm liveness.  |
| `read_encoders`       | read-only   | Returns the latest wheel encoder + odometry snapshot.    |
| `esp32_diagnostics`   | read-only   | Reports whether the ESP32 answers. Never moves or stops the rover and is NOT in `actuation_tools`, so it stays reachable. See below. |
| `set_velocity`        | actuation   | Hidden unless `mcp.expose_actuation_tools: true`. Clamped to `cfg.esp32.max_velocity_mps` / `max_omega_rads`. Refused when the safety monitor is in emergency state. |
| `emergency_stop`      | (special)   | NOT in `actuation_tools` — always callable, even during a safety emergency. |
| `tensorrt_compile`    | actuation   | Hidden unless `mcp.expose_actuation_tools: true`.        |
| `calibrate_ultrasonic`| actuation   | Hidden unless `mcp.expose_actuation_tools: true`.        |
| `export_experience`   | actuation   | Hidden unless `mcp.expose_actuation_tools: true`.        |

### Reading `esp32_diagnostics`

Only `status: ok` is evidence that the board answered; each other status says why it did not.

| `status`       | Meaning |
|----------------|---------|
| `unconfigured` | The registry was built without `Settings`, so there was nothing to ask. |
| `mock`         | `mock_hardware` is on or `esp32.enabled` is `false` (the shipped production overlay): the driver is a mock, so no board was asked. |
| `ok`           | A stock `FEEDBACK_BASE_INFO` frame arrived. `source` is `live_driver` (read through the orchestrator's own driver, which holds the exclusive serial port and is never reopened) or `probe` (a fresh connect, for a process that holds no port). `battery_v` is the bus voltage, or `null` when the board reported none. |
| `no_frame`     | `source: live_driver`: no frame within `esp32.max_consecutive_timeouts` polls. |
| `no_response`  | `source: probe`: the same, after a fresh connect. `detail` names the port and baud. |
| `error`        | The read or the probe raised, or `esp32.command_set` is not `waveshare_stock` (the legacy dialect has no frame to witness). `detail` is `<Type>: <message>`. A failure is returned, not raised, so a dead board cannot trip the MCP circuit breaker for the other tools. |

To enable actuation tools:

```yaml
mcp:
  expose_actuation_tools: true
```

## Available resources

| URI                                          | Notes                                          |
|----------------------------------------------|------------------------------------------------|
| `mousedroid://telemetry/recent`              | Last N telemetry frames (config-driven N).     |
| `mousedroid://logs/tail`                     | Recent structured log lines, redacted.         |
| `mousedroid://config/redacted`               | Current `Settings` snapshot, secrets masked.   |
| `mousedroid://memory/episodes/recent`        | Episodic memory tail (when memory tier on).    |

All resource limits live under `mcp.resources` in the schema; bump
`recent_frames_max` or `log_tail_max` rather than hard-coding.

## Verifying the connection

Before exposing the server to a real client, confirm health:

```bash
python -m mousedroid --config config/default.yaml --health-check
```

This boots the orchestrator, runs a single tick, and exits non-zero on
any subsystem failure.

## Hardware smoke — motion safety

`tests/hardware/test_motor_smoke.py` is gated by
`ESP32Config.smoke_test_allow_motion` (default **False**). With the
default the smoke test exercises the connect → send(0,0,0) → read →
e-stop → disconnect flow without spinning the wheels — safe to run
unattended with the rover on a table.

To exercise actual motion (rover on rollers, tethered, or otherwise
monitored), opt in explicitly:

```bash
MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true \
  python -m pytest tests/hardware/test_motor_smoke.py -v -m hardware
```

`scripts/jetson_full_smoke_run.sh` does NOT set this variable; the
`mcp_motor_smoke` stage therefore runs in motion-disabled mode by
default. Override with the env var above (or a YAML overlay) only when
the rover environment is safe.

## Telemetry the operator should watch

Names are the series exactly as `/metrics` exposes them, so they can be pasted
into a query. Counters carry `_total`; the histogram is queried through `_bucket`.

| Metric                                           | Meaning                                             |
|--------------------------------------------------|-----------------------------------------------------|
| `mousedroid_mcp_requests_total`                  | Total MCP requests served.                          |
| `mousedroid_mcp_tool_calls_total{tool, result}`  | Tool dispatch outcomes (ok/error/denied/refused/…). |
| `mousedroid_mcp_request_latency_ms_bucket`       | Histogram for p50/p95/p99 latency dashboards.       |

Three Grafana panels and a `mousedroid_mcp` alert group are shipped
with this release; see [grafana_dashboard.json](grafana_dashboard.json)
and [`config/prometheus/alerts.yml`](../config/prometheus/alerts.yml).
