# MouseDroid MCP — Operator Guide

This document shows external Model Context Protocol (MCP) clients how to
connect to a running MouseDroidAGI instance.

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

## Networked transports (SSE / streamable HTTP)

For multi-machine setups bind on the loopback (or a private network
behind firewall rules) and require a bearer token:

```yaml
# config/local_mcp.yaml — overlay this on top of default.yaml
mcp:
  enabled: true
  bind_transport: true
  transport: streamable_http   # or "sse"
  host: 127.0.0.1              # MUST be loopback unless a token is set
  port: 8765
```

Set the bearer token in the environment (`MCPConfig` validators reject
non-loopback bindings without one):

```bash
export MOUSEDROID_MCP_TOKEN="$(openssl rand -hex 32)"
```

Clients send the token via `Authorization: Bearer <token>`. See
`src/mousedroid/mcp/auth.py` for the validator.

---

## Available tools (default configuration)

| Tool                  | Type        | Notes                                                    |
|-----------------------|-------------|----------------------------------------------------------|
| `health_check`        | non-actuation | Always available. Used by clients to confirm liveness.  |
| `read_encoders`       | read-only   | Returns the latest wheel encoder + odometry snapshot.    |
| `set_velocity`        | actuation   | Hidden unless `mcp.expose_actuation_tools: true`. Clamped to `cfg.esp32.max_velocity_mps` / `max_omega_rads`. Refused when the safety monitor is in emergency state. |
| `emergency_stop`      | (special)   | NOT in `actuation_tools` — always callable, even during a safety emergency. |
| `tensorrt_compile`    | actuation   | Hidden unless `mcp.expose_actuation_tools: true`.        |
| `calibrate_ultrasonic`| actuation   | Hidden unless `mcp.expose_actuation_tools: true`.        |
| `export_experience`   | actuation   | Hidden unless `mcp.expose_actuation_tools: true`.        |

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

## Telemetry the operator should watch

| Metric                                           | Meaning                                             |
|--------------------------------------------------|-----------------------------------------------------|
| `mousedroid_mcp_requests`                        | Total MCP requests served.                          |
| `mousedroid_mcp_tool_calls{tool, result}`        | Tool dispatch outcomes (ok/error/denied/refused/…). |
| `mousedroid_mcp_request_latency_ms_bucket`       | Histogram for p50/p95/p99 latency dashboards.       |

Three Grafana panels and a `mousedroid_mcp` alert group are shipped
with this release; see [grafana_dashboard.json](grafana_dashboard.json)
and [`config/prometheus/alerts.yml`](../config/prometheus/alerts.yml).
