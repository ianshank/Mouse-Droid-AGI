# API Reference

The telemetry server (`src/mousedroid/telemetry/server/`) exposes a config-driven aiohttp REST + WebSocket
surface. Paths derive from config (`api_prefix`, `ws_path`, `lidar_raw_ws_path`, metrics path) — the tables
below show the defaults. Auth is a bearer token when a telemetry token is configured; `/metrics` and the
dashboard pages follow the server's own auth policy.

## REST (prefix `/api/v1`)

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/v1/status`       | Orchestrator + subsystem status |
| GET  | `/api/v1/sensors`      | Latest sensor frame |
| GET  | `/api/v1/health`       | Liveness / health summary |
| GET  | `/api/v1/network`      | Network / interface info |
| GET  | `/api/v1/logs`         | Recent structlog ring buffer |
| GET  | `/api/v1/logs/stream`  | Server-sent-events log stream |
| GET  | `/api/v1/health/cloud` | Cloud-LLM tier health (when the LLM tier is enabled) |
| POST | `/api/v1/mission`      | Submit an NL mission (when the mission route is enabled) |

## WebSocket & streaming

| Path | Purpose |
|------|---------|
| `ws_path` (WS)            | Live telemetry frames (`TelemetryFrame.to_dict()`, incl. the `fused` summary) |
| `lidar_raw_ws_path` (WS)  | Raw LiDAR scan stream |
| `/camera/stream`          | MJPEG camera stream |
| `/camera/frame.jpg`       | Single JPEG snapshot |

## Pages & metrics

| Path | Purpose |
|------|---------|
| `/` → `/dashboard` | Unified camera + lidar + sensor-fusion + status page |
| `/lidar`, `/camera` | Per-sensor pages |
| `/metrics` | Prometheus text exposition (namespaced via `metrics.namespace`) |

## MCP tools

The optional MCP server (`src/mousedroid/mcp/`) exposes the `ToolRegistry`, telemetry / log / config / memory
resources, and prompt templates to any MCP client — see [MCP_OPERATOR_GUIDE.md](MCP_OPERATOR_GUIDE.md).

Architecture: [architecture/c4-llm-gateway.md](architecture/c4-llm-gateway.md),
[architecture/c4-dashboard-proxy.md](architecture/c4-dashboard-proxy.md).
