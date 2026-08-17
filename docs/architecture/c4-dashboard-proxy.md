# C4 Component — Dashboard Proxy (PR #104)

> The bridge between a Windows / macOS browser and the auth-gated Jetson
> telemetry server. Added in PR #104 to enable live verification of the
> mousedroid dashboard without exposing the bearer token to the browser.

## Component Diagram

```mermaid
C4Component
title Dashboard Proxy — Component Diagram

Container_Boundary(workstation, "Workstation (Python 3.11)") {
    Component(browser, "Browser", "Chrome / Edge")
    Component(settings, "_resolve_settings", "argparse + os.environ", "CLI args first,\nenv vars fallback")
    Component(headers, "_client_headers", "dict[str,str]", "Strip RFC-9110 hop-by-hop\n+ inject Authorization")
    Component(http_h, "_http_handler", "aiohttp", "Plain HTTP +\nMJPEG / SSE streaming")
    Component(ws_h, "_ws_handler", "aiohttp WebSocket", "Bidirectional WS\nbridge")
    Component(dispatch, "_dispatch", "router", "Upgrade: websocket\n→ ws path else http")
    Component(session, "aiohttp.ClientSession", "TCPConnector(limit=64)", "Connection pool to\nthe Jetson upstream")
}

Container_Boundary(jetson, "Jetson Orin Nano") {
    Component(telemetry, "Telemetry Server", "aiohttp", "GET /api/v1/*\nGET /camera/frame.jpg\nGET /camera/stream\nWS /ws, /ws/v1/lidar/raw")
}

Rel(browser, dispatch, "GET / WS\n(127.0.0.1:8081)")
Rel(dispatch, http_h, "HTTP")
Rel(dispatch, ws_h, "Upgrade: websocket")
Rel(http_h, headers, "build request headers")
Rel(ws_h, headers, "build connect headers")
Rel(http_h, session, "session.request(...)")
Rel(ws_h, session, "ws_connect(...)")
Rel(session, telemetry, "HTTP/1.1 + WS + Bearer", "192.168.55.1:8080")
Rel(settings, http_h, "PROXY_PORT, UPSTREAM_HTTP, TOKEN")
Rel(settings, ws_h, "UPSTREAM_WS")
```

## Sequence — HTTP request (camera snapshot)

```mermaid
sequenceDiagram
    autonumber
    participant Br as Browser
    participant Px as Dashboard Proxy
    participant Tx as Telemetry Server (Jetson)
    participant Cam as ResilientCamera(JetsonCSICamera)

    Br->>Px: GET /camera/frame.jpg
    Note over Px: _client_headers()<br/>strips Host, Connection, etc.<br/>injects Authorization: Bearer <token>
    Px->>Tx: GET /camera/frame.jpg<br/>(Bearer ...)
    Tx->>Cam: capture_raw_jpeg()
    Cam-->>Tx: bytes (JPEG)
    Tx-->>Px: 200 OK<br/>Content-Type: image/jpeg
    Note over Px: _upstream_response_headers()<br/>strips hop-by-hop
    Px-->>Br: 200 OK<br/>(JPEG bytes streamed)
```

## Sequence — WebSocket (LiDAR raw scan)

```mermaid
sequenceDiagram
    autonumber
    participant Br as Browser
    participant Px as Dashboard Proxy
    participant Tx as Telemetry Server

    Br->>Px: GET /ws/v1/lidar/raw<br/>(Upgrade: websocket)
    Px->>Tx: WS connect<br/>(Bearer ...)
    Tx-->>Px: 101 Switching Protocols
    Px-->>Br: 101 Switching Protocols

    loop heartbeat 30s
        par Server→Client
            Tx-->>Px: scan frame (binary)
            Px-->>Br: scan frame (binary)
        and Client→Server (rare)
            Br->>Px: control msg
            Px->>Tx: control msg
        end
    end

    Br->>Px: WS close
    Px->>Tx: WS close
```

## Configuration surface

| Source | Wins over | Field |
|--------|-----------|-------|
| CLI arg 1 | env | proxy port |
| CLI arg 2 | env | upstream HTTP URL |
| CLI arg 3 | env | bearer token (optional) |
| `PROXY_PORT` env | (default 8081) | proxy port |
| `JETSON_HTTP` env | (default `http://192.168.55.1:8080`) | upstream |
| `JETSON_TOKEN` env | (documented dev default) | bearer token |
| `PROXY_HOST` env | (default `127.0.0.1`) | bind host |

CLI args ALWAYS win over env vars. The token is also looked up in the
`scripts/launch_dashboard.ps1` wrapper from the operator's `~/.config/mousedroid/`
overlay if present.

## Security boundaries

- The proxy binds to `127.0.0.1` by default — it's intentionally a
  loopback-only dev tool. Setting `PROXY_HOST=0.0.0.0` is a deliberate
  operator decision (see `tools/dashboard_proxy.py` docstring).
- Hop-by-hop headers (RFC-9110 §7.6.1) are stripped both ways. Plus
  `content-length` and `content-encoding` (which aiohttp recomputes).
  Pinned by `tests/regression/test_pr104_aqa.py`.
- The bearer token is injected at the upstream edge — the browser sees
  no Authorization header. This is the whole point of the proxy.

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| 502 from upstream | Proxy passes through any status — see `_http_handler`. Real cause is on the Jetson side. | Check `journalctl -u mousedroid-docker` on Jetson |
| WS handshake fails | Upstream cert / proto mismatch (https vs http) | Confirm `UPSTREAM_WS = UPSTREAM_HTTP.replace(...)` resolved correctly |
| Stream drops mid-MJPEG | Client disconnected; `_http_handler` suppresses `write_eof` errors | Expected; reload the browser tab |
| Grafana → "Unauthorized" through proxy | Configured token but Grafana has its own auth | Run proxy with empty token: `python dashboard_proxy.py 8082 http://192.168.55.1:3000 ""` |

## Related: the on-rover unified dashboard (`/dashboard`)

The proxy is the *workstation* bridge to the auth-gated server. The server itself
now serves a **unified overview page** at `/` (→ `/dashboard`) — a single page
(camera MJPEG + lidar polar + sensor-fusion panel + status) fed by one `/ws`
connection, served by `TelemetryServer._handle_dashboard_page` from
`telemetry/static/dashboard.html`. Any device on the WiFi reaches it directly at
`http://<rover-ip>:8080/?token=…` (or `mousedroid-telemetry.local` via mDNS); the
proxy is only needed for the Claude-Preview workstation path. The fusion panel
renders `TelemetryFrame.fused` (a pure per-modality summary — see
`docs/runbooks/jetson-full-bringup.md` and `CLAUDE.md`).
