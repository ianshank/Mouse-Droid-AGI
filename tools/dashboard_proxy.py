"""Local reverse proxy for the live Jetson dashboards.

Forwards every HTTP + WebSocket request to a configurable upstream,
optionally injecting a bearer token. Used to make the auth-gated
mousedroid telemetry server (port 8080) + the no-auth Grafana (3000) +
Prometheus (9090) all browsable from a single Claude Preview session.

Handles three transport modes the dashboards use:

1. Plain HTTP GET / POST (e.g. /api/v1/status, /lidar HTML page).
2. Streaming responses (MJPEG /camera/stream, log SSE /api/v1/logs/stream).
3. WebSocket bidirectional forwarding (/ws telemetry frames, /ws/v1/lidar/raw).

Usage:
    # CLI form (preferred for launch.json):
    python dashboard_proxy.py <proxy_port> <upstream_url> [bearer_token]
    # e.g.:
    python dashboard_proxy.py 8081 http://192.168.55.1:8080 dev-token-...
    python dashboard_proxy.py 8082 http://192.168.55.1:3000
    python dashboard_proxy.py 8083 http://192.168.55.1:9090

    # Env-var form (legacy):
    JETSON_HTTP=... JETSON_TOKEN=... PROXY_PORT=... python dashboard_proxy.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import aiohttp
from aiohttp import web


def _resolve_settings() -> tuple[str, int, str, str]:
    """Resolve (proxy_host, proxy_port, upstream_http, token) from CLI args or env.

    Token defaults to empty (no auth header injected). Operators MUST supply
    a real token via the third CLI positional or ``JETSON_TOKEN`` env var
    when proxying an auth-gated upstream — there is intentionally no
    hardcoded fallback so a deploy without the env var fails loudly at the
    upstream's 401 rather than silently re-using a baked-in dev credential.
    """
    args = sys.argv[1:]
    if len(args) >= 2:
        proxy_port = int(args[0])
        upstream_http = args[1].rstrip("/")
        token = args[2] if len(args) >= 3 else ""
    else:
        proxy_port = int(os.environ.get("PROXY_PORT", "8081"))
        upstream_http = os.environ.get("JETSON_HTTP", "http://192.168.55.1:8080").rstrip("/")
        token = os.environ.get("JETSON_TOKEN", "")
    proxy_host = os.environ.get("PROXY_HOST", "127.0.0.1")
    return proxy_host, proxy_port, upstream_http, token


PROXY_HOST, PROXY_PORT, UPSTREAM_HTTP, TOKEN = _resolve_settings()
UPSTREAM_WS = UPSTREAM_HTTP.replace("http://", "ws://").replace("https://", "wss://")

_AUTH_HEADER = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "content-encoding",
}


def _client_headers(req: web.Request) -> dict[str, str]:
    """Headers to forward upstream — drop hop-by-hop + inject Authorization."""
    out = {k: v for k, v in req.headers.items() if k.lower() not in _HOP_BY_HOP}
    # Only inject the Authorization header when a token is configured —
    # dashboards like Grafana / Prometheus have their own auth and would
    # reject (or be confused by) an inappropriate Bearer header.
    if _AUTH_HEADER:
        out.update(_AUTH_HEADER)
    return out


def _upstream_response_headers(upstream: aiohttp.ClientResponse) -> dict[str, str]:
    """Strip hop-by-hop headers from the upstream response before relaying."""
    return {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_BY_HOP}


async def _ws_handler(req: web.Request) -> web.StreamResponse:
    """Forward a WebSocket connection upstream and shuttle messages both ways."""
    ws_server = web.WebSocketResponse()
    await ws_server.prepare(req)

    session: aiohttp.ClientSession = req.app["http_session"]
    upstream_url = UPSTREAM_WS + req.rel_url.path_qs
    try:
        ws_client = await session.ws_connect(
            upstream_url,
            headers=_AUTH_HEADER,
            heartbeat=30,
        )
    except Exception as exc:
        await ws_server.close(code=1011, message=f"upstream ws connect failed: {exc}".encode())
        return ws_server

    async def _pipe_server_to_client() -> None:
        async for msg in ws_server:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await ws_client.send_str(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await ws_client.send_bytes(msg.data)
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                break

    async def _pipe_client_to_server() -> None:
        async for msg in ws_client:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await ws_server.send_str(msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await ws_server.send_bytes(msg.data)
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                break

    # When one direction of the pipe completes (client closed, upstream
    # closed, or either side erroring), we MUST cancel the peer task —
    # otherwise the survivor blocks on ``async for msg in ws_X`` forever,
    # holding a slot in the ``TCPConnector`` pool (limit=64). Under a
    # browser that closes the LiDAR-WS tab repeatedly the pool would
    # exhaust quickly. ``asyncio.wait`` with FIRST_COMPLETED gives us the
    # cancellation hook ``asyncio.gather`` doesn't.
    s2c_task = asyncio.create_task(_pipe_server_to_client())
    c2s_task = asyncio.create_task(_pipe_client_to_server())
    try:
        _done, pending = await asyncio.wait(
            {s2c_task, c2s_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        # Drain cancelled + completed tasks; ``return_exceptions=True`` so
        # the close path always runs even if one pipe raised.
        await asyncio.gather(s2c_task, c2s_task, return_exceptions=True)
    finally:
        await ws_client.close()
        if not ws_server.closed:
            await ws_server.close()
    return ws_server


async def _http_handler(req: web.Request) -> web.StreamResponse:
    """Forward an HTTP request upstream (streaming-safe for MJPEG / SSE)."""
    session: aiohttp.ClientSession = req.app["http_session"]
    upstream_url = UPSTREAM_HTTP + req.rel_url.path_qs
    body = await req.read() if req.body_exists else None

    upstream = await session.request(
        req.method,
        upstream_url,
        headers=_client_headers(req),
        data=body,
        allow_redirects=False,
        timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=None),
    )

    # ``out.prepare`` may raise if the client disconnected between us
    # receiving the upstream response and writing the response head.
    # Wrap the entire downstream-write block in a try/finally so the
    # upstream connection is ALWAYS released — otherwise the upstream
    # body stays open until the session's reaper sweeps, holding pool
    # slots that the WS pool fix above also depends on.
    try:
        out = web.StreamResponse(
            status=upstream.status,
            reason=upstream.reason,
            headers=_upstream_response_headers(upstream),
        )
        await out.prepare(req)
        async for chunk in upstream.content.iter_any():
            if not chunk:
                continue
            try:
                await out.write(chunk)
            except (ConnectionResetError, asyncio.CancelledError):
                break
    finally:
        upstream.release()

    # Suppress write_eof failures — the client may have already disconnected
    # mid-stream (common with MJPEG / SSE consumers that close on tab change).
    import contextlib

    with contextlib.suppress(Exception):
        await out.write_eof()
    return out


async def _dispatch(req: web.Request) -> web.StreamResponse:
    """Top-level router — WebSocket upgrade vs regular HTTP."""
    if req.headers.get("Upgrade", "").lower() == "websocket":
        return await _ws_handler(req)
    return await _http_handler(req)


async def _on_startup(app: web.Application) -> None:
    app["http_session"] = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(force_close=False, limit=64),
    )


async def _on_cleanup(app: web.Application) -> None:
    await app["http_session"].close()


def main() -> int:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_route("*", "/{path:.*}", _dispatch)
    app.router.add_route("*", "/", _dispatch)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    print(f"[proxy] forwarding http://{PROXY_HOST}:{PROXY_PORT} -> {UPSTREAM_HTTP}")
    # Avoid the prior misleading-log bug where ``TOKEN[:24]...`` was printed
    # even when no token was configured — the proxy correctly skipped auth
    # injection, but the log line still implied a bearer was in play. Now
    # the three states are reported faithfully (none / short / truncated).
    if not TOKEN:
        print("[proxy] auth bearer token: (none — auth injection disabled)")
    elif len(TOKEN) <= 24:
        print(f"[proxy] auth bearer token: {TOKEN[:24]}")
    else:
        print(f"[proxy] auth bearer token: {TOKEN[:24]}...")
    web.run_app(app, host=PROXY_HOST, port=PROXY_PORT, print=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
