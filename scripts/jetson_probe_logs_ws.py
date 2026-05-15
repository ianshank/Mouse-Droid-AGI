#!/usr/bin/env python3
"""P13 — /api/v1/logs/stream probe.

Subscribes to the live log-stream WebSocket and asserts at least one
structured log entry arrives within the read budget. Validates that the
payload is a JSON object with the structlog event keys (``event``,
``level``, ``timestamp``) so the dashboard's log panel has data to
render. A healthy rover emits at least one health-monitor entry every
few seconds; if the stream is silent for the full window, the operator
log surface is dead.

Exit code 0 on PASS, non-zero on FAIL.

Environment:
    MOUSEDROID_TELEMETRY_HOST   default: 127.0.0.1
    MOUSEDROID_TELEMETRY_PORT   default: 8080
    MOUSEDROID_TELEMETRY_TOKEN  bearer token (optional)
    MOUSEDROID_PROBE_TIMEOUT_S  read budget, default 15
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

try:
    import aiohttp
except ImportError as exc:  # pragma: no cover - environment guard
    print(f"FAIL: aiohttp not importable: {exc}", file=sys.stderr)
    sys.exit(2)


REQUIRED_KEYS = {"event", "level", "timestamp"}


def _build_url() -> str:
    host = os.environ.get("MOUSEDROID_TELEMETRY_HOST", "127.0.0.1")
    port = os.environ.get("MOUSEDROID_TELEMETRY_PORT", "8080")
    path = os.environ.get("MOUSEDROID_LOGS_WS_PATH", "/api/v1/logs/stream")
    return f"ws://{host}:{port}{path}"


def _auth_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    token = os.environ.get("MOUSEDROID_TELEMETRY_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    api_key = os.environ.get("MOUSEDROID_TELEMETRY_API_KEY", "").strip()
    if api_key:
        headers["X-Telemetry-Api-Key"] = api_key
    return headers


async def _probe() -> int:
    url = _build_url()
    timeout = float(os.environ.get("MOUSEDROID_PROBE_TIMEOUT_S", "15"))
    print(f"INFO: connecting to {url} (timeout={timeout}s)")

    async with aiohttp.ClientSession() as session:
        try:
            ws = await session.ws_connect(
                url,
                headers=_auth_headers(),
                heartbeat=None,
                timeout=aiohttp.ClientWSTimeout(ws_close=5.0),
            )
        except aiohttp.WSServerHandshakeError as exc:
            print(f"FAIL: WS handshake rejected: status={exc.status}", file=sys.stderr)
            return 3
        except Exception as exc:
            print(f"FAIL: cannot open WS: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 3

        async with ws:
            deadline = asyncio.get_event_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    print(
                        f"FAIL: no log entry within {timeout}s — log stream is silent",
                        file=sys.stderr,
                    )
                    return 4
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
                except asyncio.TimeoutError:
                    print(f"FAIL: read timeout after {timeout}s", file=sys.stderr)
                    return 4

                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    print(
                        f"FAIL: server closed log stream "
                        f"(code={msg.data} extra={msg.extra!r}). "
                        "Likely cause: log_buffer disabled in config.",
                        file=sys.stderr,
                    )
                    return 5
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    payload = json.loads(msg.data)
                except (TypeError, ValueError) as exc:
                    print(f"FAIL: non-JSON log payload: {exc}", file=sys.stderr)
                    return 6

                if not isinstance(payload, dict):
                    print(
                        f"FAIL: log payload not a dict: {type(payload).__name__}", file=sys.stderr
                    )
                    return 7

                missing = REQUIRED_KEYS - set(payload.keys())
                if missing:
                    # Operator-debuggable: which keys does the structlog
                    # entry actually carry? Some pipelines rename
                    # 'timestamp' -> 'ts' etc. If we hit this in
                    # practice, we either update REQUIRED_KEYS or fix the
                    # log middleware to emit canonical fields.
                    print(
                        f"FAIL: log entry missing required keys: {sorted(missing)}, "
                        f"got keys: {sorted(payload.keys())[:10]}",
                        file=sys.stderr,
                    )
                    return 8

                event = payload.get("event")
                level = payload.get("level")
                print(
                    f"PASS: logs_ws_stream — event={event!r} level={level!r} "
                    f"keys={sorted(payload.keys())[:8]}"
                )
                return 0


def main() -> int:
    try:
        return asyncio.run(_probe())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
