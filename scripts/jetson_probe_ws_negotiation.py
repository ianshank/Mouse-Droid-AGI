#!/usr/bin/env python3
"""P6 — WebSocket negotiation hard-close probe.

Sends a hello with an unsupported serialization (``cbor``) and asserts
the server honours the full ``NegotiationResult`` contract:

1. Server replies with ``hello_ack.ok == False`` AND a non-empty
   ``reason`` field (``no_serialization_overlap``).
2. Server then closes the WebSocket with code ``4400`` —
   ``WS_CLOSE_NEGOTIATION_FAILED`` per ``src/mousedroid/constants.py``.

Both must hold. Closing without the ack regresses
operator-debuggability; sending the ack without closing leaves zombie
clients that keep retrying.

Exit code 0 on PASS, non-zero on FAIL.

Environment:
    MOUSEDROID_TELEMETRY_HOST   default: 127.0.0.1
    MOUSEDROID_TELEMETRY_PORT   default: 8080
    MOUSEDROID_TELEMETRY_TOKEN  bearer token (optional)
    MOUSEDROID_PROBE_TIMEOUT_S  read budget, default 10
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

try:
    import aiohttp
except ImportError as exc:  # pragma: no cover - environment guard
    print(f"FAIL: aiohttp not importable: {exc}", file=sys.stderr)
    sys.exit(2)


WS_CLOSE_NEGOTIATION_FAILED = 4400


def _build_url() -> str:
    host = os.environ.get("MOUSEDROID_TELEMETRY_HOST", "127.0.0.1")
    port = os.environ.get("MOUSEDROID_TELEMETRY_PORT", "8080")
    # Default WebSocket path — overridable for non-standard deploys.
    path = os.environ.get("MOUSEDROID_WS_PATH", "/ws")
    return f"ws://{host}:{port}{path}"


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("MOUSEDROID_TELEMETRY_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _probe() -> int:
    url = _build_url()
    timeout = float(os.environ.get("MOUSEDROID_PROBE_TIMEOUT_S", "10"))
    print(f"INFO: connecting to {url} (timeout={timeout}s)")

    # Force ``no_serialization_overlap`` — server supports {json, msgpack}.
    bad_hello: dict[str, Any] = {
        "hello": {
            "protocol_version": 1,
            "supported_serializations": ["cbor"],
        }
    }

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
            await ws.send_str(json.dumps(bad_hello))

            ack_seen = False
            close_seen_with_4400 = False
            deadline = asyncio.get_event_loop().time() + timeout

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    print(
                        f"FAIL: timeout after {timeout}s "
                        f"(ack_seen={ack_seen} close4400_seen={close_seen_with_4400})",
                        file=sys.stderr,
                    )
                    return 4
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
                except asyncio.TimeoutError:
                    print(
                        f"FAIL: read timeout, ack_seen={ack_seen} "
                        f"close4400_seen={close_seen_with_4400}",
                        file=sys.stderr,
                    )
                    return 4

                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    code = int(msg.data) if msg.data is not None else -1
                    if code != WS_CLOSE_NEGOTIATION_FAILED:
                        print(
                            f"FAIL: server closed with code {code}, expected "
                            f"{WS_CLOSE_NEGOTIATION_FAILED} (WS_CLOSE_NEGOTIATION_FAILED)",
                            file=sys.stderr,
                        )
                        return 5
                    close_seen_with_4400 = True
                    print(f"PASS-PART: close code={code} (negotiation failed)")
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    payload = json.loads(msg.data)
                except (TypeError, ValueError) as exc:
                    print(f"FAIL: non-JSON payload pre-close: {exc}", file=sys.stderr)
                    return 6

                if not isinstance(payload, dict) or "hello_ack" not in payload:
                    # The server contract requires the ack BEFORE the close
                    # frame, so anything else here is a violation.
                    print(
                        f"FAIL: unexpected frame before ack: {sorted(payload.keys())[:6]}",
                        file=sys.stderr,
                    )
                    return 7

                ack = payload["hello_ack"]
                if not isinstance(ack, dict):
                    print(f"FAIL: hello_ack not a dict: {ack!r}", file=sys.stderr)
                    return 7
                if ack.get("ok") is not False:
                    print(
                        f"FAIL: hello_ack.ok must be False on no-overlap, got {ack.get('ok')!r}",
                        file=sys.stderr,
                    )
                    return 7
                reason = ack.get("reason")
                if not (isinstance(reason, str) and reason.strip()):
                    print(
                        f"FAIL: hello_ack.reason must be non-empty string, got {reason!r}",
                        file=sys.stderr,
                    )
                    return 7
                ack_seen = True
                print(f"PASS-PART: hello_ack ok=False reason={reason!r}")

            if not ack_seen:
                print(
                    "FAIL: server closed with 4400 but never sent the ack envelope",
                    file=sys.stderr,
                )
                return 8
            if not close_seen_with_4400:
                print("FAIL: ack received but no 4400 close followed", file=sys.stderr)
                return 9

            print("PASS: ws_negotiation_hard_close — ack(ok=False, reason set) + close 4400")
            return 0


def main() -> int:
    try:
        return asyncio.run(_probe())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
