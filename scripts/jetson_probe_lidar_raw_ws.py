#!/usr/bin/env python3
"""P3 — /ws/v1/lidar/raw probe.

Connects to the live raw-LiDAR WebSocket stream, optionally sends a
``hello`` for protocol negotiation, then waits for one decoded
``LidarRawScan`` payload. Asserts:

1. ``hello_ack`` arrives with ``ok=True``, a chosen serialization, and the
   server protocol version.
2. At least one ``LidarRawScan`` payload arrives within the read budget.
3. The payload has ``lidar_n_points > 0`` AND
   ``len(angles_rad) == len(distances_m) == lidar_n_points`` — the
   discriminator that proves the LD19 driver is actually producing
   points (a healthy "no obstacle" scan still has ~450 points; "0
   points" means the sensor is broken or stalled).

Exit code 0 on PASS, non-zero on FAIL. Designed to be wrapped by
``scripts/jetson_new_features_probe.sh``.

Environment:
    MOUSEDROID_TELEMETRY_HOST   default: 127.0.0.1
    MOUSEDROID_TELEMETRY_PORT   default: 8080
    MOUSEDROID_TELEMETRY_TOKEN  bearer token (optional — sent only if set)
    MOUSEDROID_PROBE_TIMEOUT_S  read budget, default 10
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

# aiohttp is already a runtime dependency of the telemetry server, so
# any image that has the server has aiohttp.
try:
    import aiohttp
except ImportError as exc:  # pragma: no cover - environment guard
    print(f"FAIL: aiohttp not importable: {exc}", file=sys.stderr)
    sys.exit(2)


def _build_url() -> str:
    host = os.environ.get("MOUSEDROID_TELEMETRY_HOST", "127.0.0.1")
    port = os.environ.get("MOUSEDROID_TELEMETRY_PORT", "8080")
    path = os.environ.get("MOUSEDROID_LIDAR_RAW_WS_PATH", "/ws/v1/lidar/raw")
    return f"ws://{host}:{port}{path}"


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("MOUSEDROID_TELEMETRY_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _probe() -> int:
    url = _build_url()
    timeout = float(os.environ.get("MOUSEDROID_PROBE_TIMEOUT_S", "10"))
    print(f"INFO: connecting to {url} (timeout={timeout}s)")

    hello: dict[str, Any] = {
        "hello": {
            "protocol_version": 1,
            "supported_serializations": ["json", "msgpack"],
            "preferred_serialization": "json",
        }
    }

    async with aiohttp.ClientSession() as session:
        try:
            # NOTE: do not pass ``timeout=aiohttp.ClientWSTimeout(...)`` —
            # that type only exists in aiohttp >= 3.10 but the project
            # floor is 3.9. The ``asyncio.wait_for(ws.receive(), ...)``
            # below already bounds the read budget, so an explicit
            # connect-side timeout is redundant.
            ws = await session.ws_connect(
                url,
                headers=_auth_headers(),
                heartbeat=None,
            )
        except aiohttp.WSServerHandshakeError as exc:
            print(f"FAIL: WS handshake rejected: status={exc.status}", file=sys.stderr)
            return 3
        except Exception as exc:
            print(f"FAIL: cannot open WS: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 3

        async with ws:
            await ws.send_str(json.dumps(hello))

            ack_received = False
            scan_received = False
            deadline = asyncio.get_event_loop().time() + timeout

            while not (ack_received and scan_received):
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    print(
                        f"FAIL: timeout after {timeout}s (ack={ack_received} scan={scan_received})",
                        file=sys.stderr,
                    )
                    return 4
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
                except asyncio.TimeoutError:
                    print(f"FAIL: read timeout, ack={ack_received}", file=sys.stderr)
                    return 4

                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    print(
                        f"FAIL: server closed WS unexpectedly "
                        f"(code={msg.data} extra={msg.extra!r})",
                        file=sys.stderr,
                    )
                    return 5
                if msg.type != aiohttp.WSMsgType.TEXT:
                    # Binary frames not expected for json serialization
                    print(f"WARN: ignoring non-text frame type={msg.type}")
                    continue

                try:
                    payload = json.loads(msg.data)
                except (TypeError, ValueError) as exc:
                    print(f"FAIL: non-JSON payload: {exc}", file=sys.stderr)
                    return 6

                if not ack_received and isinstance(payload, dict) and "hello_ack" in payload:
                    ack = payload["hello_ack"]
                    if not isinstance(ack, dict):
                        print(f"FAIL: hello_ack not a dict: {ack!r}", file=sys.stderr)
                        return 7
                    if not ack.get("ok"):
                        print(
                            f"FAIL: hello_ack.ok=False reason={ack.get('reason')!r}",
                            file=sys.stderr,
                        )
                        return 7
                    if not ack.get("serialization"):
                        print("FAIL: hello_ack missing serialization", file=sys.stderr)
                        return 7
                    if int(ack.get("protocol_version", 0)) < 1:
                        print(
                            f"FAIL: hello_ack invalid protocol_version={ack.get('protocol_version')!r}",
                            file=sys.stderr,
                        )
                        return 7
                    print(
                        f"PASS-PART: hello_ack ok serialization={ack['serialization']} "
                        f"protocol_version={ack['protocol_version']}"
                    )
                    ack_received = True
                    continue

                # Scan payload — LidarRawScan.to_dict() produces this shape.
                if isinstance(payload, dict) and "n_points" in payload and "angles_rad" in payload:
                    n_points = int(payload.get("n_points", 0))
                    angles = payload.get("angles_rad") or []
                    distances = payload.get("distances_m") or []
                    if n_points <= 0:
                        print(
                            f"FAIL: lidar payload n_points={n_points} (sensor is silent — "
                            "healthy LD19 reports ~450 points per scan)",
                            file=sys.stderr,
                        )
                        return 8
                    if len(angles) != n_points or len(distances) != n_points:
                        print(
                            f"FAIL: array length mismatch "
                            f"angles={len(angles)} distances={len(distances)} n_points={n_points}",
                            file=sys.stderr,
                        )
                        return 9
                    print(
                        f"PASS-PART: lidar scan n_points={n_points} "
                        f"first_angle={angles[0]:.4f} first_dist={distances[0]:.4f}m"
                    )
                    scan_received = True
                    continue

                # Unrelated frame (e.g. heartbeat in some future protocol);
                # tolerate but log so we notice protocol drift.
                print(f"INFO: skipping unrecognised frame keys={sorted(payload.keys())[:6]}")

            print("PASS: lidar_raw_ws — ack + scan received with discriminator OK")
            return 0


def main() -> int:
    try:
        return asyncio.run(_probe())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
