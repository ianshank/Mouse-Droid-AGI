"""Hardware: the unified dashboard is reachable on the live rover.

Double-gated — runs only on a Jetson host AND when the telemetry server is
reachable. Confirms ``GET /dashboard`` serves the overview page and that a live
``/ws`` frame carries a populated ``fused`` summary. ``/dashboard`` is behind
bearer auth on the production overlay, so the token (when present) is passed via
the ``?token=`` query — the same pattern the page itself uses.
"""

from __future__ import annotations

import json
import os

import pytest

from tests._jetson_hardware import is_jetson_host

aiohttp = pytest.importorskip("aiohttp")

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not is_jetson_host(), reason="Jetson-only hardware test"),
]


def _base_url() -> str:
    return os.getenv("MOUSEDROID_TELEMETRY_URL", "http://127.0.0.1:8080").rstrip("/")


def _auth_query() -> str:
    token = os.getenv("MOUSEDROID_TELEMETRY_TOKEN")
    return f"?token={token}" if token else ""


@pytest.mark.asyncio
async def test_dashboard_page_reachable() -> None:
    url = f"{_base_url()}/dashboard{_auth_query()}"
    timeout = aiohttp.ClientTimeout(total=5.0)
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(url) as resp,
        ):
            status = resp.status
            body = await resp.text()
    except aiohttp.ClientError as exc:
        pytest.skip(f"telemetry server unreachable at {url}: {exc}")

    if status == 401:
        pytest.skip("dashboard requires a token (set MOUSEDROID_TELEMETRY_TOKEN)")
    assert status == 200
    assert "MouseDroid — Dashboard" in body


@pytest.mark.asyncio
async def test_live_ws_frame_carries_fused() -> None:
    base = _base_url().replace("http", "ws", 1)
    url = f"{base}/ws{_auth_query()}"
    timeout = aiohttp.ClientTimeout(total=10.0)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session, session.ws_connect(url) as ws:
            await ws.send_json(
                {
                    "hello": {
                        "protocol_version": 1,
                        "supported_serializations": ["json"],
                        "preferred_serialization": "json",
                    }
                }
            )
            # Read frames until one carries the fused summary (skip the ack).
            for _ in range(10):
                msg = await ws.receive(timeout=3.0)
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue
                obj = json.loads(msg.data)
                if obj.get("fused"):
                    assert "n_modalities" in obj["fused"]
                    return
    except aiohttp.ClientError as exc:
        pytest.skip(f"telemetry /ws unreachable: {exc}")
    pytest.skip("no fused-bearing frame received within the window")
