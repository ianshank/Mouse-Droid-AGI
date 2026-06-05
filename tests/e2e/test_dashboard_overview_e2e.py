"""E2E: the unified dashboard is served by the real TelemetryServer.

Spins up the real :class:`TelemetryServer` in-process (aiohttp ``TestServer``)
and asserts ``GET /`` redirects to ``/dashboard`` (token-preserving) and
``GET /dashboard`` returns the overview page with camera + lidar + sensor-fusion
sections. Auth is off (default ``cfg.auth is None``), mirroring
``tests/integration/test_telemetry_secured.py``'s server-build pattern.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from mousedroid.config.schema import TelemetryConfig
from mousedroid.telemetry.protocol import TelemetryFrame

aiohttp = pytest.importorskip("aiohttp")

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from mousedroid.telemetry.server import TelemetryServer

pytestmark = pytest.mark.asyncio


def _build_app() -> web.Application:
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=8)
    health = AsyncMock()
    health.check_health = AsyncMock(return_value={"status": "ok"})
    server = TelemetryServer(cfg=TelemetryConfig(), telemetry_queue=queue, health_monitor=health)
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    return app


async def test_root_redirects_to_dashboard_preserving_token() -> None:
    async with TestClient(TestServer(_build_app())) as client:
        resp = await client.get("/?token=abc123", allow_redirects=False)
        assert resp.status == 302
        assert resp.headers["Location"] == "/dashboard?token=abc123"


async def test_dashboard_page_served_with_sections() -> None:
    async with TestClient(TestServer(_build_app())) as client:
        resp = await client.get("/dashboard")
        assert resp.status == 200
        assert resp.content_type == "text/html"
        body = await resp.text()

    # Camera + lidar + sensor-fusion + status sections are present.
    assert "/camera/stream" in body
    assert 'id="polar"' in body  # lidar polar canvas
    assert "Sensor fusion" in body
    assert "fused_norm" in body or "fnorm" in body
    # Token-aware + namespace-free (no hardcoded host/port).
    assert "withAuth" in body
    assert "http://127.0.0.1:8080" not in body


async def test_dashboard_subscribes_to_ws_and_camera() -> None:
    async with TestClient(TestServer(_build_app())) as client:
        body = await (await client.get("/dashboard")).text()
    # Single WS feeding every panel + MJPEG embed.
    assert 'withAuth("/ws")' in body
    assert "/api/v1/network" in body  # rover reach info
