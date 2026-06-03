"""Unit tests for the dashboard routes on TelemetryServer.

Exercises ``_handle_root`` (``/`` → 302 ``/dashboard``, token-preserving) and
``_handle_dashboard_page`` (serves ``static/dashboard.html``). Lives under
``tests/unit/telemetry/`` so the branch-coverage gate (which globs this dir)
sees the server handler bodies.
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


def _app() -> web.Application:
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=8)
    health = AsyncMock()
    server = TelemetryServer(cfg=TelemetryConfig(), telemetry_queue=queue, health_monitor=health)
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    return app


async def test_root_redirects_to_dashboard_no_token() -> None:
    async with TestClient(TestServer(_app())) as client:
        resp = await client.get("/", allow_redirects=False)
        assert resp.status == 302
        assert resp.headers["Location"] == "/dashboard"


async def test_root_redirect_preserves_query_token() -> None:
    async with TestClient(TestServer(_app())) as client:
        resp = await client.get("/?token=abc", allow_redirects=False)
        assert resp.status == 302
        assert resp.headers["Location"] == "/dashboard?token=abc"


async def test_dashboard_page_served() -> None:
    async with TestClient(TestServer(_app())) as client:
        resp = await client.get("/dashboard")
        assert resp.status == 200
        assert resp.content_type == "text/html"
        body = await resp.text()
    assert "MouseDroid — Dashboard" in body
    assert 'id="polar"' in body
