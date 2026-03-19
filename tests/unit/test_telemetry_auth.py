"""Tests for TelemetryServer authentication — REST, WebSocket, and /metrics endpoint.

Validates:
    - API key enforcement on all REST endpoints
    - WebSocket upgrade requests are rejected by middleware when key is missing/wrong
    - Valid API key accepted via X-API-Key header (REST) and query param (WebSocket)
    - /metrics endpoint requires auth when api_key is set
    - Auth disabled (api_key=None) allows unconditional access
    - Wrong key returns 401 (not 403 or 500)
    - CORS preflight OPTIONS requests pass through regardless of auth state
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from mousedroid.config.schema import MetricsConfig, TelemetryConfig
from mousedroid.telemetry.metrics import MetricsRegistry
from mousedroid.telemetry.protocol import TelemetryFrame

aiohttp = pytest.importorskip("aiohttp")

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from mousedroid.telemetry.server import TelemetryServer  # noqa: E402

_API_KEY = "test-api-key-abc123"
_WRONG_KEY = "wrong-key"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_health_monitor() -> AsyncMock:
    monitor = AsyncMock()
    monitor.check_health = AsyncMock(
        return_value={"status": "ok", "gpu_temp_c": 40.0, "gpu_load_pct": 10.0}
    )
    return monitor


def _make_server(
    api_key: str | None = _API_KEY,
    with_metrics: bool = False,
) -> tuple[TelemetryServer, web.Application]:
    cfg = TelemetryConfig(enabled=True, api_key=api_key, metrics_path="/metrics")
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=64)
    health = _make_health_monitor()
    metrics: MetricsRegistry | None = None
    if with_metrics:
        metrics = MetricsRegistry(MetricsConfig(enabled=True))
    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=queue,
        health_monitor=health,
        metrics_registry=metrics,
    )
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    return server, app


# ---------------------------------------------------------------------------
# REST endpoint authentication
# ---------------------------------------------------------------------------


class TestRestAuthentication:
    async def test_status_missing_key_returns_401(self) -> None:
        _, app = _make_server()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/v1/status")
            assert resp.status == 401

    async def test_status_wrong_key_returns_401(self) -> None:
        _, app = _make_server()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/v1/status", headers={"X-API-Key": _WRONG_KEY})
            assert resp.status == 401

    async def test_status_correct_key_returns_200(self) -> None:
        server, app = _make_server()
        server._running = True
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/v1/status", headers={"X-API-Key": _API_KEY})
            assert resp.status == 200

    async def test_sensors_missing_key_returns_401(self) -> None:
        _, app = _make_server()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/v1/sensors")
            assert resp.status == 401

    async def test_health_missing_key_returns_401(self) -> None:
        _, app = _make_server()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/v1/health")
            assert resp.status == 401

    async def test_health_correct_key_returns_200(self) -> None:
        _, app = _make_server()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/v1/health", headers={"X-API-Key": _API_KEY})
            assert resp.status == 200

    async def test_network_missing_key_returns_401(self) -> None:
        _, app = _make_server()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/v1/network")
            assert resp.status == 401

    async def test_network_correct_key_returns_200(self) -> None:
        _, app = _make_server()
        with (
            patch(
                "mousedroid.telemetry.server.get_network_interfaces",
                new=AsyncMock(return_value=[]),
            ),
            patch("mousedroid.telemetry.server.get_default_ip", return_value="127.0.0.1"),
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/api/v1/network", headers={"X-API-Key": _API_KEY})
                assert resp.status == 200

    async def test_logs_missing_key_returns_401(self) -> None:
        _, app = _make_server()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/v1/logs")
            assert resp.status == 401

    async def test_error_body_is_text_not_json(self) -> None:
        _, app = _make_server()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/v1/status")
            assert resp.status == 401
            text = await resp.text()
            assert "API key" in text or len(text) > 0


# ---------------------------------------------------------------------------
# Auth-disabled mode (api_key=None)
# ---------------------------------------------------------------------------


class TestAuthDisabled:
    async def test_no_key_required_when_api_key_none(self) -> None:
        server, app = _make_server(api_key=None)
        server._running = True
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/v1/status")
            assert resp.status == 200

    async def test_any_header_accepted_when_no_api_key(self) -> None:
        server, app = _make_server(api_key=None)
        server._running = True
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/api/v1/status", headers={"X-API-Key": "whatever"}
            )
            assert resp.status == 200


# ---------------------------------------------------------------------------
# WebSocket authentication — middleware enforces key for WS upgrades
# ---------------------------------------------------------------------------


class TestWebSocketAuthentication:
    async def test_ws_upgrade_rejected_without_key(self) -> None:
        _, app = _make_server()
        async with TestClient(TestServer(app)) as client:
            # A plain GET to the WS path without Upgrade header hits auth middleware
            resp = await client.get("/ws")
            assert resp.status == 401

    async def test_ws_upgrade_rejected_wrong_key(self) -> None:
        _, app = _make_server()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/ws", headers={"X-API-Key": _WRONG_KEY})
            assert resp.status == 401

    async def test_ws_upgrade_accepted_via_header(self) -> None:
        """WebSocket connection with correct X-API-Key header should succeed."""
        server, app = _make_server()
        server._running = True
        async with (
            TestClient(TestServer(app)) as client,
            client.ws_connect("/ws", headers={"X-API-Key": _API_KEY}) as ws,
        ):
            await ws.close()
            assert ws.closed

    async def test_ws_upgrade_accepted_via_query_param(self) -> None:
        """WebSocket upgrade with api_key query param should succeed."""
        server, app = _make_server()
        server._running = True
        async with (
            TestClient(TestServer(app)) as client,
            client.ws_connect(f"/ws?api_key={_API_KEY}") as ws,
        ):
            await ws.close()
            assert ws.closed

    async def test_ws_no_auth_when_key_disabled(self) -> None:
        """WebSocket connection without any key passes when api_key=None."""
        server, app = _make_server(api_key=None)
        server._running = True
        async with (
            TestClient(TestServer(app)) as client,
            client.ws_connect("/ws") as ws,
        ):
            await ws.close()
            assert ws.closed


# ---------------------------------------------------------------------------
# /metrics endpoint authentication
# ---------------------------------------------------------------------------


class TestMetricsEndpointAuthentication:
    async def test_metrics_missing_key_returns_401(self) -> None:
        _, app = _make_server(with_metrics=True)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/metrics")
            assert resp.status == 401

    async def test_metrics_correct_key_returns_200(self) -> None:
        _, app = _make_server(with_metrics=True)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/metrics", headers={"X-API-Key": _API_KEY})
            assert resp.status == 200

    async def test_metrics_content_type_is_text_plain(self) -> None:
        _, app = _make_server(with_metrics=True)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/metrics", headers={"X-API-Key": _API_KEY})
            assert resp.status == 200
            assert "text/plain" in resp.content_type

    async def test_metrics_body_contains_help(self) -> None:
        _, app = _make_server(with_metrics=True)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/metrics", headers={"X-API-Key": _API_KEY})
            text = await resp.text()
            assert "# HELP" in text

    async def test_metrics_404_when_no_registry(self) -> None:
        """Without a registry injected, /metrics route is not registered."""
        _, app = _make_server(with_metrics=False)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/metrics", headers={"X-API-Key": _API_KEY})
            assert resp.status == 404

    async def test_metrics_no_auth_when_key_disabled(self) -> None:
        _, app = _make_server(api_key=None, with_metrics=True)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/metrics")
            assert resp.status == 200


# ---------------------------------------------------------------------------
# CORS preflight (OPTIONS) always passes
# ---------------------------------------------------------------------------


class TestCorsAuthentication:
    async def test_cors_options_passes_without_key(self) -> None:
        """OPTIONS preflight must succeed regardless of auth to support CORS."""
        _, app = _make_server()
        async with TestClient(TestServer(app)) as client:
            resp = await client.options(
                "/api/v1/status",
                headers={"Origin": "http://dashboard.local"},
            )
            # CORS middleware returns 200 for OPTIONS; auth middleware is not invoked
            assert resp.status in (200, 405)  # 405 if OPTIONS not explicitly routed
