"""Integration tests for the secured telemetry server.

Validates end-to-end auth behaviour with the full TelemetryServer stack:
    - All endpoints require bearer auth except exempted paths
    - WebSocket auth with bearer token
    - Prometheus metrics accessible based on exempt config
    - Server starts and stops cleanly with auth enabled
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

from mousedroid.config.schema import MetricsConfig, TelemetryAuthConfig, TelemetryConfig
from mousedroid.telemetry.metrics import MetricsRegistry
from mousedroid.telemetry.protocol import TelemetryFrame

aiohttp = pytest.importorskip("aiohttp")

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from mousedroid.telemetry.server import TelemetryServer

_BEARER_TOKEN = "integration-test-token-secure"  # noqa: S105
_ENV_VAR = "MOUSEDROID_TELEMETRY_TOKEN"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_health_monitor() -> AsyncMock:
    monitor = AsyncMock()
    monitor.check_health = AsyncMock(
        return_value={"status": "ok", "gpu_temp_c": 42.0, "gpu_load_pct": 15.0}
    )
    return monitor


def _make_secured_server(
    exempt_paths: list[str] | None = None,
    with_metrics: bool = True,
) -> tuple[TelemetryServer, web.Application]:
    """Build a server with bearer auth enabled."""
    auth_cfg = TelemetryAuthConfig(
        auth_enabled=True,
        token_env_var=_ENV_VAR,
        allowed_origins=["http://localhost:3000"],
        exempt_paths=exempt_paths if exempt_paths is not None else ["/api/v1/health", "/metrics"],
    )
    metrics_cfg = MetricsConfig(enabled=True, path="/metrics")
    cfg = TelemetryConfig(
        enabled=True,
        api_key=None,
        metrics_path="/metrics",
        auth=auth_cfg,
        cors_origins=["http://localhost:3000"],
    )
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=64)
    health = _make_health_monitor()
    metrics: MetricsRegistry | None = None
    metrics_path = cfg.metrics_path
    if with_metrics:
        metrics = MetricsRegistry(metrics_cfg)
        metrics_path = metrics_cfg.path

    server = TelemetryServer(
        cfg=cfg,
        telemetry_queue=queue,
        health_monitor=health,
        metrics_registry=metrics,
        metrics_path=metrics_path,
    )
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    return server, app


# ---------------------------------------------------------------------------
# Integration: all endpoints require auth except exempted
# ---------------------------------------------------------------------------


class TestSecuredEndpoints:
    """Verify all endpoints respect bearer auth."""

    async def test_status_requires_auth(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: _BEARER_TOKEN}):
            _, app = _make_secured_server()
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/api/v1/status")
                assert resp.status == 401

    async def test_sensors_requires_auth(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: _BEARER_TOKEN}):
            _, app = _make_secured_server()
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/api/v1/sensors")
                assert resp.status == 401

    async def test_logs_requires_auth(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: _BEARER_TOKEN}):
            _, app = _make_secured_server()
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/api/v1/logs")
                assert resp.status == 401

    async def test_network_requires_auth(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: _BEARER_TOKEN}):
            _, app = _make_secured_server()
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/api/v1/network")
                assert resp.status == 401

    async def test_status_accessible_with_token(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: _BEARER_TOKEN}):
            server, app = _make_secured_server()
            server._running = True
            async with TestClient(TestServer(app)) as client:
                resp = await client.get(
                    "/api/v1/status",
                    headers={"Authorization": f"Bearer {_BEARER_TOKEN}"},
                )
                assert resp.status == 200

    async def test_health_exempt_no_token_needed(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: _BEARER_TOKEN}):
            _, app = _make_secured_server()
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/api/v1/health")
                assert resp.status == 200

    async def test_metrics_exempt_no_token_needed(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: _BEARER_TOKEN}):
            _, app = _make_secured_server()
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/metrics")
                assert resp.status == 200


# ---------------------------------------------------------------------------
# Integration: WebSocket auth
# ---------------------------------------------------------------------------


class TestSecuredWebSocket:
    """WebSocket auth with bearer token."""

    async def test_ws_rejected_without_token(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: _BEARER_TOKEN}):
            _, app = _make_secured_server()
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/ws")
                assert resp.status == 401

    async def test_ws_accepted_with_bearer_header(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: _BEARER_TOKEN}):
            server, app = _make_secured_server()
            server._running = True
            async with (
                TestClient(TestServer(app)) as client,
                client.ws_connect(
                    "/ws",
                    headers={"Authorization": f"Bearer {_BEARER_TOKEN}"},
                ) as ws,
            ):
                await ws.close()
                assert ws.closed

    async def test_ws_accepted_with_query_token(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: _BEARER_TOKEN}):
            server, app = _make_secured_server()
            server._running = True
            async with (
                TestClient(TestServer(app)) as client,
                client.ws_connect(f"/ws?token={_BEARER_TOKEN}") as ws,
            ):
                await ws.close()
                assert ws.closed


# ---------------------------------------------------------------------------
# Integration: CORS with specific origins
# ---------------------------------------------------------------------------


class TestSecuredCors:
    """CORS behaviour with auth and specific allowed origins."""

    async def test_allowed_origin_gets_cors_header(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: _BEARER_TOKEN}):
            _, app = _make_secured_server()
            async with TestClient(TestServer(app)) as client:
                resp = await client.get(
                    "/api/v1/health",
                    headers={"Origin": "http://localhost:3000"},
                )
                assert resp.status == 200
                assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"

    async def test_disallowed_origin_no_cors_header(self) -> None:
        with patch.dict(os.environ, {_ENV_VAR: _BEARER_TOKEN}):
            _, app = _make_secured_server()
            async with TestClient(TestServer(app)) as client:
                resp = await client.get(
                    "/api/v1/health",
                    headers={"Origin": "http://evil.com"},
                )
                assert resp.status == 200
                assert "Access-Control-Allow-Origin" not in resp.headers


# ---------------------------------------------------------------------------
# Integration: Prometheus metrics with auth configuration
# ---------------------------------------------------------------------------


class TestMetricsAccessControl:
    """Verify metrics endpoint behaves based on exempt config."""

    async def test_metrics_not_exempt_requires_auth(self) -> None:
        """When /metrics is NOT in exempt_paths, it requires auth."""
        with patch.dict(os.environ, {_ENV_VAR: _BEARER_TOKEN}):
            _, app = _make_secured_server(exempt_paths=["/api/v1/health"])
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/metrics")
                assert resp.status == 401

    async def test_metrics_not_exempt_accessible_with_token(self) -> None:
        """When /metrics is NOT exempt, token grants access."""
        with patch.dict(os.environ, {_ENV_VAR: _BEARER_TOKEN}):
            _, app = _make_secured_server(exempt_paths=["/api/v1/health"])
            async with TestClient(TestServer(app)) as client:
                resp = await client.get(
                    "/metrics",
                    headers={"Authorization": f"Bearer {_BEARER_TOKEN}"},
                )
                assert resp.status == 200
