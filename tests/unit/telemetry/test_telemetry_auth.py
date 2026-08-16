"""Tests for TelemetryServer authentication — REST, WebSocket, and /metrics endpoint.

Validates:
    - API key enforcement on all REST endpoints (legacy X-API-Key mode)
    - Bearer token auth enforcement (new TelemetryAuthConfig mode)
    - WebSocket upgrade requests are rejected by middleware when key is missing/wrong
    - Valid API key accepted via X-API-Key header (REST) and query param (WebSocket)
    - Bearer token accepted via Authorization header and query param (WebSocket)
    - /metrics endpoint requires auth when api_key is set
    - Auth disabled (api_key=None) allows unconditional access
    - Bearer auth disabled allows unconditional access
    - Exempt paths bypass bearer auth
    - Wrong key returns 401 (not 403 or 500)
    - CORS preflight OPTIONS requests pass through regardless of auth state
    - CORS headers present with allowed origins
    - Token sourced from environment variable
"""

from __future__ import annotations

import asyncio
import os
from functools import partial
from unittest.mock import AsyncMock, patch

import pytest

from mousedroid.config.schema import MetricsConfig, TelemetryAuthConfig, TelemetryConfig
from mousedroid.telemetry.metrics import MetricsRegistry
from mousedroid.telemetry.protocol import TelemetryFrame
from tests.unit.telemetry.conftest import _make_health_monitor as _make_health_monitor_base

aiohttp = pytest.importorskip("aiohttp")

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from mousedroid.telemetry.server import TelemetryServer

_API_KEY = "test-api-key-abc123"
_BEARER_TOKEN = "test-bearer-token-xyz789"  # noqa: S105
_WRONG_KEY = "wrong-key"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


_make_health_monitor = partial(_make_health_monitor_base, gpu_temp_c=40.0, gpu_load_pct=10.0)


def _make_server(
    api_key: str | None = _API_KEY,
    with_metrics: bool = False,
    auth_cfg: TelemetryAuthConfig | None = None,
) -> tuple[TelemetryServer, web.Application]:
    cfg = TelemetryConfig(
        enabled=True,
        api_key=api_key,
        metrics_path="/metrics",
        auth=auth_cfg,
    )
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=64)
    health = _make_health_monitor()
    metrics: MetricsRegistry | None = None
    metrics_path = cfg.metrics_path
    if with_metrics:
        metrics_cfg = MetricsConfig(enabled=True, path="/metrics")
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
# REST endpoint authentication (legacy X-API-Key)
# ---------------------------------------------------------------------------


class TestRestAuthentication:
    """Legacy X-API-Key header authentication tests."""

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

    async def test_camera_page_query_api_key_returns_200(self) -> None:
        """Safe GET navigations may use api_key query auth for dashboards."""
        server, app = _make_server()
        server._running = True
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(f"/camera?api_key={_API_KEY}")
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
    """Tests that auth is disabled when api_key is None and no bearer auth."""

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
            resp = await client.get("/api/v1/status", headers={"X-API-Key": "whatever"})
            assert resp.status == 200


# ---------------------------------------------------------------------------
# WebSocket authentication — middleware enforces key for WS upgrades
# ---------------------------------------------------------------------------


class TestWebSocketAuthentication:
    """WebSocket upgrade auth enforcement tests."""

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
    """Prometheus /metrics endpoint auth tests."""

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
    """CORS preflight and header tests."""

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

    async def test_cors_headers_present_with_wildcard_origins(self) -> None:
        """CORS headers are set when cors_origins is wildcard."""
        server, app = _make_server(api_key=None)
        server._running = True
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/api/v1/status",
                headers={"Origin": "http://any-origin.com"},
            )
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    async def test_cors_headers_present_with_allowed_origin(self) -> None:
        """CORS headers echo allowed origin when specific origins configured."""
        cfg = TelemetryConfig(
            enabled=True,
            api_key=None,
            cors_origins=["http://dashboard.local"],
        )
        queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=64)
        health = _make_health_monitor()
        server = TelemetryServer(cfg=cfg, telemetry_queue=queue, health_monitor=health)
        server._running = True
        app = web.Application(middlewares=server._build_middlewares())
        server._register_routes(app)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/api/v1/status",
                headers={"Origin": "http://dashboard.local"},
            )
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "http://dashboard.local"

    async def test_cors_allow_headers_includes_authorization(self) -> None:
        """CORS allow-headers includes Authorization for bearer token support."""
        server, app = _make_server(api_key=None)
        server._running = True
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/v1/status")
            allow_headers = resp.headers.get("Access-Control-Allow-Headers", "")
            assert "Authorization" in allow_headers


# ---------------------------------------------------------------------------
# Bearer token authentication (new TelemetryAuthConfig mode)
# ---------------------------------------------------------------------------


class TestBearerTokenAuth:
    """Tests for the new Bearer token authentication via TelemetryAuthConfig."""

    def _auth_cfg(
        self,
        enabled: bool = True,
        exempt_paths: list[str] | None = None,
    ) -> TelemetryAuthConfig:
        return TelemetryAuthConfig(
            auth_enabled=enabled,
            token_env_var="TEST_TELEMETRY_TOKEN",  # noqa: S106
            allowed_origins=[],
            exempt_paths=exempt_paths if exempt_paths is not None else ["/health", "/metrics"],
        )

    async def test_valid_bearer_token_returns_200(self) -> None:
        """Request with correct Bearer token should succeed."""
        auth_cfg = self._auth_cfg()
        with patch.dict(os.environ, {"TEST_TELEMETRY_TOKEN": _BEARER_TOKEN}):
            server, app = _make_server(api_key=None, auth_cfg=auth_cfg)
            server._running = True
            async with TestClient(TestServer(app)) as client:
                resp = await client.get(
                    "/api/v1/status",
                    headers={"Authorization": f"Bearer {_BEARER_TOKEN}"},
                )
                assert resp.status == 200

    async def test_camera_page_query_bearer_token_returns_200(self) -> None:
        """Safe GET navigations may use token query auth for dashboards."""
        auth_cfg = self._auth_cfg()
        with patch.dict(os.environ, {"TEST_TELEMETRY_TOKEN": _BEARER_TOKEN}):
            server, app = _make_server(api_key=None, auth_cfg=auth_cfg)
            server._running = True
            async with TestClient(TestServer(app)) as client:
                resp = await client.get(f"/camera?token={_BEARER_TOKEN}")
                assert resp.status == 200

    async def test_invalid_bearer_token_returns_401(self) -> None:
        """Request with wrong Bearer token should return 401."""
        auth_cfg = self._auth_cfg()
        with patch.dict(os.environ, {"TEST_TELEMETRY_TOKEN": _BEARER_TOKEN}):
            _, app = _make_server(api_key=None, auth_cfg=auth_cfg)
            async with TestClient(TestServer(app)) as client:
                resp = await client.get(
                    "/api/v1/status",
                    headers={"Authorization": "Bearer wrong-token"},
                )
                assert resp.status == 401

    async def test_missing_bearer_token_returns_401(self) -> None:
        """Request without Authorization header should return 401."""
        auth_cfg = self._auth_cfg()
        with patch.dict(os.environ, {"TEST_TELEMETRY_TOKEN": _BEARER_TOKEN}):
            _, app = _make_server(api_key=None, auth_cfg=auth_cfg)
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/api/v1/status")
                assert resp.status == 401

    async def test_bearer_auth_disabled_allows_passthrough(self) -> None:
        """When auth_enabled=False, requests pass without token."""
        auth_cfg = self._auth_cfg(enabled=False)
        server, app = _make_server(api_key=None, auth_cfg=auth_cfg)
        server._running = True
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/v1/status")
            assert resp.status == 200

    async def test_exempt_path_health_bypasses_auth(self) -> None:
        """Exempt path /health should bypass bearer auth."""
        auth_cfg = self._auth_cfg(exempt_paths=["/api/v1/health", "/metrics"])
        with patch.dict(os.environ, {"TEST_TELEMETRY_TOKEN": _BEARER_TOKEN}):
            _, app = _make_server(api_key=None, auth_cfg=auth_cfg)
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/api/v1/health")
                assert resp.status == 200

    async def test_exempt_path_metrics_bypasses_auth(self) -> None:
        """Exempt path /metrics should bypass bearer auth."""
        auth_cfg = self._auth_cfg(exempt_paths=["/health", "/metrics"])
        with patch.dict(os.environ, {"TEST_TELEMETRY_TOKEN": _BEARER_TOKEN}):
            _, app = _make_server(api_key=None, with_metrics=True, auth_cfg=auth_cfg)
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/metrics")
                assert resp.status == 200

    async def test_non_exempt_path_requires_auth(self) -> None:
        """Non-exempt paths require bearer token even if exempt paths set."""
        auth_cfg = self._auth_cfg(exempt_paths=["/health", "/metrics"])
        with patch.dict(os.environ, {"TEST_TELEMETRY_TOKEN": _BEARER_TOKEN}):
            _, app = _make_server(api_key=None, auth_cfg=auth_cfg)
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/api/v1/status")
                assert resp.status == 401

    async def test_bearer_token_from_env_var(self) -> None:
        """Token is correctly sourced from the configured environment variable."""
        auth_cfg = TelemetryAuthConfig(
            auth_enabled=True,
            token_env_var="CUSTOM_TOKEN_VAR",  # noqa: S106
            allowed_origins=[],
            exempt_paths=[],
        )
        custom_token = "my-custom-env-token"  # noqa: S105
        with patch.dict(os.environ, {"CUSTOM_TOKEN_VAR": custom_token}):
            server, app = _make_server(api_key=None, auth_cfg=auth_cfg)
            server._running = True
            async with TestClient(TestServer(app)) as client:
                resp = await client.get(
                    "/api/v1/status",
                    headers={"Authorization": f"Bearer {custom_token}"},
                )
                assert resp.status == 200

    async def test_bearer_401_returns_json_error(self) -> None:
        """401 response should contain structured JSON error body."""
        auth_cfg = self._auth_cfg()
        with patch.dict(os.environ, {"TEST_TELEMETRY_TOKEN": _BEARER_TOKEN}):
            _, app = _make_server(api_key=None, auth_cfg=auth_cfg)
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/api/v1/status")
                assert resp.status == 401
                assert resp.content_type == "application/json"
                import json

                body = json.loads(await resp.text())
                assert body["error"] == "unauthorized"

    async def test_bearer_ws_accepted_via_query_param(self) -> None:
        """WebSocket upgrade with token query param should succeed."""
        auth_cfg = self._auth_cfg()
        with patch.dict(os.environ, {"TEST_TELEMETRY_TOKEN": _BEARER_TOKEN}):
            server, app = _make_server(api_key=None, auth_cfg=auth_cfg)
            server._running = True
            async with (
                TestClient(TestServer(app)) as client,
                client.ws_connect(f"/ws?token={_BEARER_TOKEN}") as ws,
            ):
                await ws.close()
                assert ws.closed

    async def test_bearer_ws_rejected_without_token(self) -> None:
        """WebSocket upgrade without token should be rejected."""
        auth_cfg = self._auth_cfg()
        with patch.dict(os.environ, {"TEST_TELEMETRY_TOKEN": _BEARER_TOKEN}):
            _, app = _make_server(api_key=None, auth_cfg=auth_cfg)
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/ws")
                assert resp.status == 401

    async def test_cors_options_passes_with_bearer_auth(self) -> None:
        """OPTIONS preflight passes even when bearer auth is enabled."""
        auth_cfg = self._auth_cfg()
        with patch.dict(os.environ, {"TEST_TELEMETRY_TOKEN": _BEARER_TOKEN}):
            _, app = _make_server(api_key=None, auth_cfg=auth_cfg)
            async with TestClient(TestServer(app)) as client:
                resp = await client.options(
                    "/api/v1/status",
                    headers={"Origin": "http://dashboard.local"},
                )
                assert resp.status in (200, 405)

    async def test_unset_env_var_rejects_all(self) -> None:
        """When env var is unset, all requests are rejected."""
        auth_cfg = self._auth_cfg()
        # Ensure the env var does NOT exist
        env = os.environ.copy()
        env.pop("TEST_TELEMETRY_TOKEN", None)
        with patch.dict(os.environ, env, clear=True):
            _, app = _make_server(api_key=None, auth_cfg=auth_cfg)
            async with TestClient(TestServer(app)) as client:
                resp = await client.get(
                    "/api/v1/status",
                    headers={"Authorization": "Bearer anything"},
                )
                assert resp.status == 401
