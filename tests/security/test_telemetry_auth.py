"""Tests for telemetry server access control."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from mousedroid.config.schema import TelemetryConfig
from mousedroid.telemetry.server import TelemetryServer


@pytest.fixture
def auth_telemetry_config() -> TelemetryConfig:
    cfg = TelemetryConfig(port=8080)
    cfg.api_key = "valid-key-123"
    return cfg


@pytest.fixture
def telemetry_server(auth_telemetry_config: TelemetryConfig) -> TelemetryServer:
    queue = asyncio.Queue()
    health_monitor = AsyncMock()
    health_monitor.check_health.return_value = {"status": "ok"}
    server = TelemetryServer(
        cfg=auth_telemetry_config,
        telemetry_queue=queue,
        health_monitor=health_monitor,
    )
    return server


@pytest.mark.asyncio
async def test_unauthenticated_request_rejected(telemetry_server: TelemetryServer) -> None:
    app = web.Application(middlewares=telemetry_server._build_middlewares())
    telemetry_server._register_routes(app)

    # We can test the middleware directly
    req = make_mocked_request("GET", "/api/v1/sensors")

    # Need to simulate running through middleware
    handler = AsyncMock(return_value=web.Response(text="ok"))

    middleware = telemetry_server._build_middlewares()[-1]  # Assuming last is auth
    with pytest.raises(web.HTTPUnauthorized):
        await middleware(req, handler)


@pytest.mark.asyncio
async def test_invalid_token_rejected(telemetry_server: TelemetryServer) -> None:
    req = make_mocked_request("GET", "/api/v1/sensors", headers={"X-API-Key": "invalid-key"})
    handler = AsyncMock(return_value=web.Response(text="ok"))

    middleware = telemetry_server._build_middlewares()[-1]
    with pytest.raises(web.HTTPUnauthorized):
        await middleware(req, handler)


@pytest.mark.asyncio
async def test_valid_token_accepted(telemetry_server: TelemetryServer) -> None:
    req = make_mocked_request("GET", "/api/v1/sensors", headers={"X-API-Key": "valid-key-123"})
    handler = AsyncMock(return_value=web.Response(text="ok"))

    middleware = telemetry_server._build_middlewares()[-1]
    res = await middleware(req, handler)
    assert res.text == "ok"
