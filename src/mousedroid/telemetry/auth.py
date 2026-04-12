"""Telemetry authentication middleware — Bearer token auth for aiohttp.

Provides configurable Bearer token authentication sourced from an
environment variable. Supports path exemptions (e.g. /health, /metrics),
CORS preflight passthrough, and structured JSON error responses.

When ``auth_enabled`` is False in config, the middleware is a no-op
passthrough.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from aiohttp import web

    from mousedroid.config.schema import TelemetryAuthConfig

_log = get_logger(__name__)


def build_bearer_auth_middleware(
    auth_cfg: TelemetryAuthConfig,
) -> Any:
    """Build an aiohttp middleware that enforces Bearer token authentication.

    The token is read from the environment variable specified in
    ``auth_cfg.token_env_var``. If the variable is unset or empty,
    all requests are rejected with 401 (unless auth is disabled).

    Args:
        auth_cfg: Authentication configuration.

    Returns:
        An aiohttp middleware function.
    """
    from aiohttp import web

    token = os.environ.get(auth_cfg.token_env_var, "")

    @web.middleware
    async def bearer_auth_middleware(
        request: web.Request,
        handler: Any,
    ) -> web.StreamResponse:
        """Validate Authorization: Bearer <token> header.

        Exempt paths and CORS preflight (OPTIONS) bypass auth.
        WebSocket upgrades accept the token from either the
        Authorization header or the ``?token=...`` query parameter.
        """
        if not auth_cfg.auth_enabled:
            resp: web.StreamResponse = await handler(request)
            return resp

        # CORS preflight always passes
        if request.method == "OPTIONS":
            resp = await handler(request)
            return resp

        # Check exempt paths
        path = request.path
        for exempt in auth_cfg.exempt_paths:
            if path == exempt or path.startswith(exempt + "/"):
                resp = await handler(request)
                return resp

        # Extract token from Authorization header or query param (for WS)
        supplied_token = ""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            supplied_token = auth_header[7:]
        elif request.headers.get("Upgrade", "").lower() == "websocket":
            # WebSocket clients may pass token as query param
            supplied_token = request.query.get("token", "")

        if not token or supplied_token != token:
            _log.warning(
                "telemetry_auth_failed",
                path=path,
                method=request.method,
                peer=request.remote or "unknown",
            )
            raise web.HTTPUnauthorized(
                text='{"error": "unauthorized", "message": "Invalid or missing bearer token"}',
                content_type="application/json",
            )

        resp = await handler(request)
        return resp

    return bearer_auth_middleware


def build_cors_middleware(
    allowed_origins: list[str],
) -> Any:
    """Build an aiohttp CORS middleware with configurable allowed origins.

    Args:
        allowed_origins: List of allowed origins. Empty or ["*"] means
            unrestricted.

    Returns:
        An aiohttp middleware function.
    """
    from aiohttp import web

    @web.middleware
    async def cors_middleware(
        request: web.Request,
        handler: Any,
    ) -> web.StreamResponse:
        """Add CORS headers to responses."""
        if request.method == "OPTIONS":
            resp = web.Response()
        else:
            resp = await handler(request)

        origin_header_value: str | None = None
        if not allowed_origins or allowed_origins == ["*"]:
            origin_header_value = "*"
        else:
            req_origin = request.headers.get("Origin")
            if req_origin and req_origin in allowed_origins:
                origin_header_value = req_origin
                resp.headers.add("Vary", "Origin")

        if origin_header_value is not None:
            resp.headers["Access-Control-Allow-Origin"] = origin_header_value
        resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-API-Key"
        return resp

    return cors_middleware
