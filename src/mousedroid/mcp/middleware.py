"""Pure-ASGI bearer-auth middleware for the MCP transports.

Wraps :class:`~mousedroid.mcp.auth.BearerTokenValidator` so the same
constant-time comparison and structured logging that govern in-process
calls also apply to every HTTP/SSE request reaching the SDK transports
in :mod:`mousedroid.mcp.transport`.

We intentionally avoid Starlette's :class:`BaseHTTPMiddleware` because
that base class buffers the full response body before forwarding —
incompatible with the unbounded SSE stream the MCP transport produces.
A pure-ASGI middleware (an ``__init__(app)`` + ``__call__(scope, receive,
send)`` callable) leaves streaming responses untouched.

Only the ``Authorization: Bearer <token>`` header is accepted —
query-parameter fallbacks (used by the telemetry server for browser-only
flows) are intentionally NOT supported here because every MCP client
we ship can set headers.
"""

from __future__ import annotations

import json
from typing import Any

from mousedroid.logging.setup import get_logger
from mousedroid.mcp.auth import BearerTokenValidator, MCPAuthError

_log = get_logger(__name__)


class BearerAuthMiddleware:
    """Reject requests that fail :class:`BearerTokenValidator`.

    Implements the ASGI ``__call__(scope, receive, send)`` contract
    directly so streaming responses (SSE in particular) pass through
    without buffering. Non-HTTP scopes (lifespan, websocket) are
    forwarded as-is — auth on those is the wrapped app's responsibility.
    """

    def __init__(
        self,
        app: Any,
        *,
        validator: BearerTokenValidator,
        exempt_paths: tuple[str, ...] = (),
    ) -> None:
        """Wire the middleware.

        Args:
            app: The wrapped ASGI app (Starlette passes this in via
                ``Middleware(BearerAuthMiddleware, ...)``).
            validator: The bearer validator (constructed once with the
                env-var name from :class:`MCPConfig.auth_token_env_var`).
            exempt_paths: Paths exempt from auth (e.g. ``("/health",)``).
                Empty by default so nothing leaks unless an operator
                explicitly opts in via config.
        """
        self._app = app
        self._validator = validator
        self._exempt = frozenset(exempt_paths)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """ASGI entry point — gate HTTP requests, forward everything else."""
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        if path in self._exempt:
            await self._app(scope, receive, send)
            return
        presented = _extract_bearer_from_scope(scope)
        try:
            self._validator.validate(presented)
        except MCPAuthError as exc:
            _log.info(
                "mcp_request_unauthenticated",
                path=path,
                peer=_peer_from_scope(scope),
                reason=str(exc),
            )
            await _send_401(send, str(exc))
            return
        _log.debug(
            "mcp_request_authenticated",
            path=path,
            peer=_peer_from_scope(scope),
        )
        await self._app(scope, receive, send)


def _extract_bearer_from_scope(scope: Any) -> str | None:
    """Pull the bearer token from the ASGI scope's headers."""
    headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
    for name, value in headers:
        if name.lower() == b"authorization":
            try:
                raw = value.decode("latin-1")
            except UnicodeDecodeError:  # pragma: no cover - latin-1 maps every byte
                return None
            parts = raw.split(None, 1)
            if len(parts) != 2 or parts[0].lower() != "bearer":
                return None
            token = parts[1].strip()
            return token or None
    return None


def _peer_from_scope(scope: Any) -> str:
    """Best-effort peer identifier for logs from an ASGI scope."""
    client = scope.get("client")
    if client is None:
        return "unknown"
    host, port = client
    return f"{host}:{port}"


async def _send_401(send: Any, message: str) -> None:
    """Emit a self-contained 401 JSON response over ASGI."""
    body = json.dumps({"error": "unauthorized", "message": message}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"www-authenticate", b'Bearer realm="mousedroid-mcp"'),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


__all__ = ["BearerAuthMiddleware"]
