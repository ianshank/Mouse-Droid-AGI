"""WebSocket serialization negotiation — JSON vs msgpack handshake.

When a client opens a telemetry WebSocket it can OPTIONALLY send a
``hello`` message identifying its supported protocol versions and
serialization formats. The server picks the best mutual choice and
replies with a ``hello_ack``. Subsequent frames use the negotiated
format.

If the client never sends a hello within
``TelemetryConfig.ws_handshake_timeout_s`` the server falls back to
``TelemetryConfig.serialization`` so existing dashboards (which do not
negotiate) keep working byte-identically.

This module is intentionally framework-agnostic — it parses dicts and
returns decisions, it does NOT touch the aiohttp WebSocket. That makes
it trivial to unit-test without spinning up a server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

SerializationName = Literal["json", "msgpack"]
"""Set of supported serializations. Extend cautiously — every value
must round-trip via aiohttp WebSockets (``send_str`` / ``send_bytes``)."""

SUPPORTED_SERIALIZATIONS: frozenset[str] = frozenset({"json", "msgpack"})
"""Canonical set of server-side supported serializations."""

# Centralised reason codes so log/test consumers can match on a stable
# string rather than free-form prose.
REASON_INVALID_HELLO = "invalid_hello"
REASON_NO_OVERLAP = "no_serialization_overlap"
REASON_UNSUPPORTED_VERSION = "unsupported_protocol_version"


@dataclass(frozen=True)
class NegotiationResult:
    """Outcome of a hello negotiation.

    Attributes:
        ok: ``True`` when negotiation succeeded and the client and server
            agreed on a serialization. ``False`` means the server should
            close the WebSocket (after sending the ack payload as an
            error indicator).
        serialization: The chosen serialization name. Always set, even
            when ``ok=False`` (falls back to the server default for the
            error response itself).
        protocol_version: The chosen protocol version (server-side).
        reason: When ``ok=False``, machine-readable failure reason
            (see ``REASON_*`` constants). ``None`` on success.
        ack_payload: Dictionary the server should send back to the
            client as the hello-ack. Always populated.
    """

    ok: bool
    serialization: SerializationName
    protocol_version: int
    reason: str | None
    ack_payload: dict[str, Any]


def build_default_ack(
    *,
    serialization: SerializationName,
    protocol_version: int,
) -> dict[str, Any]:
    """Construct the no-hello fallback ack payload.

    The server sends this when the client did NOT negotiate within the
    handshake window — it advertises the active server defaults so the
    client can adapt mid-stream if it wants.

    Args:
        serialization: Server-default serialization name.
        protocol_version: Server-side protocol version.

    Returns:
        Dictionary payload conforming to the ``hello_ack`` schema.
    """
    return {
        "hello_ack": {
            "ok": True,
            "negotiated": False,
            "serialization": serialization,
            "protocol_version": protocol_version,
        }
    }


def negotiate(
    client_hello: Any,
    *,
    server_serialization: SerializationName,
    server_protocol_version: int,
    msgpack_client_lib_url: str,
) -> NegotiationResult:
    """Pick a mutually-acceptable serialization + protocol version.

    The hello schema (best-effort, version 1)::

        {
          "hello": {
            "protocol_version": 1,
            "supported_serializations": ["json", "msgpack"],
            "preferred_serialization": "json"   # optional
          }
        }

    Rules:

    1. ``client_hello`` must be a mapping containing a ``"hello"`` mapping.
       Any other shape returns ``REASON_INVALID_HELLO``.
    2. The intersection of the client's ``supported_serializations`` and
       the server's :data:`SUPPORTED_SERIALIZATIONS` must be non-empty.
       Otherwise returns ``REASON_NO_OVERLAP``.
    3. Preference order: client's ``preferred_serialization`` (if
       supported by the server) → server's configured serialization (if
       offered by the client) → first item in the intersection.
    4. Protocol version: accept any client version ``<=`` the server's
       version; otherwise return ``REASON_UNSUPPORTED_VERSION``.

    The hint URL for a msgpack decoder is bundled into every ack so a
    client that picks msgpack but doesn't have a decoder can surface a
    helpful error.

    Args:
        client_hello: Raw dict the client sent. Typed as ``Any`` because
            it has crossed a deserialization boundary; this function
            validates shape internally.
        server_serialization: The server's configured default.
        server_protocol_version: The server's protocol version.
        msgpack_client_lib_url: URL bundled into the ack for clients
            that fall back to msgpack without a decoder.

    Returns:
        A :class:`NegotiationResult` describing the outcome and the
        payload the server should send back to the client.
    """
    if not isinstance(client_hello, dict):
        return _reject(
            REASON_INVALID_HELLO,
            f"expected mapping, got {type(client_hello).__name__}",
            server_serialization=server_serialization,
            server_protocol_version=server_protocol_version,
            msgpack_client_lib_url=msgpack_client_lib_url,
        )

    hello = client_hello.get("hello")
    if not isinstance(hello, dict):
        return _reject(
            REASON_INVALID_HELLO,
            "missing 'hello' sub-object",
            server_serialization=server_serialization,
            server_protocol_version=server_protocol_version,
            msgpack_client_lib_url=msgpack_client_lib_url,
        )

    client_version_raw = hello.get("protocol_version", server_protocol_version)
    try:
        client_version = int(client_version_raw)
    except (TypeError, ValueError):
        return _reject(
            REASON_INVALID_HELLO,
            f"protocol_version not an int: {client_version_raw!r}",
            server_serialization=server_serialization,
            server_protocol_version=server_protocol_version,
            msgpack_client_lib_url=msgpack_client_lib_url,
        )

    if client_version > server_protocol_version or client_version < 1:
        return _reject(
            REASON_UNSUPPORTED_VERSION,
            f"client wants v{client_version}, server speaks v{server_protocol_version}",
            server_serialization=server_serialization,
            server_protocol_version=server_protocol_version,
            msgpack_client_lib_url=msgpack_client_lib_url,
        )

    client_supports_raw = hello.get("supported_serializations", [])
    if not isinstance(client_supports_raw, list):
        return _reject(
            REASON_INVALID_HELLO,
            "supported_serializations must be a list",
            server_serialization=server_serialization,
            server_protocol_version=server_protocol_version,
            msgpack_client_lib_url=msgpack_client_lib_url,
        )

    client_supports = [s for s in client_supports_raw if isinstance(s, str)]
    overlap = [s for s in client_supports if s in SUPPORTED_SERIALIZATIONS]
    if not overlap:
        server_supported = sorted(SUPPORTED_SERIALIZATIONS)
        return _reject(
            REASON_NO_OVERLAP,
            f"client supports {client_supports}, server supports {server_supported}",
            server_serialization=server_serialization,
            server_protocol_version=server_protocol_version,
            msgpack_client_lib_url=msgpack_client_lib_url,
        )

    chosen: SerializationName
    preferred = hello.get("preferred_serialization")
    if isinstance(preferred, str) and preferred in overlap:
        chosen = preferred  # type: ignore[assignment]
    elif server_serialization in overlap:
        chosen = server_serialization
    else:
        chosen = overlap[0]  # type: ignore[assignment]

    _log.info(
        "telemetry_ws_negotiation_succeeded",
        client_version=client_version,
        chosen_serialization=chosen,
        client_supports=client_supports,
    )

    return NegotiationResult(
        ok=True,
        serialization=chosen,
        protocol_version=server_protocol_version,
        reason=None,
        ack_payload={
            "hello_ack": {
                "ok": True,
                "negotiated": True,
                "serialization": chosen,
                "protocol_version": server_protocol_version,
                "msgpack_client_lib_url": msgpack_client_lib_url,
            }
        },
    )


def _reject(
    reason: str,
    detail: str,
    *,
    server_serialization: SerializationName,
    server_protocol_version: int,
    msgpack_client_lib_url: str,
) -> NegotiationResult:
    """Build a rejection result and emit a structured log entry."""
    _log.warning(
        "telemetry_ws_negotiation_failed",
        reason=reason,
        detail=detail,
    )
    return NegotiationResult(
        ok=False,
        serialization=server_serialization,
        protocol_version=server_protocol_version,
        reason=reason,
        ack_payload={
            "hello_ack": {
                "ok": False,
                "negotiated": False,
                "reason": reason,
                "detail": detail,
                "serialization": server_serialization,
                "protocol_version": server_protocol_version,
                "msgpack_client_lib_url": msgpack_client_lib_url,
                "supported_serializations": sorted(SUPPORTED_SERIALIZATIONS),
            }
        },
    )
