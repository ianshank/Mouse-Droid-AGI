"""Tests for the per-client serialization paths in ``_negotiate_ws``.

Exercises the in-memory negotiation logic via a lightweight WS stub so
we avoid spinning up aiohttp for every assertion. Integration coverage
of the over-the-wire path lives in
``tests/integration/test_lidar_raw_websocket.py``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from aiohttp import WSMsgType

from mousedroid.config.schema import (
    HealthConfig,
    JetsonConfig,
    TelemetryConfig,
)
from mousedroid.health.monitor import HealthMonitor
from mousedroid.telemetry.publisher import TelemetryPublisher
from mousedroid.telemetry.server import TelemetryServer


@dataclass
class _StubWsMsg:
    type: int
    data: str = ""


class _StubWs:
    """Minimal ``ws.receive``/``ws.send_json``/``close`` stub for negotiation tests."""

    def __init__(self, queued: list[_StubWsMsg]) -> None:
        self._queue: list[_StubWsMsg] = list(queued)
        self.sent: list[Any] = []
        self.closed: bool = False
        self.close_calls: list[tuple[int, bytes]] = []

    async def receive(self) -> _StubWsMsg:
        if not self._queue:
            raise asyncio.TimeoutError
        return self._queue.pop(0)

    async def send_json(self, payload: Any) -> None:
        self.sent.append(payload)

    async def close(self, *, code: int = 1000, message: bytes = b"") -> None:
        """Mirror the aiohttp ``WebSocketResponse.close`` signature.

        PR #79 honours the NegotiationResult contract by closing the
        WebSocket on hard negotiation failures; tests exercise the
        close path via this stub.
        """
        self.closed = True
        self.close_calls.append((code, message))


def _make_server() -> TelemetryServer:
    cfg = TelemetryConfig(
        enabled=True,
        host="127.0.0.1",
        port=1,
        port_discovery_strategy="kernel_assigned",
        mdns_enabled=False,
        ws_handshake_timeout_s=0.05,
    )
    publisher = TelemetryPublisher(cfg)
    return TelemetryServer(
        cfg=cfg,
        telemetry_queue=publisher.get_queue(),
        health_monitor=HealthMonitor(HealthConfig(), JetsonConfig()),
        publisher=publisher,
        lidar_raw_queue=publisher.get_lidar_raw_queue(),
    )


@pytest.mark.asyncio
async def test_negotiate_returns_server_default_on_timeout() -> None:
    """No hello within timeout → server default + default-ack sent."""
    server = _make_server()
    ws = _StubWs(queued=[])
    chosen = await server._negotiate_ws(ws)  # type: ignore[arg-type]
    assert chosen == "json"
    assert ws.sent
    assert ws.sent[0]["hello_ack"]["negotiated"] is False


@pytest.mark.asyncio
async def test_negotiate_returns_server_default_on_non_text_first_msg() -> None:
    """Binary first message → silent fallback, no ack sent."""
    server = _make_server()
    ws = _StubWs(queued=[_StubWsMsg(type=WSMsgType.BINARY, data="")])
    chosen = await server._negotiate_ws(ws)  # type: ignore[arg-type]
    assert chosen == "json"
    assert ws.sent == []


@pytest.mark.asyncio
async def test_negotiate_returns_server_default_on_malformed_json() -> None:
    """Unparseable text → fallback + no ack (warning logged)."""
    server = _make_server()
    ws = _StubWs(queued=[_StubWsMsg(type=WSMsgType.TEXT, data="{not-json")])
    chosen = await server._negotiate_ws(ws)  # type: ignore[arg-type]
    assert chosen == "json"


@pytest.mark.asyncio
async def test_negotiate_picks_client_preferred() -> None:
    """Client preferred takes precedence when server supports it."""
    server = _make_server()
    hello = {
        "hello": {
            "protocol_version": 1,
            "supported_serializations": ["json", "msgpack"],
            "preferred_serialization": "msgpack",
        }
    }
    ws = _StubWs(queued=[_StubWsMsg(type=WSMsgType.TEXT, data=json.dumps(hello))])
    chosen = await server._negotiate_ws(ws)  # type: ignore[arg-type]
    assert chosen == "msgpack"
    assert ws.sent[0]["hello_ack"]["negotiated"] is True


@pytest.mark.asyncio
async def test_negotiate_returns_default_when_client_sends_close() -> None:
    """CLOSE frame before hello → server falls back silently (no ack sent)."""
    server = _make_server()
    ws = _StubWs(queued=[_StubWsMsg(type=WSMsgType.CLOSE, data="")])
    chosen = await server._negotiate_ws(ws)  # type: ignore[arg-type]
    assert chosen == "json"
    assert ws.sent == []  # no ack because client is gone


@pytest.mark.asyncio
async def test_negotiate_rejects_oversized_hello() -> None:
    """A hello above ``WS_HELLO_MAX_BYTES`` is dropped with a recorded failure."""
    from mousedroid.constants import WS_HELLO_MAX_BYTES

    server = _make_server()
    oversize = json.dumps({"hello": {"padding": "x" * (WS_HELLO_MAX_BYTES + 1)}})
    ws = _StubWs(queued=[_StubWsMsg(type=WSMsgType.TEXT, data=oversize)])

    calls: list[tuple[str, str]] = []

    def _spy(subsystem: str, reason: str, **_kw: Any) -> None:
        calls.append((subsystem, reason))

    server._failure_recorder.record = _spy  # type: ignore[method-assign]

    chosen = await server._negotiate_ws(ws)  # type: ignore[arg-type]
    assert chosen == "json"
    assert ws.sent == []  # oversized payload is dropped, no ack sent
    assert calls
    assert calls[0] == ("telemetry", "ws_negotiation_oversized")


@pytest.mark.asyncio
async def test_negotiate_records_failure_when_rejected() -> None:
    """Mismatch → ack with ``ok=False`` + FailureRecorder hit."""
    server = _make_server()
    # Spy on the recorder to confirm it's invoked.
    calls: list[tuple[str, str]] = []

    def _spy(subsystem: str, reason: str, **_kw: Any) -> None:
        calls.append((subsystem, reason))

    server._failure_recorder.record = _spy  # type: ignore[method-assign]

    hello = {
        "hello": {
            "protocol_version": 1,
            "supported_serializations": ["cbor"],
        }
    }
    ws = _StubWs(queued=[_StubWsMsg(type=WSMsgType.TEXT, data=json.dumps(hello))])
    chosen = await server._negotiate_ws(ws)  # type: ignore[arg-type]
    assert chosen == "json"
    assert ws.sent[0]["hello_ack"]["ok"] is False
    assert calls
    assert calls[0][0] == "telemetry"
    assert calls[0][1] == "ws_negotiation_failed"
    # Honour the NegotiationResult contract: hard failure closes the WS.
    from mousedroid.constants import WS_CLOSE_NEGOTIATION_FAILED

    assert ws.closed is True
    assert ws.close_calls
    assert ws.close_calls[0][0] == WS_CLOSE_NEGOTIATION_FAILED
