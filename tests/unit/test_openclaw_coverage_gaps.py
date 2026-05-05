"""Coverage gap-fill tests for the OpenClaw integration.

Targets the small set of branches that the headline tests don't reach
because they're either narrow boundary cases, programming-error guards,
or factory wiring decisions:

* :class:`DeferredOrchestratorRef.process_mission` raising before bind.
* :func:`build_mission_dispatcher` returning ``(None, None)`` vs the wired pair.
* :class:`MarkdownReplayExporter.path` accessor and ``entry_truncate_chars`` knob.
* :class:`BearerAuthMiddleware` short-circuit on non-HTTP scopes (lifespan / websocket).
* :func:`_peer_from_scope` fallback when ``scope["client"]`` is ``None``.
* :class:`RegexInjectionFilter.max_len` accessor.
* ``POST /api/v1/mission`` defensive ``openclaw_disabled``, ``timeout``,
  and generic-exception branches.

Each test exercises one branch end-to-end with an assertion on the
log event or the response payload — coverage and observability both.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from mousedroid.config.schema import OpenClawConfig, TelemetryConfig
from mousedroid.llm_gateway.protocol import GoalVector
from mousedroid.mcp.middleware import (
    BearerAuthMiddleware,
    _extract_bearer_from_scope,
    _peer_from_scope,
)
from mousedroid.memory.exporter import MarkdownReplayExporter
from mousedroid.orchestrator.mission_dispatcher import (
    DeferredOrchestratorRef,
    OrchestratorMissionDispatcher,
    build_mission_dispatcher,
)
from mousedroid.security.injection_filter import RegexInjectionFilter
from mousedroid.telemetry.protocol import TelemetryFrame

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from mousedroid.telemetry.server import TelemetryServer

# ---------------------------------------------------------------------------
# DeferredOrchestratorRef raises before bind
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deferred_ref_raises_before_bind() -> None:
    ref = DeferredOrchestratorRef()
    with pytest.raises(RuntimeError, match="bind"):
        await ref.process_mission("anything")


# ---------------------------------------------------------------------------
# build_mission_dispatcher factory branches
# ---------------------------------------------------------------------------


def _filter() -> RegexInjectionFilter:
    return RegexInjectionFilter([], max_len=64)


def test_build_mission_dispatcher_returns_none_pair_when_cfg_none() -> None:
    dispatcher, deferred = build_mission_dispatcher(None, injection_filter=_filter())
    assert dispatcher is None
    assert deferred is None


def test_build_mission_dispatcher_returns_none_pair_when_disabled() -> None:
    cfg = OpenClawConfig(enabled=False)
    dispatcher, deferred = build_mission_dispatcher(cfg, injection_filter=_filter())
    assert dispatcher is None
    assert deferred is None


def test_build_mission_dispatcher_returns_wired_pair_when_enabled() -> None:
    cfg = OpenClawConfig(enabled=True)
    dispatcher, deferred = build_mission_dispatcher(cfg, injection_filter=_filter())
    assert isinstance(dispatcher, OrchestratorMissionDispatcher)
    assert isinstance(deferred, DeferredOrchestratorRef)
    # Late binding: dispatcher uses ``deferred`` for ``process_mission``.
    deferred.bind(_StubOrch())
    # Surface the wiring works end-to-end without the orchestrator itself.

    async def _drive() -> GoalVector:
        result = await dispatcher.dispatch("hold", channel="rest", peer="unit")
        return result.goal_vector

    goal = asyncio.run(_drive())
    assert goal == GoalVector(0.0, 0.0, 0.0)


class _StubOrch:
    async def process_mission(self, _nl: str) -> GoalVector:
        return GoalVector(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# MarkdownReplayExporter knobs
# ---------------------------------------------------------------------------


def test_exporter_path_property(tmp_path: Path) -> None:
    out = tmp_path / "MEMORY.md"
    assert MarkdownReplayExporter(out).path == out


def test_exporter_rejects_non_positive_truncate(tmp_path: Path) -> None:
    out = tmp_path / "MEMORY.md"
    with pytest.raises(ValueError, match="entry_truncate_chars"):
        MarkdownReplayExporter(out, entry_truncate_chars=0)


@pytest.mark.asyncio
async def test_exporter_truncates_long_payloads(tmp_path: Path) -> None:
    """``entry_truncate_chars`` actually caps the rendered repr."""
    from mousedroid.config.schema import MemoryConfig
    from mousedroid.memory.episodic import EpisodicReplay

    out = tmp_path / "MEMORY.md"
    exporter = MarkdownReplayExporter(out, entry_truncate_chars=32)
    replay = EpisodicReplay(MemoryConfig(episodic_capacity=2), seed=0)
    # Push a payload whose repr is much longer than 32.
    replay.push({"k": "x" * 200})
    await exporter.export(replay)
    body = out.read_text(encoding="utf-8")
    # The single bullet line carries the truncated repr ending in '…'.
    bullet_lines = [line for line in body.splitlines() if line.startswith("- `")]
    assert len(bullet_lines) == 1
    assert "…" in bullet_lines[0]


# ---------------------------------------------------------------------------
# BearerAuthMiddleware non-HTTP scope passthrough + peer fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_forwards_non_http_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lifespan / websocket scopes bypass auth and reach the wrapped app."""
    from mousedroid.mcp.auth import BearerTokenValidator

    monkeypatch.setenv("_OPENCLAW_TEST_TOKEN", "secret")
    validator = BearerTokenValidator("_OPENCLAW_TEST_TOKEN", required=True)

    forwarded: list[str] = []

    async def downstream(scope: Any, _receive: Any, _send: Any) -> None:
        forwarded.append(scope["type"])

    mw = BearerAuthMiddleware(downstream, validator=validator)
    await mw({"type": "lifespan"}, AsyncMock(), AsyncMock())
    await mw({"type": "websocket", "path": "/"}, AsyncMock(), AsyncMock())
    assert forwarded == ["lifespan", "websocket"]


def test_peer_from_scope_handles_missing_client() -> None:
    """``client=None`` (in-process ASGI runners) yields ``"unknown"``."""
    assert _peer_from_scope({"client": None}) == "unknown"
    assert _peer_from_scope({"client": ("10.0.0.1", 1234)}) == "10.0.0.1:1234"
    assert _peer_from_scope({}) == "unknown"


def test_extract_bearer_from_scope_variants() -> None:
    """No header, malformed scheme, latin-1-decodable bytes all handled."""
    # No Authorization header at all
    assert _extract_bearer_from_scope({"headers": []}) is None
    # Wrong scheme
    assert _extract_bearer_from_scope({"headers": [(b"authorization", b"Basic abc")]}) is None
    # Single token (no scheme prefix)
    assert _extract_bearer_from_scope({"headers": [(b"authorization", b"naked-token")]}) is None
    # Valid bearer token
    assert _extract_bearer_from_scope({"headers": [(b"authorization", b"Bearer tok-1")]}) == "tok-1"
    # Empty token after stripping
    assert _extract_bearer_from_scope({"headers": [(b"authorization", b"Bearer    ")]}) is None


# ---------------------------------------------------------------------------
# RegexInjectionFilter accessors
# ---------------------------------------------------------------------------


def test_injection_filter_max_len_accessor() -> None:
    f = RegexInjectionFilter([], max_len=128)
    assert f.max_len == 128
    assert f.has_regex is False


# ---------------------------------------------------------------------------
# POST /api/v1/mission defensive branches
# ---------------------------------------------------------------------------


def _build_app(server: TelemetryServer) -> web.Application:
    app = web.Application(middlewares=server._build_middlewares())
    server._register_routes(app)
    return app


def _make_server_with_dispatcher(
    dispatcher: Any,
) -> TelemetryServer:
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=4)
    health = AsyncMock()
    health.check_health = AsyncMock(return_value={"status": "ok"})
    server = TelemetryServer(
        cfg=TelemetryConfig(enabled=True),
        telemetry_queue=queue,
        health_monitor=health,
        mission_dispatcher=dispatcher,
        openclaw_cfg=OpenClawConfig(enabled=True),
    )
    server._running = True
    return server


@pytest.mark.asyncio
async def test_mission_endpoint_disabled_branch_logs_and_503s() -> None:
    """If the route is force-registered but the dispatcher is None, return 503.

    Constructed by manually flipping ``_mission_route_enabled`` on a
    server built without a dispatcher; production wiring never reaches
    this state but the defensive guard is a real branch and must
    behave as documented.
    """
    queue: asyncio.Queue[TelemetryFrame] = asyncio.Queue(maxsize=4)
    health = AsyncMock()
    health.check_health = AsyncMock(return_value={"status": "ok"})
    server = TelemetryServer(
        cfg=TelemetryConfig(enabled=True),
        telemetry_queue=queue,
        health_monitor=health,
    )
    server._running = True
    # Force the gate open without a dispatcher to hit the belt-and-braces branch.
    server._mission_route_enabled = True
    app = web.Application(middlewares=server._build_middlewares())
    app.router.add_post("/api/v1/mission", server._handle_mission_post)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/v1/mission", json={"nl_command": "go"})
        assert resp.status == 503
        assert (await resp.json())["error"] == "openclaw_disabled"


@pytest.mark.asyncio
async def test_mission_endpoint_dispatcher_timeout_returns_504() -> None:
    class _SlowDispatcher:
        @property
        def mission_just_completed(self) -> bool:
            return False

        def clear_mission_completed(self) -> None:
            return None

        async def dispatch(self, *_args: Any, **_kw: Any) -> None:
            raise TimeoutError("downstream timeout")

    server = _make_server_with_dispatcher(_SlowDispatcher())
    app = _build_app(server)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/v1/mission", json={"nl_command": "patrol"})
        assert resp.status == 504
        assert (await resp.json())["error"] == "timeout"


@pytest.mark.asyncio
async def test_mission_endpoint_dispatcher_generic_error_returns_500() -> None:
    class _BrokenDispatcher:
        @property
        def mission_just_completed(self) -> bool:
            return False

        def clear_mission_completed(self) -> None:
            return None

        async def dispatch(self, *_args: Any, **_kw: Any) -> None:
            raise RuntimeError("boom")

    server = _make_server_with_dispatcher(_BrokenDispatcher())
    app = _build_app(server)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/api/v1/mission", json={"nl_command": "patrol"})
        assert resp.status == 500
        assert (await resp.json())["error"] == "internal_error"
