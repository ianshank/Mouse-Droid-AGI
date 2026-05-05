from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
from mousedroid.config.schema import MCPConfig, Settings
from mousedroid.mcp.tool_bridge import MCPToolBridge, _TokenBucket


def _make_registry(extra_specs: list[ToolSpec] | None = None) -> ToolRegistry:
    reg = ToolRegistry()

    async def _ok() -> dict[str, str]:
        return {"status": "ok"}

    async def _slow() -> dict[str, str]:
        await asyncio.sleep(1.0)
        return {"status": "slow"}

    async def _boom() -> dict[str, str]:
        msg = "boom"
        raise RuntimeError(msg)

    reg.register(ToolSpec("health_check", "ok", _ok))
    reg.register(ToolSpec("calibrate_ultrasonic", "actuation", _ok))
    reg.register(ToolSpec("slow_tool", "slow", _slow))
    reg.register(ToolSpec("boom_tool", "boom", _boom))
    if extra_specs:
        for s in extra_specs:
            reg.register(s)
    return reg


def _bridge(
    *,
    cfg: MCPConfig,
    monitor: Any,
    registry: ToolRegistry | None = None,
    root: Settings | None = None,
) -> MCPToolBridge:
    return MCPToolBridge(
        cfg=cfg,
        root_cfg=root or Settings.model_validate({"mock_hardware": True}),
        tool_registry=registry or _make_registry(),
        safety_monitor=monitor,
    )


class TestVisibility:
    def test_health_check_always_visible(self, mcp_cfg, safe_safety_monitor) -> None:
        b = _bridge(cfg=mcp_cfg, monitor=safe_safety_monitor)
        assert "health_check" in b.visible_tool_names()

    def test_actuation_hidden_by_default(self, mcp_cfg, safe_safety_monitor) -> None:
        b = _bridge(cfg=mcp_cfg, monitor=safe_safety_monitor)
        assert "calibrate_ultrasonic" not in b.visible_tool_names()

    def test_actuation_shown_when_enabled(self, safe_safety_monitor) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "expose_actuation_tools": True})
        b = _bridge(cfg=cfg, monitor=safe_safety_monitor)
        assert "calibrate_ultrasonic" in b.visible_tool_names()

    def test_denylist_wins_over_allowlist(self, safe_safety_monitor) -> None:
        cfg = MCPConfig.model_validate(
            {
                "enabled": True,
                "tools_allowlist": ["health_check", "slow_tool"],
                "tools_denylist": ["slow_tool"],
            }
        )
        b = _bridge(cfg=cfg, monitor=safe_safety_monitor)
        names = b.visible_tool_names()
        assert "health_check" in names
        assert "slow_tool" not in names

    def test_unknown_allowlist_entries_silently_dropped(self, safe_safety_monitor) -> None:
        cfg = MCPConfig.model_validate(
            {
                "enabled": True,
                "tools_allowlist": ["health_check", "does_not_exist"],
            }
        )
        b = _bridge(cfg=cfg, monitor=safe_safety_monitor)
        names = b.visible_tool_names()
        assert names == ["health_check"]

    def test_visible_specs_match_names(self, mcp_cfg, safe_safety_monitor) -> None:
        b = _bridge(cfg=mcp_cfg, monitor=safe_safety_monitor)
        spec_names = [s.name for s in b.visible_tool_specs()]
        assert spec_names == b.visible_tool_names()


class TestDispatch:
    @pytest.mark.asyncio
    async def test_ok_tool_returns_ok(self, mcp_cfg, safe_safety_monitor) -> None:
        b = _bridge(cfg=mcp_cfg, monitor=safe_safety_monitor)
        ctx = b.make_request_context(peer="test")
        r = await b.call_tool("health_check", {}, ctx)
        assert r.status == "ok"
        assert r.payload == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_denied(self, mcp_cfg, safe_safety_monitor) -> None:
        b = _bridge(cfg=mcp_cfg, monitor=safe_safety_monitor)
        ctx = b.make_request_context()
        r = await b.call_tool("unknown_tool_name", None, ctx)
        assert r.status == "denied"

    @pytest.mark.asyncio
    async def test_denylisted_tool_returns_denied(self, safe_safety_monitor) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "tools_denylist": ["slow_tool"]})
        b = _bridge(cfg=cfg, monitor=safe_safety_monitor)
        ctx = b.make_request_context()
        r = await b.call_tool("slow_tool", None, ctx)
        assert r.status == "denied"

    @pytest.mark.asyncio
    async def test_actuation_disabled_status(self, mcp_cfg, safe_safety_monitor) -> None:
        b = _bridge(cfg=mcp_cfg, monitor=safe_safety_monitor)
        ctx = b.make_request_context()
        r = await b.call_tool("calibrate_ultrasonic", None, ctx)
        assert r.status == "actuation_disabled"

    @pytest.mark.asyncio
    async def test_actuation_refused_when_emergency(self, emergency_safety_monitor) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "expose_actuation_tools": True})
        b = _bridge(cfg=cfg, monitor=emergency_safety_monitor)
        ctx = b.make_request_context()
        r = await b.call_tool("calibrate_ultrasonic", None, ctx)
        assert r.status == "refused_emergency"

    @pytest.mark.asyncio
    async def test_timeout_returns_timeout(self, safe_safety_monitor) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "request_timeout_s": 0.05})
        b = _bridge(cfg=cfg, monitor=safe_safety_monitor)
        ctx = b.make_request_context()
        r = await b.call_tool("slow_tool", None, ctx)
        assert r.status == "timeout"

    @pytest.mark.asyncio
    async def test_handler_exception_returns_error(self, mcp_cfg, safe_safety_monitor) -> None:
        b = _bridge(cfg=mcp_cfg, monitor=safe_safety_monitor)
        ctx = b.make_request_context()
        r = await b.call_tool("boom_tool", None, ctx)
        assert r.status == "error"
        assert r.error is not None
        assert "boom" in r.error

    @pytest.mark.asyncio
    async def test_rate_limit_returns_rate_limited(self, safe_safety_monitor) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "rate_limit_rps": 1.0})
        b = _bridge(cfg=cfg, monitor=safe_safety_monitor)
        ctx = b.make_request_context()
        # First call passes (token bucket starts full at rate=1).
        first = await b.call_tool("health_check", None, ctx)
        assert first.status == "ok"
        # Subsequent immediate call exhausts bucket.
        second = await b.call_tool("health_check", None, ctx)
        assert second.status == "rate_limited"


class TestTokenBucket:
    @pytest.mark.asyncio
    async def test_starts_full(self) -> None:
        bucket = _TokenBucket(2.0)
        ok, _ = await bucket.take()
        assert ok is True
        ok, _ = await bucket.take()
        assert ok is True

    @pytest.mark.asyncio
    async def test_refills_over_time(self) -> None:
        bucket = _TokenBucket(50.0)  # ~1 token / 20 ms
        for _ in range(50):
            await bucket.take()
        # Sleep enough to refill several tokens
        await asyncio.sleep(0.05)
        ok, _ = await bucket.take()
        assert ok is True


class TestEdgeCases:
    """Cover the harder-to-reach branches identified by gap analysis."""

    @pytest.mark.asyncio
    async def test_allowlist_excludes_known_tool(self, safe_safety_monitor) -> None:
        cfg = MCPConfig.model_validate({"enabled": True, "tools_allowlist": ["health_check"]})
        b = _bridge(cfg=cfg, monitor=safe_safety_monitor)
        ctx = b.make_request_context()
        # slow_tool is registered but absent from the allowlist.
        r = await b.call_tool("slow_tool", None, ctx)
        assert r.status == "denied"

    @pytest.mark.asyncio
    async def test_safety_monitor_exception_returns_error(self) -> None:
        from unittest.mock import MagicMock

        broken_monitor = MagicMock()
        broken_monitor.evaluate.side_effect = RuntimeError("monitor broken")
        cfg = MCPConfig.model_validate({"enabled": True, "expose_actuation_tools": True})
        b = _bridge(cfg=cfg, monitor=broken_monitor)
        ctx = b.make_request_context()
        r = await b.call_tool("calibrate_ultrasonic", None, ctx)
        assert r.status == "error"
        assert r.error is not None
        assert "monitor broken" in r.error

    @pytest.mark.asyncio
    async def test_observation_provider_used_when_provided(self, safe_safety_monitor) -> None:
        from mousedroid.sensing.bundle import MouseDroidObservationBundle

        called: list[bool] = []

        def _provider() -> MouseDroidObservationBundle:
            called.append(True)
            return MouseDroidObservationBundle()

        cfg = MCPConfig.model_validate({"enabled": True, "expose_actuation_tools": True})
        b = MCPToolBridge(
            cfg=cfg,
            root_cfg=Settings.model_validate({"mock_hardware": True}),
            tool_registry=_make_registry(),
            safety_monitor=safe_safety_monitor,
            observation_provider=_provider,
        )
        ctx = b.make_request_context()
        await b.call_tool("calibrate_ultrasonic", None, ctx)
        assert called == [True]

    @pytest.mark.asyncio
    async def test_circuit_open_returns_circuit_open_status(self, safe_safety_monitor) -> None:
        # Force the breaker open by triggering enough failures to trip
        # the threshold, then assert subsequent calls return "circuit_open".
        from mousedroid.config.schema import CircuitBreakerConfig

        cfg = MCPConfig.model_validate(
            {
                "enabled": True,
                "circuit_breaker": CircuitBreakerConfig(
                    failure_threshold=1, recovery_timeout_s=60.0
                ).model_dump(),
            }
        )
        b = _bridge(cfg=cfg, monitor=safe_safety_monitor)
        ctx = b.make_request_context()
        # First call fails -> breaker opens.
        first = await b.call_tool("boom_tool", None, ctx)
        assert first.status == "error"
        # Second call rejected by open breaker.
        second = await b.call_tool("health_check", None, ctx)
        assert second.status == "circuit_open"

    @pytest.mark.asyncio
    async def test_non_dict_handler_payload_wrapped(self, safe_safety_monitor) -> None:
        from mousedroid.common.tools.registry import ToolRegistry, ToolSpec

        async def _scalar() -> str:
            return "primitive"

        reg = ToolRegistry()
        reg.register(ToolSpec("health_check", "ok", _scalar))
        cfg = MCPConfig.model_validate({"enabled": True})
        b = _bridge(cfg=cfg, monitor=safe_safety_monitor, registry=reg)
        ctx = b.make_request_context()
        r = await b.call_tool("health_check", None, ctx)
        assert r.status == "ok"
        assert r.payload == {"value": "primitive"}

    def test_breaker_name_constant(self) -> None:
        from mousedroid.mcp.tool_bridge import BREAKER_NAME

        assert BREAKER_NAME == "mcp_tool_call"

    def test_make_request_context_records_token_present(self, mcp_cfg, safe_safety_monitor) -> None:
        b = _bridge(cfg=mcp_cfg, monitor=safe_safety_monitor)
        ctx = b.make_request_context(peer="remote", token_present=True)
        assert ctx.peer == "remote"
        assert ctx.token_present is True
        # request_id is a UUID4 hex (32 chars)
        assert len(ctx.request_id) == 32
