"""Tests for ToolRegistry — full coverage including builtins."""

from __future__ import annotations

import pytest

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec, create_default_registry


async def _dummy_handler() -> str:
    return "ok"


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


def test_constructor(registry: ToolRegistry) -> None:
    assert len(registry) == 0


def test_register_adds_tool(registry: ToolRegistry) -> None:
    spec = ToolSpec(name="test", description="a test tool", handler=_dummy_handler)
    registry.register(spec)
    assert len(registry) == 1
    assert registry.get("test") is not None


def test_register_multiple(registry: ToolRegistry) -> None:
    for i in range(3):
        registry.register(ToolSpec(name=f"tool_{i}", description="", handler=_dummy_handler))
    assert len(registry) == 3


@pytest.mark.asyncio
async def test_dispatch_calls_handler(registry: ToolRegistry) -> None:
    spec = ToolSpec(name="ping", description="ping", handler=_dummy_handler)
    registry.register(spec)
    result = await registry.dispatch("ping")
    assert result == "ok"


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_raises(registry: ToolRegistry) -> None:
    with pytest.raises(KeyError, match="Tool not registered"):
        await registry.dispatch("nonexistent")


def test_create_default_registry_has_8_tools() -> None:
    reg = create_default_registry()
    assert len(reg) == 8


def test_names_property(registry: ToolRegistry) -> None:
    registry.register(ToolSpec(name="a", description="", handler=_dummy_handler))
    registry.register(ToolSpec(name="b", description="", handler=_dummy_handler))
    assert set(registry.names) == {"a", "b"}


def test_get_returns_none_for_missing(registry: ToolRegistry) -> None:
    assert registry.get("missing") is None


def test_register_duplicate_warns(registry: ToolRegistry) -> None:
    spec = ToolSpec(name="dup", description="", handler=_dummy_handler)
    registry.register(spec)
    registry.register(spec)  # Should warn but not raise
    assert len(registry) == 1


# -- Built-in tool handler dispatch tests --


@pytest.mark.asyncio
async def test_dispatch_health_check() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("health_check")
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_dispatch_calibrate_ultrasonic() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("calibrate_ultrasonic")
    assert result["status"] == "calibration_complete"


@pytest.mark.asyncio
async def test_dispatch_esp32_diagnostics() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("esp32_diagnostics")
    assert result["status"] == "esp32_ok"


@pytest.mark.asyncio
async def test_dispatch_tensorrt_compile() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("tensorrt_compile")
    assert result["status"] == "compilation_pending"


@pytest.mark.asyncio
async def test_dispatch_benchmark_latency() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("benchmark_latency")
    assert result["status"] == "benchmark_pending"


@pytest.mark.asyncio
async def test_dispatch_export_experience() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("export_experience", path="/tmp/test_export")
    assert result["status"] == "exported"
    assert result["path"] == "/tmp/test_export"


@pytest.mark.asyncio
async def test_dispatch_translate_nl_mission() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("translate_nl_mission", mission="go left")
    assert result["status"] == "translated"
    assert result["mission"] == "go left"


@pytest.mark.asyncio
async def test_dispatch_system_info() -> None:
    reg = create_default_registry()
    result = await reg.dispatch("system_info")
    assert "platform" in result
    assert "python" in result
    assert "machine" in result
