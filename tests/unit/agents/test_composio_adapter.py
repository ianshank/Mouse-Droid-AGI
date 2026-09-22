"""Unit tests — ComposioToolAdapter.

Tests the Composio cloud tool adapter with mocked SDK,
verifying dry-run mode, tool scoping, and degraded behaviour.
"""

from __future__ import annotations

from typing import Any

import pytest

from mousedroid.config.schema.agents import ComposioConfig


@pytest.fixture
def composio_config() -> ComposioConfig:
    """Composio config with dry_run=True and allowed tools."""
    return ComposioConfig(
        enabled=True,
        dry_run=True,
        allowed_tools=["send_email", "create_ticket"],
    )


@pytest.fixture
def adapter(composio_config: ComposioConfig) -> Any:
    """ComposioToolAdapter instance."""
    from mousedroid.agents.composio_adapter import ComposioToolAdapter

    return ComposioToolAdapter(composio_config)


@pytest.mark.asyncio
async def test_execute_tool_dry_run(adapter: Any) -> None:
    """In dry_run mode, tools are logged but not executed."""
    result = await adapter.execute_tool("send_email", {"to": "test@test.com"})
    assert result.get("dry_run") is True
    assert result.get("tool") == "send_email"


@pytest.mark.asyncio
async def test_execute_tool_rejects_unlisted(adapter: Any) -> None:
    """Tools not in allowed_tools are rejected."""
    result = await adapter.execute_tool("delete_database", {})
    assert "error" in result or "rejected" in str(result).lower()


@pytest.mark.asyncio
async def test_list_available_tools(adapter: Any) -> None:
    """list_available_tools returns the configured allowed_tools."""
    tools = adapter.list_available_tools()
    assert tools == ["send_email", "create_ticket"]


@pytest.mark.asyncio
async def test_degraded_when_sdk_missing(composio_config: ComposioConfig) -> None:
    """start() without composio SDK sets _degraded=True."""
    from unittest.mock import patch
    from mousedroid.agents.composio_adapter import ComposioToolAdapter

    with patch.dict("sys.modules", {"composio": None}):  # type: ignore[dict-item]
        adapter = ComposioToolAdapter(composio_config)
        await adapter.start()
        assert adapter.is_degraded is True


def test_is_dry_run_property(adapter: Any) -> None:
    """is_dry_run reflects cfg.dry_run."""
    assert adapter.is_dry_run is True


@pytest.mark.asyncio
async def test_start_success_and_execute_tool() -> None:
    from unittest.mock import MagicMock, patch

    from mousedroid.agents.composio_adapter import ComposioToolAdapter
    from mousedroid.config.schema.agents import ComposioConfig

    mock_composio = MagicMock()
    mock_client = MagicMock()
    mock_client.execute_action.return_value = {"status": "success"}
    mock_composio.Composio.return_value = mock_client

    with patch.dict("sys.modules", {"composio": mock_composio}):
        cfg = ComposioConfig(enabled=True, dry_run=False, allowed_tools=["send_email"])
        adapter = ComposioToolAdapter(cfg)

        await adapter.start()
        assert adapter.is_degraded is False

        # Test execute tool success path
        res = await adapter.execute_tool("send_email", {"to": "test"})
        assert res["result"]["status"] == "success"

        # Test stop
        await adapter.stop()
