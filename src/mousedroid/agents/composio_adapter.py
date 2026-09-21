"""Composio cloud tool adapter — operator-scoped remote actions.

Provides operator-approved cloud tool execution with dry-run mode.
Strictly off-loop. Lazy SDK import.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema.agents import ComposioConfig

_log = get_logger(__name__)


class ComposioToolAdapter:
    """Composio cloud tool adapter."""

    def __init__(self, cfg: ComposioConfig) -> None:
        """Initialize the Composio adapter.

        Args:
            cfg: The Composio configuration.
        """
        self._cfg = cfg
        self._composio: Any = None
        self._client: Any = None
        self._degraded = False

    async def start(self) -> None:
        """Start the adapter and lazily import Composio SDK."""
        try:
            import composio
            self._composio = composio
            self._client = composio.Composio()
            _log.info("composio_adapter_started")
        except ImportError:
            self._degraded = True
            _log.warning("composio_sdk_missing_degraded", degraded=True)
        except Exception as e:
            self._degraded = True
            _log.warning("composio_adapter_start_failed", error=str(e), degraded=True)

    async def stop(self) -> None:
        """Stop the adapter."""
        self._composio = None
        self._client = None
        _log.info("composio_adapter_stopped")

    async def execute_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool if it is in the allowed tools list.

        Args:
            tool_name: The name of the tool to execute.
            params: Parameters for the tool.

        Returns:
            A dictionary containing the result or error.
        """
        if tool_name not in self._cfg.allowed_tools:
            _log.warning("composio_tool_not_allowed", tool=tool_name)
            return {'error': 'tool_not_allowed', 'tool': tool_name}

        if self.is_dry_run:
            _log.info("composio_tool_dry_run", tool=tool_name, params=params)
            return {'dry_run': True, 'tool': tool_name}

        if self._degraded or not self._client:
            return {'error': 'adapter_degraded'}

        try:
            result = await asyncio.to_thread(
                self._client.execute_action,
                action_name=tool_name,
                params=params
            )
            return {'result': result}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._degraded = True
            _log.warning(
                "composio_tool_execution_failed",
                tool=tool_name,
                error=str(e),
                degraded=True,
            )
            return {'error': 'execution_failed', 'details': str(e)}

    def list_available_tools(self) -> list[str]:
        """List all available (allowed) tools.

        Returns:
            A list of allowed tool names.
        """
        return list(self._cfg.allowed_tools)

    @property
    def is_dry_run(self) -> bool:
        """Check if the adapter is in dry-run mode."""
        return self._cfg.dry_run

    @property
    def is_degraded(self) -> bool:
        """Check if the adapter is in a degraded state."""
        return self._degraded
