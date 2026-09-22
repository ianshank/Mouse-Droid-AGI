"""Cloud tool adapter protocol for operator-scoped remote actions."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["CloudToolAdapterProtocol"]


@runtime_checkable
class CloudToolAdapterProtocol(Protocol):
    """Cloud tool adapter protocol for operator-scoped remote actions."""

    @property
    def is_dry_run(self) -> bool:
        """Returns True if the adapter is running in dry-run mode.

        Returns:
            bool: Dry run state.
        """
        ...

    @property
    def is_degraded(self) -> bool:
        """Returns True if the adapter is in a degraded state.

        Returns:
            bool: Degraded state.
        """
        ...

    async def execute_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Executes a cloud tool.

        Args:
            tool_name: The name of the tool to execute.
            params: The parameters to pass to the tool.

        Returns:
            dict[str, Any]: The result of the tool execution.
        """
        ...

    def list_available_tools(self) -> list[str]:
        """Lists available cloud tools.

        Returns:
            list[str]: A list of available tool names.
        """
        ...
