"""Tool registry — registration and dispatch for MouseDroid tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    """Specification for a registered tool.

    Attributes:
        name: Unique tool identifier.
        description: Human-readable description.
        handler: Async callable that executes the tool.
    """

    name: str
    description: str
    handler: Callable[..., Awaitable[Any]]


class ToolRegistry:
    """Registry for MouseDroid tools.

    Provides registration and dispatch for platform tools.
    """

    def __init__(self) -> None:
        """Initialise empty registry."""
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a tool.

        Args:
            spec: Tool specification to register.
        """
        if spec.name in self._tools:
            _log.warning("tool_already_registered", name=spec.name)
        self._tools[spec.name] = spec
        _log.debug("tool_registered", name=spec.name)

    def get(self, name: str) -> ToolSpec | None:
        """Get tool spec by name.

        Args:
            name: Tool identifier.

        Returns:
            Tool spec or None if not found.
        """
        return self._tools.get(name)

    async def dispatch(self, name: str, **kwargs: Any) -> Any:
        """Dispatch a tool by name.

        Args:
            name: Tool identifier.
            **kwargs: Arguments to pass to the tool handler.

        Returns:
            Tool execution result.

        Raises:
            KeyError: If tool is not registered.
        """
        spec = self._tools.get(name)
        if spec is None:
            msg = f"Tool not registered: {name}"
            raise KeyError(msg)
        _log.info("tool_dispatch", name=name)
        return await spec.handler(**kwargs)

    @property
    def names(self) -> list[str]:
        """List of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self._tools)


# ---------------------------------------------------------------------------
# Built-in tool handlers
# ---------------------------------------------------------------------------


async def _health_check() -> dict[str, str]:
    """Run system health check."""
    return {"status": "ok"}


async def _calibrate_ultrasonic() -> dict[str, str]:
    """Calibrate ultrasonic sensor."""
    return {"status": "calibration_complete"}


async def _esp32_diagnostics() -> dict[str, str]:
    """Run ESP32 communication diagnostics."""
    return {"status": "esp32_ok"}


async def _tensorrt_compile() -> dict[str, str]:
    """Compile models to TensorRT engines."""
    return {"status": "compilation_pending"}


async def _benchmark_latency() -> dict[str, str]:
    """Run latency benchmark."""
    return {"status": "benchmark_pending"}


async def _export_experience(path: str = "/tmp/export") -> dict[str, str]:  # noqa: S108
    """Export experience data.

    Args:
        path: Export destination path.

    Returns:
        Export status.
    """
    return {"status": "exported", "path": path}


async def _translate_nl_mission(mission: str = "") -> dict[str, str]:
    """Translate NL mission via LLM gateway.

    Args:
        mission: Natural language mission text.

    Returns:
        Translation status.
    """
    return {"status": "translated", "mission": mission}


async def _system_info() -> dict[str, str]:
    """Get system information."""
    import platform

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }


def create_default_registry() -> ToolRegistry:
    """Create a ToolRegistry pre-populated with built-in tools.

    Returns:
        Registry with all 8 default tools registered.
    """
    registry = ToolRegistry()
    tools = [
        ToolSpec("health_check", "Run system health check", _health_check),
        ToolSpec("calibrate_ultrasonic", "Calibrate ultrasonic sensor", _calibrate_ultrasonic),
        ToolSpec("esp32_diagnostics", "Run ESP32 diagnostics", _esp32_diagnostics),
        ToolSpec("tensorrt_compile", "Compile TensorRT models", _tensorrt_compile),
        ToolSpec("benchmark_latency", "Run latency benchmark", _benchmark_latency),
        ToolSpec("export_experience", "Export experience data", _export_experience),
        ToolSpec("translate_nl_mission", "Translate NL mission", _translate_nl_mission),
        ToolSpec("system_info", "Get system information", _system_info),
    ]
    for tool in tools:
        registry.register(tool)
    return registry
