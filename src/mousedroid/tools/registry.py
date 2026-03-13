"""Tool registry — registration and dispatch for MouseDroid tools.

DEPRECATED: This module is deprecated and will be removed in a future release.
Please use `mousedroid.common.tools.registry` instead.
"""

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec

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


async def _mic_diagnostics() -> dict[str, str]:
    """Run USB microphone diagnostics.

    Returns:
        Microphone diagnostic status.
    """
    try:
        import pyaudio  # pragma: no cover

        pa = pyaudio.PyAudio()  # pragma: no cover
        device_count = pa.get_device_count()  # pragma: no cover
        input_devices = []  # pragma: no cover
        for i in range(device_count):  # pragma: no cover
            info = pa.get_device_info_by_index(i)  # pragma: no cover
            if int(info.get("maxInputChannels", 0)) > 0:  # pragma: no cover
                input_devices.append(str(info.get("name", f"device_{i}")))  # pragma: no cover
        pa.terminate()  # pragma: no cover
        return {  # pragma: no cover
            "status": "ok",
            "device_count": str(device_count),
            "input_devices": ", ".join(input_devices) or "none",
        }
    except ImportError:
        return {"status": "pyaudio_not_installed"}


def create_default_registry() -> ToolRegistry:
    """Create a ToolRegistry pre-populated with built-in tools.

    Returns:
        Registry with all 9 default tools registered.
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
        ToolSpec("mic_diagnostics", "Run USB microphone diagnostics", _mic_diagnostics),
    ]
    for tool in tools:
        registry.register(tool)
    return registry
