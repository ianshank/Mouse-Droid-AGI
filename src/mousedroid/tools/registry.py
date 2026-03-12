"""Tool registry — registration and dispatch for MouseDroid tools.

DEPRECATED: This module is deprecated and will be removed in a future release.
Please use `mousedroid.common.tools.registry` instead.
"""

from __future__ import annotations

import warnings

from mousedroid.common.tools.registry import (
    ToolRegistry,
    ToolSpec,
    _benchmark_latency,
    _calibrate_ultrasonic,
    _esp32_diagnostics,
    _export_experience,
    _health_check,
    _system_info,
    _tensorrt_compile,
    _translate_nl_mission,
    create_default_registry,
)

warnings.warn(
    "mousedroid.tools.registry is deprecated. Use mousedroid.common.tools.registry instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "_benchmark_latency",
    "_calibrate_ultrasonic",
    "_esp32_diagnostics",
    "_export_experience",
    "_health_check",
    "_system_info",
    "_tensorrt_compile",
    "_translate_nl_mission",
    "create_default_registry",
]
