"""Domain protocols and interface definitions for MouseDroid."""

from __future__ import annotations

from mousedroid.interfaces.cloud_tool_adapter import CloudToolAdapterProtocol
from mousedroid.interfaces.memory_mirror import MemoryMirrorProtocol
from mousedroid.interfaces.mission_decomposer import MissionDecomposerProtocol
from mousedroid.interfaces.protocols import (
    CameraProtocol,
    GoalVector,
    LiDARProtocol,
    LLMGatewayProtocol,
    MetricsRegistryProtocol,
    MotorControllerProtocol,
    PromptInjectionFilterProtocol,
)

__all__ = [
    "CameraProtocol",
    "CloudToolAdapterProtocol",
    "GoalVector",
    "LLMGatewayProtocol",
    "LiDARProtocol",
    "MemoryMirrorProtocol",
    "MetricsRegistryProtocol",
    "MissionDecomposerProtocol",
    "MotorControllerProtocol",
    "PromptInjectionFilterProtocol",
]
