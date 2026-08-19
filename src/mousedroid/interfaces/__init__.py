"""Domain protocols and interface definitions for MouseDroid."""

from __future__ import annotations

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
    "GoalVector",
    "LLMGatewayProtocol",
    "LiDARProtocol",
    "MetricsRegistryProtocol",
    "MotorControllerProtocol",
    "PromptInjectionFilterProtocol",
]
