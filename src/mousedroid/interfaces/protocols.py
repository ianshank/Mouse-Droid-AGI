"""Runtime checkable protocol interfaces for MouseDroid subsystems."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class GoalVector(BaseModel):
    """High-level normalized motion control command."""

    linear_velocity: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Normalized linear velocity in [-1, 1]"
    )
    angular_velocity: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Normalized angular velocity in [-1, 1]"
    )
    arm_action: str = Field(default="idle", description="Discrete manipulator arm action or state")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score of mission goal"
    )
    is_safe: bool = Field(default=True, description="Safety clearance gate flag")

    @classmethod
    def neutral_stop(cls) -> GoalVector:
        """Emergency stop neutral vector."""
        return cls(linear_velocity=0.0, angular_velocity=0.0, arm_action="e_stop", is_safe=True)


@runtime_checkable
class MotorControllerProtocol(Protocol):
    """Interface for low-level motor drive systems."""

    async def set_velocity(self, linear: float, angular: float) -> bool:
        """Set linear and angular drive velocity."""
        ...

    async def emergency_stop(self) -> None:
        """Trigger immediate emergency stop."""
        ...

    def is_healthy(self) -> bool:
        """Return True if controller is communicating and healthy."""
        ...

    async def close(self) -> None:
        """Teardown and close motor controller connection."""
        ...


@runtime_checkable
class CameraProtocol(Protocol):
    """Interface for camera perception subsystems."""

    async def capture_frame(self) -> Any:
        """Capture raw perception frame."""
        ...

    def is_healthy(self) -> bool:
        """Return True if camera driver is operating cleanly."""
        ...

    async def close(self) -> None:
        """Release camera sensor hardware resources."""
        ...


@runtime_checkable
class LiDARProtocol(Protocol):
    """Interface for LiDAR range scanner subsystems."""

    async def get_latest_scan(self) -> list[float]:
        """Return latest 360-degree range scan in meters."""
        ...

    def is_healthy(self) -> bool:
        """Return True if LiDAR hardware is healthy."""
        ...

    async def close(self) -> None:
        """Stop LiDAR motor and close serial interface."""
        ...


@runtime_checkable
class LLMGatewayProtocol(Protocol):
    """Interface for natural language mission translation."""

    async def translate_mission(self, command: str) -> GoalVector:
        """Translate natural language mission into normalized goal vector."""
        ...

    def is_ready(self) -> bool:
        """Return True if gateway backend is active."""
        ...

    def is_degraded(self) -> bool:
        """Return True if operating in failover or degraded mode."""
        ...

    async def stop(self) -> None:
        """Teardown and cancel pending LLM tasks."""
        ...


@runtime_checkable
class MetricsRegistryProtocol(Protocol):
    """Interface for Prometheus observability registries."""

    def record_counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Increment Prometheus counter."""
        ...

    def record_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record value in Prometheus histogram."""
        ...

    def render_prometheus(self) -> str:
        """Render metrics output in Prometheus exposition format."""
        ...


@runtime_checkable
class PromptInjectionFilterProtocol(Protocol):
    """Interface for pre-egress prompt sanitization."""

    def sanitize(self, command: str) -> str:
        """Sanitize command, raising an exception if malicious injection is detected."""
        ...
