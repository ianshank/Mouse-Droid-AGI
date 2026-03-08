"""Safety monitor protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.protocol import ObservationProtocol


@runtime_checkable
class SafetyMonitorProtocol(Protocol):
    """Interface for platform safety monitors."""

    def evaluate(
        self,
        observation: ObservationProtocol,
        loop_time_ms: float,
    ) -> SafetyContext:
        """Evaluate safety state from current observation."""
        ...
