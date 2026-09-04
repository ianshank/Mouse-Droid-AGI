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
        *,
        tick_index: int | None = None,
    ) -> SafetyContext:
        """Evaluate safety state from current observation.

        ``tick_index`` is keyword-only with a default so an implementation that
        predates loop-overrun debouncing still satisfies this protocol, and so
        existing two-positional-argument call sites keep working.
        """
        ...
