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

        ``tick_index`` is part of the interface, not an optional extra. Every
        implementation must accept it: the orchestrator passes it on both of
        its ``evaluate`` call sites, and ``mypy --strict`` rejects a narrower
        signature. Because this protocol is ``@runtime_checkable``, an
        implementation that omits the parameter still passes ``isinstance`` --
        that check inspects attribute presence, not signatures -- and then
        raises ``TypeError`` on the first real call. Do not read the default
        as a compatibility guarantee for implementations.

        What the keyword-only default *does* buy is caller compatibility: the
        two-positional-argument form ``evaluate(observation, loop_time_ms)``
        keeps working unchanged, and ``None`` selects the pre-debounce
        semantics of counting every call.

        Args:
            observation: Current sensor observation bundle.
            loop_time_ms: Duration of the tick being judged, in milliseconds.
            tick_index: Monotonic tick counter used to dedupe repeated
                evaluations within a single tick, or ``None`` when the caller
                does not track ticks.

        Returns:
            The evaluated safety context for this tick.
        """
        ...
