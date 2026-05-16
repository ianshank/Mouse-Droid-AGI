"""Safety action projector protocol (Tier C2 / C2.1).

A safety action projector clamps a proposed action onto the safety-feasible
set as a soft constraint applied AFTER the policy returns an action and
BEFORE the orchestrator executes it. The projection is a pure function of
the frozen :class:`~mousedroid.safety.context.SafetyContext` plus the
proposed action — there is no internal state across ticks.

This is the soft-constraint complement to the hard E-stop short-circuit
in :meth:`Orchestrator.tick`:

* Hard constraint — ``safety_ctx.is_emergency`` short-circuits the tick
  BEFORE any policy runs. Motors go to zero, the rover stops.
* Soft constraint — the projector clamps any policy's proposed action so
  approaching a human at high speed cannot get past the policy step.

Operators can swap implementations (geometric clamp / future CfC-trace
aware / test stubs) via ``cfg.safety.projector.kind`` once additional
implementations land. The default :class:`GeometricSafetyProjector` is
stateless and CPU-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from mousedroid.safety.context import SafetyContext


@runtime_checkable
class SafetyActionProjectorProtocol(Protocol):
    """Clamps a proposed action onto the safety-feasible set.

    Pure function of the frozen :class:`SafetyContext` + the proposed
    action. Stateless: no internal state survives across ticks.
    """

    def project(
        self,
        action: NDArray[np.float32],
        safety_ctx: SafetyContext,
    ) -> NDArray[np.float32]:
        """Return a possibly-clamped copy of ``action``.

        Implementations MUST be pure (no observable side-effects beyond
        optional structured logging / metric increments) and deterministic
        for identical inputs. They MUST NOT mutate ``action`` in place —
        callers may keep references to the original tensor.

        Args:
            action: Proposed action vector. Shape is policy-defined; the
                projector treats index 0 as the forward-velocity component
                (mouse-droid convention).
            safety_ctx: Frozen safety context for the current tick.

        Returns:
            New action array with the same shape and dtype as ``action``.
        """
        ...


__all__ = ["SafetyActionProjectorProtocol"]
