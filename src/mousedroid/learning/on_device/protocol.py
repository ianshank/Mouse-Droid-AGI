"""Protocol + result dataclass for the on-device incremental-learning update (Phase 6 WS2).

The on-device learner runs a *bounded* gradient update on fresh rover
experience to refresh policy/world-model weights between cloud retraining
cycles. The contract here is deliberately small and SYNC: it performs the
torch update and returns a **candidate** weight slot — it never mutates the
live/base policy in place, and never blocks an async hot loop. Async
orchestration (trigger threshold, regression gate, slot promotion/revert) is
WS3/WS5's job and wraps this protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True, slots=True)
class OnDeviceUpdateResult:
    """Result of one bounded on-device update cycle.

    The candidate parameters live in ``candidate_state_dict`` — a detached,
    CPU/clone-safe state-dict that the caller (WS3/WS5) scores against the
    cloud baseline before deciding to promote or discard it. It is a SEPARATE
    object from the base/live policy's parameters: applying it is an explicit
    downstream act, never an implicit side effect of running the update.

    Attributes:
        candidate_state_dict: The updated (candidate) parameter set, keyed by
            parameter name. Detached from the autograd graph; safe to persist.
        train_loss: Final training loss (task loss + EWC penalty) after the
            last step, as a Python float (NaN/Inf never stored here).
        n_steps: Number of optimizer steps actually executed (equals
            ``cfg.update_steps`` on a normal run).
    """

    candidate_state_dict: Mapping[str, Tensor]
    train_loss: float
    n_steps: int
    metadata: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class OnDeviceLearner(Protocol):
    """Interface for a bounded, candidate-producing on-device update.

    Implementations MUST:

    * run exactly ``cfg.update_steps`` steps at ``cfg.learning_rate``;
    * apply the EWC Fisher penalty weighted by ``cfg.ewc_lambda`` (reusing the
      shared ``learning/ewc.py`` Fisher/penalty API), anchored on the
      consolidated/base weights;
    * produce a candidate parameter set WITHOUT mutating the base/live policy
      parameters in place (the load-bearing safety invariant);
    * keep the call SYNC and free of any blocking I/O.
    """

    def update(self, batch: Tensor) -> OnDeviceUpdateResult:
        """Run a bounded online update on ``batch`` and return the candidate.

        Args:
            batch: Experience batch tensor shaped ``(n, input_dim)`` matching
                the protected model's first-layer in-features.

        Returns:
            An :class:`OnDeviceUpdateResult` carrying the candidate state-dict,
            final train loss, and the number of steps executed.
        """
        ...


@runtime_checkable
class RSSMSequenceLearner(Protocol):
    """SIBLING of :class:`OnDeviceLearner` for the RSSM-refinement path (WS-E2).

    The original :class:`OnDeviceLearner` is Tensor-typed: ``update(batch: Tensor)``
    assumes a ``forward(tensor)`` policy net. The RSSM has no ``forward()`` — its
    gradient path is ``train_sequence(batch: dict[str, Tensor], decoders)`` over a
    ``(B, T, ...)`` sequence dict. This sibling protocol accepts that dict batch so
    the #134 Tensor path + its green tests are left untouched (additive contract).

    Implementations MUST:

    * deep-copy the base RSSM before any gradient flows (base bitwise-unchanged);
    * run exactly ``cfg.update_steps`` steps at ``cfg.learning_rate``;
    * persist ONLY the refined RSSM ``state_dict`` (never the throwaway decoders);
    * keep the call SYNC and free of blocking I/O (the coordinator offloads it via
      ``asyncio.to_thread``).
    """

    def update(self, batch: Mapping[str, Tensor]) -> OnDeviceUpdateResult:
        """Run a bounded RSSM-refinement cycle on ``batch`` and return the candidate.

        Args:
            batch: A ``(B, T, ...)`` sequence dict with keys ``motor`` / ``action``
                / ``valid_mask`` (always) plus the enabled modalities — exactly the
                shape :meth:`mousedroid.world_model.rssm.RSSM.train_sequence` expects.

        Returns:
            An :class:`OnDeviceUpdateResult` carrying the refined RSSM state-dict,
            final train loss, and the number of steps executed.
        """
        ...


__all__ = ["OnDeviceLearner", "OnDeviceUpdateResult", "RSSMSequenceLearner"]
