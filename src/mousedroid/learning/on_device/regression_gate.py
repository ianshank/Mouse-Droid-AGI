"""RSSM-vs-RSSM recon-loss safety-regression gate + auto-revert (Phase 6 WS-E3).

This is the highest-risk seam of the on-device-learning loop: it decides whether
a freshly-refined candidate **RSSM** is safe to promote into the live world
model, or must be reverted. The SPIKE-LOCKED WS-E3 contract:

* Score the candidate RSSM AND the live baseline RSSM by their held-out
  **reconstruction+KL loss** (``train_sequence(batch, decoders)["loss"]``) on the
  SAME FIXED held-out ``(B, T, ...)`` batch, with the SAME shared
  :class:`~mousedroid.world_model.rssm.RawModalityDecoders` and the SAME
  ``scoring_seed``, via the deterministic :func:`~.scoring.score_dynamics`.
  **LOWER IS BETTER.**
* PROMOTE iff ``candidate_loss <= baseline_loss + regression_tolerance`` (and
  ``candidate_loss`` is finite). This INVERTS the pre-ENABLEMENT higher-is-better
  imagined-return gate; ``GateDecision.delta`` is ``candidate_loss -
  baseline_loss`` with the sign convention **positive delta = WORSE**.
* On PROMOTE, mark the candidate slot ACTIVE in the slot store (the live-model
  hot-swap itself is the separate, ``enable_hot_swap``-gated WS-E4 seam).
* Otherwise REVERT: leave the live model untouched (do NOT mark the slot active)
  and increment ``inc_on_device_learning_reverted("regression_bound")``. A
  non-finite candidate loss (a heavily-degraded candidate blows the KL up) is
  correctly ``> baseline`` ⇒ REVERT — the gate never assumes a finite loss.

Why recon-loss and not imagined return: the pre-ENABLEMENT imagined-return metric
summed the model's OWN ``reward_head`` along a prior rollout, so a candidate that
inflates its (recon-graph-unused) reward head scored HIGHER while its real
dynamics were unchanged or worse — it SELF-GAMED the gate (proven in the
WS-E-SPIKE). That metric is retired entirely; the recon-loss gate replaced it.

Determinism is load-bearing: both losses come from the same deterministic scorer
on the same fixed inputs, so the decision is reproducible. The scoring call is
torch-heavy; the coordinator runs the gate OFF the event loop on the slow cadence
(``asyncio.to_thread``), never on the 30 Hz hot loop.

The ``score_fn`` is injectable (DI) so the decision logic is unit-testable with
pinned losses. The default ``score_fn`` closes over the shared held-out batch +
shared decoders + ``scoring_seed`` and calls :func:`~.scoring.score_dynamics`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mousedroid.learning.on_device.scoring import score_dynamics
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from mousedroid.config.schema import OnDeviceLearningConfig
    from mousedroid.learning.on_device.slot_store import CandidateSlot, OnDeviceSlotStore
    from mousedroid.world_model.protocol import WorldModelProtocol
    from mousedroid.world_model.rssm import RawModalityDecoders

_log = get_logger(__name__)

#: Revert reason recorded on the Prometheus counter when the candidate fails the
#: regression bound. Must stay in ``telemetry.metrics._ON_DEVICE_REVERT_REASONS``.
_REGRESSION_BOUND_REASON: str = "regression_bound"

#: A loss-score function maps a world model to its scalar held-out recon+KL loss
#: (LOWER is better). Note the INVERTED semantics vs the pre-ENABLEMENT
#: ``policy -> return`` (higher-was-better) signature.
LossScoreFn = Callable[["WorldModelProtocol"], float]


@runtime_checkable
class RevertCounterProtocol(Protocol):
    """Minimal interface for the on-device-learning revert counter.

    Implemented by ``telemetry.metrics.MetricsRegistry``. Kept tiny so the gate
    never imports the heavy metrics module and stays unit-testable with a spy.
    """

    def inc_on_device_learning_reverted(self, reason: str, amount: int = 1) -> None:
        """Increment the revert counter for ``reason`` (default amount 1)."""
        ...


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Outcome of one RSSM-vs-RSSM recon-loss regression-gate evaluation.

    Attributes:
        promoted: ``True`` iff the candidate passed the (inverted) regression
            bound and was marked active; ``False`` iff it was reverted.
        candidate_loss: The candidate RSSM's held-out recon+KL loss (lower is
            better; may be non-finite for a heavily-degraded candidate).
        baseline_loss: The live baseline RSSM's held-out recon+KL loss.
        delta: ``candidate_loss - baseline_loss``. **POSITIVE ⇒ candidate WORSE**
            (its loss is higher). This INVERTS the pre-ENABLEMENT convention
            where ``delta`` was a return difference (negative ⇒ worse).
        tolerance: The ``regression_tolerance`` the decision used (added to the
            baseline loss to form the promote ceiling).
    """

    promoted: bool
    candidate_loss: float
    baseline_loss: float
    delta: float
    tolerance: float


class RegressionGate:
    """Score a candidate RSSM vs the live baseline RSSM and promote-or-revert.

    Args:
        cfg: On-device-learning config (``regression_tolerance``, ``scoring_seed``).
        slot_store: SHA-256-stamped slot store. ``mark_active`` is called on a
            PROMOTE decision; never touched on REVERT.
        metrics: Optional revert counter. When ``None`` the revert path still
            reverts (just no metric is recorded).
        held_out_batch: The FIXED held-out ``(B, T, ...)`` sequence-dict batch the
            default scorer scores both models against. Required when ``score_fn``
            is not injected. Built (WS-E2) over a held-out replay slice DISJOINT
            from the refine batch.
        decoders: The SHARED reconstruction heads used to score BOTH baseline and
            candidate (recon heads are external to the RSSM ``state_dict``).
            Required when ``score_fn`` is not injected.
        score_fn: Optional injected scorer (``world_model -> loss``, LOWER is
            better). When ``None`` a default closing over ``held_out_batch`` +
            ``decoders`` + ``cfg.scoring_seed`` (calling
            :func:`~.scoring.score_dynamics`) is built. Injecting it decouples the
            decision logic from the torch scoring (unit tests).
    """

    def __init__(
        self,
        *,
        cfg: OnDeviceLearningConfig,
        slot_store: OnDeviceSlotStore,
        metrics: RevertCounterProtocol | None = None,
        held_out_batch: Mapping[str, Tensor] | None = None,
        decoders: RawModalityDecoders | None = None,
        score_fn: LossScoreFn | None = None,
    ) -> None:
        self._cfg = cfg
        self._slot_store = slot_store
        self._metrics = metrics
        self._held_out_batch = held_out_batch
        self._decoders = decoders
        self._score_fn = score_fn or self._build_default_score_fn()

    def _build_default_score_fn(self) -> LossScoreFn:
        """Build the default recon-loss scorer closing over the shared inputs.

        The SAME held-out batch, SAME shared decoders, and SAME ``scoring_seed``
        score both baseline and candidate so the comparison is apples-to-apples
        and deterministic.
        """
        held_out_batch = self._held_out_batch
        decoders = self._decoders
        if held_out_batch is None or decoders is None:
            msg = (
                "RegressionGate needs both held_out_batch and decoders when no "
                "score_fn is injected"
            )
            raise ValueError(msg)
        seed = self._cfg.scoring_seed

        def _default(world_model: WorldModelProtocol) -> float:
            return score_dynamics(world_model, held_out_batch, decoders, seed=seed)

        return _default

    def evaluate(
        self,
        *,
        candidate_world_model: WorldModelProtocol,
        baseline_world_model: WorldModelProtocol,
        slot: CandidateSlot,
    ) -> GateDecision:
        """Score candidate vs baseline RSSM and promote-or-revert ``slot``.

        Args:
            candidate_world_model: The refined candidate RSSM (the refined slot
                loaded into a deep copy of the live model).
            baseline_world_model: The live baseline RSSM (current weights).
            slot: The persisted candidate slot; marked active on a PROMOTE.

        Returns:
            A :class:`GateDecision` describing the outcome + both losses + the
            signed delta (positive ⇒ candidate worse).
        """
        tolerance = self._cfg.regression_tolerance
        candidate_loss = self._score_fn(candidate_world_model)
        baseline_loss = self._score_fn(baseline_world_model)
        delta = candidate_loss - baseline_loss

        # PROMOTE iff the candidate's loss is FINITE and not worse than the
        # baseline by more than the tolerance: candidate_loss <= baseline_loss +
        # tolerance. A non-finite candidate loss (KL blown up by a heavily-
        # degraded candidate) is correctly treated as worse-than-baseline and
        # reverts — the comparison is guarded by an explicit finiteness check so a
        # NaN candidate (where ``nan <= anything`` is False) AND a +inf candidate
        # both deterministically REVERT.
        promoted = math.isfinite(candidate_loss) and candidate_loss <= baseline_loss + tolerance

        if promoted:
            self._slot_store.mark_active(slot)
            _log.info(
                "on_device_candidate_promoted",
                candidate_loss=candidate_loss,
                baseline_loss=baseline_loss,
                delta=delta,
                tolerance=tolerance,
                digest=slot.digest,
            )
        else:
            if self._metrics is not None:
                self._metrics.inc_on_device_learning_reverted(_REGRESSION_BOUND_REASON)
            _log.warning(
                "on_device_candidate_reverted",
                reason=_REGRESSION_BOUND_REASON,
                candidate_loss=candidate_loss,
                baseline_loss=baseline_loss,
                delta=delta,
                tolerance=tolerance,
                candidate_finite=math.isfinite(candidate_loss),
                digest=slot.digest,
            )

        return GateDecision(
            promoted=promoted,
            candidate_loss=candidate_loss,
            baseline_loss=baseline_loss,
            delta=delta,
            tolerance=tolerance,
        )


__all__ = ["GateDecision", "LossScoreFn", "RegressionGate", "RevertCounterProtocol"]
