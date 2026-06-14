"""Safety-regression gate + auto-revert for on-device learning (Phase 6 WS4).

This is the highest-risk seam of the on-device-learning loop: it decides whether
a freshly-produced candidate policy is safe to promote into the live policy, or
must be reverted. The user-chosen contract:

* Score the candidate AND the live baseline with the SAME fixed seed-states +
  seed via the world-model rollout-return harness (``scoring.score_policy``).
* PROMOTE iff ``candidate_score >= baseline_score - regression_tolerance``.
  On promote, mark the candidate slot ACTIVE in the slot store (the live-policy
  hot-swap itself stays WS5).
* Otherwise REVERT: leave the live policy untouched (do NOT mark the slot
  active) and increment ``inc_on_device_learning_reverted("regression_bound")``.

Determinism is load-bearing: both scores come from the same deterministic
harness on the same inputs, so the decision is reproducible. The scoring call is
torch-heavy; the coordinator runs the gate OFF the event loop on the slow
cadence (``asyncio.to_thread``), never on the 30 Hz hot loop.

The ``score_fn`` is injectable (DI) so the decision logic is unit-testable with
pinned scores, and so WS5 can swap the live policy net behind the same seam. The
default ``score_fn`` closes over the world model + seed-states + config knobs and
calls the REUSED ``score_policy``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mousedroid.learning.on_device.scoring import PolicyProtocol, SeedState, score_policy
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import OnDeviceLearningConfig
    from mousedroid.learning.on_device.slot_store import CandidateSlot, OnDeviceSlotStore
    from mousedroid.world_model.protocol import WorldModelProtocol

_log = get_logger(__name__)

#: Revert reason recorded on the Prometheus counter when the candidate fails the
#: regression bound. Must stay in ``telemetry.metrics._ON_DEVICE_REVERT_REASONS``.
_REGRESSION_BOUND_REASON: str = "regression_bound"

#: A score function maps a policy to its scalar rollout-return score.
ScoreFn = Callable[[PolicyProtocol], float]


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
    """Outcome of one safety-regression gate evaluation.

    Attributes:
        promoted: ``True`` iff the candidate passed the regression bound and was
            marked active; ``False`` iff it was reverted.
        candidate_score: The candidate policy's rollout-return score.
        baseline_score: The live baseline policy's rollout-return score.
        delta: ``candidate_score - baseline_score`` (negative ⇒ candidate worse).
        tolerance: The ``regression_tolerance`` the decision used.
    """

    promoted: bool
    candidate_score: float
    baseline_score: float
    delta: float
    tolerance: float


class RegressionGate:
    """Score a candidate vs the live baseline and promote-or-revert.

    Args:
        cfg: On-device-learning config (``regression_tolerance``,
            ``rollout_horizon``, ``n_scoring_rollouts``, ``scoring_seed``).
        slot_store: SHA-256-stamped slot store. ``mark_active`` is called on a
            PROMOTE decision; never touched on REVERT.
        metrics: Optional revert counter. When ``None`` the revert path still
            reverts (just no metric is recorded).
        world_model: REUSED RSSM world model for the default scorer. May be
            ``None`` only when ``score_fn`` is injected (the unit-test seam).
        seed_states: Fixed ``(hidden, latent)`` start states for the default
            scorer. Required when ``score_fn`` is not injected.
        score_fn: Optional injected scorer (``policy -> float``). When ``None``
            a default closing over ``world_model`` + ``seed_states`` + the
            config knobs is built. Injecting it decouples the decision logic
            from the stochastic rollout (unit tests) and is the WS5 seam.
    """

    def __init__(
        self,
        *,
        cfg: OnDeviceLearningConfig,
        slot_store: OnDeviceSlotStore,
        metrics: RevertCounterProtocol | None = None,
        world_model: WorldModelProtocol | None = None,
        seed_states: Sequence[SeedState] | None = None,
        score_fn: ScoreFn | None = None,
    ) -> None:
        self._cfg = cfg
        self._slot_store = slot_store
        self._metrics = metrics
        self._world_model = world_model
        self._seed_states = seed_states
        self._score_fn = score_fn or self._build_default_score_fn()

    def _build_default_score_fn(self) -> ScoreFn:
        """Build the default scorer closing over the world model + seed-states."""
        world_model = self._world_model
        seed_states = self._seed_states
        if world_model is None or seed_states is None:
            msg = (
                "RegressionGate needs both world_model and seed_states when no "
                "score_fn is injected"
            )
            raise ValueError(msg)
        horizon = self._cfg.rollout_horizon
        n_rollouts = self._cfg.n_scoring_rollouts
        seed = self._cfg.scoring_seed

        def _default(policy: PolicyProtocol) -> float:
            return score_policy(
                policy,
                world_model,
                seed_states,
                horizon=horizon,
                n_rollouts=n_rollouts,
                seed=seed,
            )

        return _default

    def evaluate(
        self,
        *,
        candidate: PolicyProtocol,
        baseline: PolicyProtocol,
        slot: CandidateSlot,
    ) -> GateDecision:
        """Score candidate vs baseline and promote-or-revert ``slot``.

        Args:
            candidate: The freshly-produced candidate policy.
            baseline: The live/cloud baseline policy to beat.
            slot: The persisted candidate slot; marked active on a PROMOTE.

        Returns:
            A :class:`GateDecision` describing the outcome + both scores.
        """
        tolerance = self._cfg.regression_tolerance
        candidate_score = self._score_fn(candidate)
        baseline_score = self._score_fn(baseline)
        delta = candidate_score - baseline_score

        # PROMOTE iff the candidate is not worse than the baseline by more than
        # the tolerance, i.e. candidate >= baseline - tolerance.
        promoted = candidate_score >= baseline_score - tolerance

        if promoted:
            self._slot_store.mark_active(slot)
            _log.info(
                "on_device_candidate_promoted",
                candidate_score=candidate_score,
                baseline_score=baseline_score,
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
                candidate_score=candidate_score,
                baseline_score=baseline_score,
                delta=delta,
                tolerance=tolerance,
                digest=slot.digest,
            )

        return GateDecision(
            promoted=promoted,
            candidate_score=candidate_score,
            baseline_score=baseline_score,
            delta=delta,
            tolerance=tolerance,
        )


__all__ = ["GateDecision", "RegressionGate", "RevertCounterProtocol", "ScoreFn"]
