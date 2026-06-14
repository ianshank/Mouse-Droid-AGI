"""Unit tests for the WS-E3 recon-loss regression-gate DECISION boundary.

Focused decision-logic tests with an injected pinned-loss ``score_fn`` (so no
torch rollout runs here — the real ``score_dynamics`` over real RSSMs is
exercised in ``test_rssm_gate.py``). Pins the SPIKE-LOCKED WS-E3 contract:

* the metric is a held-out recon+KL LOSS — **LOWER IS BETTER**;
* candidate is PROMOTED iff ``candidate_loss <= baseline_loss + tolerance``
  (INVERTED from the pre-ENABLEMENT higher-is-better imagined-return gate);
* ``GateDecision.delta == candidate_loss - baseline_loss`` with the sign
  convention POSITIVE delta = WORSE;
* on PROMOTE the slot is marked active; on REVERT the slot is left un-blessed
  AND ``inc_on_device_learning_reverted("regression_bound")`` increments once;
* a non-finite candidate loss reverts.

The gate's ``score_fn`` (``world_model -> loss``) is injectable so these decision
tests pin exact, ordered losses without the stochastic world-model rollout.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mousedroid.config.schema import ExperienceConfig, OnDeviceLearningConfig
from mousedroid.learning.on_device.regression_gate import GateDecision, RegressionGate
from mousedroid.learning.on_device.slot_store import OnDeviceSlotStore
from mousedroid.world_model.protocol import WorldModelProtocol


class _SpyCounter:
    """Records ``inc_on_device_learning_reverted`` calls (RevertCounter shape)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def inc_on_device_learning_reverted(self, reason: str, amount: int = 1) -> None:
        self.calls.append(reason)


def _make_store(tmp_path: Path, cfg: OnDeviceLearningConfig) -> OnDeviceSlotStore:
    experience = ExperienceConfig(path=str(tmp_path / "experience_root"))
    return OnDeviceSlotStore(experience_cfg=experience, on_device_cfg=cfg)


def _make_gate(
    tmp_path: Path,
    cfg: OnDeviceLearningConfig,
    counter: _SpyCounter,
    losses: dict[int, float],
) -> tuple[RegressionGate, OnDeviceSlotStore]:
    """Build a gate whose score_fn returns a pinned loss per model id()."""
    store = _make_store(tmp_path, cfg)

    def _score_fn(world_model: WorldModelProtocol) -> float:
        return losses[id(world_model)]

    gate = RegressionGate(cfg=cfg, slot_store=store, metrics=counter, score_fn=_score_fn)
    return gate, store


def _evaluate(
    gate: RegressionGate, store: OnDeviceSlotStore, candidate: object, baseline: object
) -> GateDecision:
    slot = store.persist({"w": torch.zeros(2)})
    return gate.evaluate(candidate_world_model=candidate, baseline_world_model=baseline, slot=slot)


def test_promote_when_candidate_loss_not_worse(tmp_path: Path) -> None:
    """An equal-or-lower-loss candidate is promoted; slot marked active."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=0.01)
    counter = _SpyCounter()
    baseline, candidate = object(), object()
    gate, store = _make_gate(tmp_path, cfg, counter, {id(baseline): 1.0, id(candidate): 1.0})

    decision = _evaluate(gate, store, candidate, baseline)

    assert decision.promoted is True
    assert counter.calls == []
    assert store.load_active() is not None


def test_revert_when_candidate_loss_worse_beyond_tolerance(tmp_path: Path) -> None:
    """A higher-loss candidate is reverted; counter increments; slot not active."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=0.0)
    counter = _SpyCounter()
    baseline, candidate = object(), object()
    gate, store = _make_gate(tmp_path, cfg, counter, {id(baseline): 1.0, id(candidate): 1.5})

    decision = _evaluate(gate, store, candidate, baseline)

    assert decision.promoted is False
    assert counter.calls == ["regression_bound"]
    assert store.load_active() is None


def test_boundary_equal_to_tolerance_promotes(tmp_path: Path) -> None:
    """At exactly ``baseline_loss + tolerance`` the candidate is promoted (<=)."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=1.0)
    counter = _SpyCounter()
    baseline, candidate = object(), object()
    gate, store = _make_gate(tmp_path, cfg, counter, {id(baseline): 1.0, id(candidate): 2.0})

    decision = _evaluate(gate, store, candidate, baseline)

    assert decision.promoted is True
    assert counter.calls == []


def test_just_above_boundary_reverts(tmp_path: Path) -> None:
    """Just past the tolerance ceiling the candidate is reverted."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=1.0)
    counter = _SpyCounter()
    baseline, candidate = object(), object()
    gate, store = _make_gate(tmp_path, cfg, counter, {id(baseline): 1.0, id(candidate): 2.01})

    decision = _evaluate(gate, store, candidate, baseline)

    assert decision.promoted is False
    assert counter.calls == ["regression_bound"]


def test_decision_carries_losses_and_signed_delta(tmp_path: Path) -> None:
    """The returned decision exposes both losses + the signed delta (positive=worse)."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=1.0)
    counter = _SpyCounter()
    baseline, candidate = object(), object()
    gate, store = _make_gate(tmp_path, cfg, counter, {id(baseline): 2.0, id(candidate): 2.5})

    decision = _evaluate(gate, store, candidate, baseline)

    assert isinstance(decision, GateDecision)
    assert decision.candidate_loss == 2.5
    assert decision.baseline_loss == 2.0
    # Positive delta = candidate loss higher = worse.
    assert decision.delta == pytest.approx(2.5 - 2.0)
    assert decision.delta > 0
    assert decision.promoted is True  # within the generous tolerance


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_non_finite_candidate_loss_reverts(tmp_path: Path, bad: float) -> None:
    """A non-finite candidate loss reverts (never assumed finite)."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=1e9)
    counter = _SpyCounter()
    baseline, candidate = object(), object()
    gate, store = _make_gate(tmp_path, cfg, counter, {id(baseline): 2.0, id(candidate): bad})

    decision = _evaluate(gate, store, candidate, baseline)

    assert decision.promoted is False
    assert counter.calls == ["regression_bound"]
    assert store.load_active() is None


def test_default_score_fn_requires_batch_and_decoders(tmp_path: Path) -> None:
    """Omitting both ``score_fn`` and the default scorer's deps raises."""
    cfg = OnDeviceLearningConfig(enabled=True)
    store = _make_store(tmp_path, cfg)

    with pytest.raises(ValueError, match="held_out_batch and decoders"):
        RegressionGate(cfg=cfg, slot_store=store, metrics=None)


def test_metrics_none_revert_does_not_raise(tmp_path: Path) -> None:
    """A revert with no metrics counter wired is a safe no-op (still reverts)."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=0.0)
    store = _make_store(tmp_path, cfg)
    baseline, candidate = object(), object()
    losses = {id(baseline): 1.0, id(candidate): 2.0}

    def _score_fn(world_model: WorldModelProtocol) -> float:
        return losses[id(world_model)]

    gate = RegressionGate(cfg=cfg, slot_store=store, metrics=None, score_fn=_score_fn)

    decision = _evaluate(gate, store, candidate, baseline)

    assert decision.promoted is False
    assert store.load_active() is None
