"""Unit tests for the WS-E3 RSSM-vs-RSSM recon-loss regression gate.

Pins the SPIKE-LOCKED WS-E3 ENABLEMENT contract (see
``docs/superpowers/plans/2026-06-14-phase6-enablement.md`` — "SPIKE RESULTS …
WS-E3 gate — LOCKED"):

* the gate metric is the held-out **reconstruction+KL loss**
  ``train_sequence(batch, decoders)["loss"]`` on a FIXED ``(B, T, ...)`` batch
  under ``model.eval()`` + ``torch.no_grad()`` — **LOWER IS BETTER** (it scores
  the WORLD MODEL's dynamics quality on real data, NOT a policy's imagined
  return — the retired imagined-return metric self-games on reward-head
  inflation);
* the direction INVERTS the pre-ENABLEMENT gate: PROMOTE iff
  ``candidate_loss <= baseline_loss + regression_tolerance``;
* ``GateDecision.delta`` is ``candidate_loss - baseline_loss`` with the sign
  convention POSITIVE delta = WORSE (a higher loss is worse);
* on PROMOTE the slot is marked active; on REVERT the slot is left un-blessed,
  the baseline RSSM is untouched, AND
  ``inc_on_device_learning_reverted("regression_bound")`` increments once;
* the decision is DETERMINISTIC: the same fixed seed + batch + decoders +
  weights ALWAYS yields the identical promote/revert decision;
* a non-finite (very-large / NaN / Inf) candidate loss is correctly > baseline
  ⇒ REVERT (a heavily-degraded candidate blows KL up).

The gate's ``score_fn`` (``world_model -> loss``) is injectable so the decision
logic is unit-testable with pinned losses without the torch rollout; the default
``score_dynamics`` scorer is exercised through the real-RSSM path below.

Uses tiny ``ModelConfig`` dims (hidden=8, latent=4, action=3) and
``Settings(mock_hardware=True)`` (a bare ``Settings()`` raises the distance-
sensor validator).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from mousedroid.config.schema import (
    ExperienceConfig,
    ModelConfig,
    OnDeviceLearningConfig,
)
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.learning.on_device.regression_gate import GateDecision, RegressionGate
from mousedroid.learning.on_device.rssm_refiner import build_sequence_batch
from mousedroid.learning.on_device.scoring import score_dynamics
from mousedroid.learning.on_device.slot_store import OnDeviceSlotStore
from mousedroid.world_model.protocol import WorldModelProtocol
from mousedroid.world_model.rssm import RSSM, RawModalityDecoders

_DEVICE = torch.device("cpu")
_SEED = 1234


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_cfg() -> ModelConfig:
    """Tiny deterministic ModelConfig (vision + lidar OFF)."""
    return ModelConfig(
        vision_dim=0,
        vision_proj_dim=0,
        ultrasonic_dim=1,
        ultrasonic_proj_dim=4,
        motor_state_dim=4,
        hidden_dim=8,
        latent_dim=4,
        action_dim=3,
        obs_dim=8,
        motor_proj_dim=4,
    )


def _make_rssm(cfg: ModelConfig, *, seed: int = 0) -> RSSM:
    torch.manual_seed(seed)
    wm = RSSM(cfg)
    wm.eval()
    return wm


def _make_records(n: int, *, seed: int = 7) -> list[MouseDroidExperienceRecord]:
    rng = np.random.default_rng(seed)
    records: list[MouseDroidExperienceRecord] = []
    for i in range(n):
        records.append(
            MouseDroidExperienceRecord(
                timestamp=float(i),
                vision_features=np.zeros(0, dtype=np.float32),
                distance_m=float(rng.uniform(0.1, 2.0)),
                motor_state=rng.standard_normal(4).astype(np.float32),
                action=rng.standard_normal(3).astype(np.float32),
                reward=float(rng.uniform(-1.0, 1.0)),
            )
        )
    return records


def _make_batch(cfg: ModelConfig, wm: RSSM) -> dict[str, torch.Tensor]:
    return build_sequence_batch(
        _make_records(40), cfg, wm.encoder, sequence_length=4, n_episodes=3, device=_DEVICE
    )


def _ocfg(**overrides: object) -> OnDeviceLearningConfig:
    base: dict[str, object] = {"enabled": True, "regression_tolerance": 0.0, "scoring_seed": _SEED}
    base.update(overrides)
    return OnDeviceLearningConfig(**base)  # type: ignore[arg-type]


def _store(tmp_path: Path, cfg: OnDeviceLearningConfig, tag: str = "exp") -> OnDeviceSlotStore:
    experience = ExperienceConfig(path=str(tmp_path / tag))
    return OnDeviceSlotStore(experience_cfg=experience, on_device_cfg=cfg)


class _SpyCounter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def inc_on_device_learning_reverted(self, reason: str, amount: int = 1) -> None:
        self.calls.append(reason)


def _degrade(wm: RSSM, scale: float) -> RSSM:
    """Return a deep copy of ``wm`` with all params scaled by ``scale`` (degrades it).

    A modest ``scale`` (e.g. 2.0) keeps the held-out loss FINITE but strictly
    higher than baseline (a clean "worse but finite" candidate); an aggressive
    scale blows the KL up to non-finite (covered separately by the pinned-loss
    non-finite revert test).
    """
    degraded = copy.deepcopy(wm)
    with torch.no_grad():
        for param in degraded.parameters():
            param.mul_(scale)
    degraded.eval()
    return degraded


# ---------------------------------------------------------------------------
# score_dynamics — the held-out recon+KL loss scorer (lower is better)
# ---------------------------------------------------------------------------


def test_score_dynamics_returns_train_sequence_loss() -> None:
    """score_dynamics returns float(train_sequence(batch, decoders)["loss"])."""
    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    batch = _make_batch(cfg, wm)
    decoders = RawModalityDecoders(cfg)

    loss = score_dynamics(wm, batch, decoders, seed=_SEED)

    # Recompute the loss with the SAME seed directly to confirm the value.
    torch.manual_seed(_SEED)
    wm.eval()
    with torch.no_grad():
        expected = float(wm.train_sequence(batch, decoders)["loss"])
    assert loss == pytest.approx(expected)
    assert isinstance(loss, float)


def test_score_dynamics_is_deterministic() -> None:
    """Same model + batch + decoders + seed ⇒ byte-identical loss."""
    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    batch = _make_batch(cfg, wm)
    decoders = RawModalityDecoders(cfg)

    a = score_dynamics(wm, batch, decoders, seed=_SEED)
    b = score_dynamics(wm, batch, decoders, seed=_SEED)
    assert a == b


def test_score_dynamics_runs_under_no_grad_and_restores_state() -> None:
    """No grads leak onto params; global RNG + train-mode are restored."""
    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    wm.train()
    batch = _make_batch(cfg, wm)
    decoders = RawModalityDecoders(cfg)

    torch.manual_seed(999)
    before_rng = torch.get_rng_state()

    score_dynamics(wm, batch, decoders, seed=_SEED)

    assert torch.equal(before_rng, torch.get_rng_state())
    assert wm.training is True  # train-mode restored
    for param in wm.parameters():
        assert param.grad is None


def test_score_dynamics_degraded_model_scores_worse() -> None:
    """A genuinely-degraded RSSM has a strictly HIGHER held-out loss (worse)."""
    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    batch = _make_batch(cfg, wm)
    decoders = RawModalityDecoders(cfg)

    baseline_loss = score_dynamics(wm, batch, decoders, seed=_SEED)
    degraded_loss = score_dynamics(_degrade(wm, 2.0), batch, decoders, seed=_SEED)

    assert degraded_loss > baseline_loss


def test_score_dynamics_shared_decoders_score_both(tmp_path: Path) -> None:
    """The SAME decoders instance scores baseline + candidate (recon heads external)."""
    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    batch = _make_batch(cfg, wm)
    decoders = RawModalityDecoders(cfg)

    # Scoring the identical model twice with the same shared decoders + seed is
    # byte-identical regardless of order.
    first = score_dynamics(wm, batch, decoders, seed=_SEED)
    second = score_dynamics(copy.deepcopy(wm), batch, decoders, seed=_SEED)
    assert first == pytest.approx(second)


def test_score_dynamics_rejects_non_rssm_engine() -> None:
    """A non-RSSM engine (no ``train_sequence``) raises TypeError (capability guard)."""
    cfg = _model_cfg()
    wm = _make_rssm(cfg)
    batch = _make_batch(cfg, wm)
    decoders = RawModalityDecoders(cfg)

    class _NotAnRSSM:
        """A stand-in engine without ``train_sequence`` (e.g. DualStreamRSSM)."""

    with pytest.raises(TypeError, match="train_sequence"):
        score_dynamics(_NotAnRSSM(), batch, decoders, seed=_SEED)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RegressionGate — recon-loss promote/revert (INVERTED direction)
# ---------------------------------------------------------------------------


def _gate_with_pinned_losses(
    tmp_path: Path,
    cfg: OnDeviceLearningConfig,
    counter: _SpyCounter,
    losses: dict[int, float],
) -> tuple[RegressionGate, OnDeviceSlotStore]:
    """Build a gate whose injected score_fn returns a pinned loss per model id()."""
    store = _store(tmp_path, cfg)

    def _score_fn(world_model: WorldModelProtocol) -> float:
        return losses[id(world_model)]

    return RegressionGate(cfg=cfg, slot_store=store, metrics=counter, score_fn=_score_fn), store


def test_promote_at_tolerance_boundary(tmp_path: Path) -> None:
    """candidate_loss == baseline_loss + tol ⇒ PROMOTE (<=)."""
    cfg = _ocfg(regression_tolerance=1.0)
    counter = _SpyCounter()
    baseline = object()
    candidate = object()
    losses = {id(baseline): 2.0, id(candidate): 3.0}  # 3.0 == 2.0 + 1.0
    gate, store = _gate_with_pinned_losses(tmp_path, cfg, counter, losses)
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate_world_model=candidate, baseline_world_model=baseline, slot=slot
    )

    assert decision.promoted is True
    assert counter.calls == []
    assert store.load_active() == slot.digest


def test_revert_just_past_boundary(tmp_path: Path) -> None:
    """candidate_loss just above baseline_loss + tol ⇒ REVERT + counter."""
    cfg = _ocfg(regression_tolerance=1.0)
    counter = _SpyCounter()
    baseline = object()
    candidate = object()
    losses = {id(baseline): 2.0, id(candidate): 3.01}  # > 2.0 + 1.0
    gate, store = _gate_with_pinned_losses(tmp_path, cfg, counter, losses)
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate_world_model=candidate, baseline_world_model=baseline, slot=slot
    )

    assert decision.promoted is False
    assert counter.calls == ["regression_bound"]
    assert store.load_active() is None


def test_promote_when_candidate_loss_lower(tmp_path: Path) -> None:
    """A candidate with a LOWER loss (better dynamics) is promoted."""
    cfg = _ocfg(regression_tolerance=0.0)
    counter = _SpyCounter()
    baseline = object()
    candidate = object()
    losses = {id(baseline): 5.0, id(candidate): 4.0}
    gate, store = _gate_with_pinned_losses(tmp_path, cfg, counter, losses)
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate_world_model=candidate, baseline_world_model=baseline, slot=slot
    )

    assert decision.promoted is True
    assert counter.calls == []


def test_gate_decision_delta_positive_means_worse(tmp_path: Path) -> None:
    """GateDecision.delta == candidate_loss - baseline_loss; positive = worse."""
    cfg = _ocfg(regression_tolerance=10.0)
    counter = _SpyCounter()
    baseline = object()
    candidate = object()
    losses = {id(baseline): 2.0, id(candidate): 5.0}
    gate, store = _gate_with_pinned_losses(tmp_path, cfg, counter, losses)
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate_world_model=candidate, baseline_world_model=baseline, slot=slot
    )

    assert isinstance(decision, GateDecision)
    assert decision.candidate_loss == 5.0
    assert decision.baseline_loss == 2.0
    # Positive delta = the candidate's loss is HIGHER = WORSE.
    assert decision.delta == pytest.approx(5.0 - 2.0)
    assert decision.delta > 0
    # Within the generous tolerance, still promoted despite a positive delta.
    assert decision.promoted is True


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 1e30])
def test_non_finite_candidate_loss_reverts(tmp_path: Path, bad: float) -> None:
    """A very-large / non-finite candidate loss is > baseline ⇒ REVERT."""
    cfg = _ocfg(regression_tolerance=0.0)
    counter = _SpyCounter()
    baseline = object()
    candidate = object()
    losses = {id(baseline): 3.35, id(candidate): bad}
    gate, store = _gate_with_pinned_losses(tmp_path, cfg, counter, losses)
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate_world_model=candidate, baseline_world_model=baseline, slot=slot
    )

    assert decision.promoted is False
    assert counter.calls == ["regression_bound"]
    assert store.load_active() is None


def test_metrics_none_revert_does_not_raise(tmp_path: Path) -> None:
    """A revert with no metrics counter is a safe no-op (still reverts)."""
    cfg = _ocfg(regression_tolerance=0.0)
    store = _store(tmp_path, cfg)
    baseline = object()
    candidate = object()
    losses = {id(baseline): 1.0, id(candidate): 2.0}

    def _score_fn(world_model: WorldModelProtocol) -> float:
        return losses[id(world_model)]

    gate = RegressionGate(cfg=cfg, slot_store=store, metrics=None, score_fn=_score_fn)
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate_world_model=candidate, baseline_world_model=baseline, slot=slot
    )

    assert decision.promoted is False
    assert store.load_active() is None


# ---------------------------------------------------------------------------
# RegressionGate — default scorer end-to-end over real RSSMs
# ---------------------------------------------------------------------------


def test_default_scorer_promotes_identical_candidate(tmp_path: Path) -> None:
    """An identical-weights candidate scores == baseline ⇒ delta 0 ⇒ PROMOTE.

    Baseline = the live RSSM; candidate = a deep copy with identical weights. The
    shared decoders + held-out batch + fixed seed give byte-identical losses, so
    the delta is exactly 0 and the candidate is promoted at any tolerance >= 0.
    """
    cfg = _ocfg(regression_tolerance=0.0)
    model_cfg = _model_cfg()
    baseline_wm = _make_rssm(model_cfg)
    candidate_wm = copy.deepcopy(baseline_wm)
    batch = _make_batch(model_cfg, baseline_wm)
    decoders = RawModalityDecoders(model_cfg)
    counter = _SpyCounter()
    store = _store(tmp_path, cfg)

    gate = RegressionGate(
        cfg=cfg,
        slot_store=store,
        metrics=counter,
        held_out_batch=batch,
        decoders=decoders,
    )
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate_world_model=candidate_wm, baseline_world_model=baseline_wm, slot=slot
    )

    assert decision.delta == pytest.approx(0.0)
    assert decision.promoted is True
    assert counter.calls == []
    assert store.load_active() == slot.digest


def test_default_scorer_reverts_degraded_candidate(tmp_path: Path) -> None:
    """A genuinely-degraded candidate RSSM is reverted under the recon-loss gate."""
    cfg = _ocfg(regression_tolerance=0.0)
    model_cfg = _model_cfg()
    baseline_wm = _make_rssm(model_cfg)
    candidate_wm = _degrade(baseline_wm, 2.0)
    batch = _make_batch(model_cfg, baseline_wm)
    decoders = RawModalityDecoders(model_cfg)
    counter = _SpyCounter()
    store = _store(tmp_path, cfg)

    gate = RegressionGate(
        cfg=cfg, slot_store=store, metrics=counter, held_out_batch=batch, decoders=decoders
    )
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate_world_model=candidate_wm, baseline_world_model=baseline_wm, slot=slot
    )

    assert decision.delta > 0
    assert decision.promoted is False
    assert counter.calls == ["regression_bound"]
    assert store.load_active() is None


def test_default_scorer_is_deterministic(tmp_path: Path) -> None:
    """Same seed + batch + decoders + weights ⇒ identical decision across runs."""
    cfg = _ocfg(regression_tolerance=0.0)
    model_cfg = _model_cfg()
    baseline_wm = _make_rssm(model_cfg)
    candidate_wm = _degrade(baseline_wm, 2.0)
    batch = _make_batch(model_cfg, baseline_wm)
    # Determinism is defined for a FIXED held-out batch + FIXED shared decoders +
    # FIXED seed + FIXED weights — so the SAME decoders instance scores both runs
    # (mirrors the factory, which constructs decoders ONCE and reuses them).
    decoders = RawModalityDecoders(model_cfg)

    def _decide(tag: str) -> GateDecision:
        store = _store(tmp_path, cfg, tag=tag)
        gate = RegressionGate(
            cfg=cfg,
            slot_store=store,
            metrics=None,
            held_out_batch=batch,
            decoders=decoders,
        )
        slot = store.persist({"w": torch.zeros(2)})
        return gate.evaluate(
            candidate_world_model=candidate_wm, baseline_world_model=baseline_wm, slot=slot
        )

    d1 = _decide("det_a")
    d2 = _decide("det_b")
    assert d1.candidate_loss == d2.candidate_loss
    assert d1.baseline_loss == d2.baseline_loss
    assert d1.promoted == d2.promoted


def test_default_scorer_baseline_rssm_untouched_on_revert(tmp_path: Path) -> None:
    """The baseline RSSM's params are bitwise-unchanged after a revert."""
    cfg = _ocfg(regression_tolerance=0.0)
    model_cfg = _model_cfg()
    baseline_wm = _make_rssm(model_cfg)
    candidate_wm = _degrade(baseline_wm, 2.0)
    batch = _make_batch(model_cfg, baseline_wm)
    decoders = RawModalityDecoders(model_cfg)
    store = _store(tmp_path, cfg)

    before = {n: p.detach().clone() for n, p in baseline_wm.named_parameters()}
    gate = RegressionGate(
        cfg=cfg, slot_store=store, metrics=None, held_out_batch=batch, decoders=decoders
    )
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate_world_model=candidate_wm, baseline_world_model=baseline_wm, slot=slot
    )

    assert decision.promoted is False
    for name, param in baseline_wm.named_parameters():
        assert torch.equal(param, before[name]), f"baseline param {name!r} mutated on revert"


def test_default_scorer_requires_batch_and_decoders(tmp_path: Path) -> None:
    """Omitting both score_fn and the default scorer's deps raises."""
    cfg = _ocfg()
    store = _store(tmp_path, cfg)

    with pytest.raises(ValueError, match="held_out_batch and decoders"):
        RegressionGate(cfg=cfg, slot_store=store, metrics=None)
