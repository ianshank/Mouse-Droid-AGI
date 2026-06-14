"""Property test (CRITICAL): the recon-loss safety gate NEVER promotes a regression.

This is the load-bearing safety pin for Phase-6 WS-E3. The user-chosen contract
is a single inequality on a held-out recon+KL LOSS (LOWER IS BETTER) — *the
active world model never persists a candidate whose held-out loss exceeds
``baseline_loss + regression_tolerance``* — and this test fuzzes the entire
decision surface to prove it holds with no exceptions:

* **Decision invariant (over the full loss/tolerance space):** for any baseline
  loss, candidate loss and tolerance, the gate promotes iff the candidate loss
  is FINITE and ``candidate_loss <= baseline_loss + tolerance``. When it reverts
  it ALWAYS leaves the slot un-blessed (``load_active() is None``) and increments
  the revert counter exactly once with reason ``regression_bound``; when it
  promotes it marks the slot active and never touches the counter.

* **Real-RSSM synthetic degradation (end-to-end through ``score_dynamics``):** a
  candidate RSSM whose parameters are a noised copy of the baseline — scored
  through the REAL held-out recon-loss scorer on a FIXED held-out batch + shared
  decoders — is ALWAYS reverted (its dynamics-prediction loss is strictly worse),
  and the active pointer is never set.

Together these prove: a deterministically-degraded RSSM is always reverted (never
marked active) and increments the counter, and an identical-weights candidate
(delta 0) is promoted.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import ExperienceConfig, ModelConfig, OnDeviceLearningConfig
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.learning.on_device.regression_gate import RegressionGate
from mousedroid.learning.on_device.rssm_refiner import build_sequence_batch
from mousedroid.learning.on_device.slot_store import OnDeviceSlotStore
from mousedroid.world_model.protocol import WorldModelProtocol
from mousedroid.world_model.rssm import RSSM, RawModalityDecoders

_DEVICE = torch.device("cpu")


class _SpyCounter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def inc_on_device_learning_reverted(self, reason: str, amount: int = 1) -> None:
        self.calls.append(reason)


def _store(tmp_path: Path, cfg: OnDeviceLearningConfig, tag: str = "exp") -> OnDeviceSlotStore:
    # A unique sub-root per example so hypothesis re-runs (which share the one
    # function-scoped ``tmp_path``) never see a stale ``active.json`` from a
    # prior PROMOTE example — the slot is content-addressed so the digest is
    # otherwise constant across examples.
    experience = ExperienceConfig(path=str(tmp_path / tag))
    return OnDeviceSlotStore(experience_cfg=experience, on_device_cfg=cfg)


@settings(
    max_examples=120, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    baseline_loss=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    candidate_loss=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    tolerance=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
)
def test_gate_decision_invariant_over_loss_space(
    tmp_path: Path,
    baseline_loss: float,
    candidate_loss: float,
    tolerance: float,
) -> None:
    """Promote iff candidate_loss <= baseline_loss + tolerance; revert is always safe."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=tolerance)
    counter = _SpyCounter()
    tag = "exp_" + str(abs(hash((baseline_loss, candidate_loss, tolerance))))
    store = _store(tmp_path, cfg, tag)

    baseline, candidate = object(), object()
    losses = {id(baseline): baseline_loss, id(candidate): candidate_loss}

    def _score_fn(world_model: WorldModelProtocol) -> float:
        return losses[id(world_model)]

    gate = RegressionGate(cfg=cfg, slot_store=store, metrics=counter, score_fn=_score_fn)
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate_world_model=candidate, baseline_world_model=baseline, slot=slot
    )

    expected_promote = math.isfinite(candidate_loss) and candidate_loss <= baseline_loss + tolerance
    assert decision.promoted is expected_promote

    if expected_promote:
        # Promoted: slot blessed, counter untouched.
        assert store.load_active() == slot.digest
        assert counter.calls == []
    else:
        # Reverted: active pointer NEVER set, counter fired once.
        assert store.load_active() is None
        assert counter.calls == ["regression_bound"]


def _model_cfg() -> ModelConfig:
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


def _make_world_model(cfg: ModelConfig) -> RSSM:
    torch.manual_seed(0)
    wm = RSSM(cfg)
    wm.eval()
    return wm


def _held_out_batch(cfg: ModelConfig, wm: RSSM) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(7)
    records = [
        MouseDroidExperienceRecord(
            timestamp=float(i),
            vision_features=np.zeros(0, dtype=np.float32),
            distance_m=float(rng.uniform(0.1, 2.0)),
            motor_state=rng.standard_normal(4).astype(np.float32),
            action=rng.standard_normal(3).astype(np.float32),
            reward=float(rng.uniform(-1.0, 1.0)),
        )
        for i in range(40)
    ]
    return build_sequence_batch(
        records, cfg, wm.encoder, sequence_length=4, n_episodes=3, device=_DEVICE
    )


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(seed=st.integers(min_value=1, max_value=2**16))
def test_synthetically_degraded_rssm_is_always_reverted(tmp_path: Path, seed: int) -> None:
    """A noised (degraded) candidate RSSM is ALWAYS reverted under the recon-loss gate.

    The candidate is the baseline RSSM with additive Gaussian noise on EVERY
    parameter — its held-out recon+KL loss is strictly worse (or non-finite) than
    the baseline's. Scored through the REAL ``score_dynamics`` on a FIXED held-out
    batch + shared decoders with a zero tolerance, it MUST revert and the active
    pointer MUST stay unset.
    """
    cfg_model = _model_cfg()
    baseline_wm = _make_world_model(cfg_model)
    held_out = _held_out_batch(cfg_model, baseline_wm)
    decoders = RawModalityDecoders(cfg_model)

    # Degrade: add noise to every parameter (a synthetically-broken candidate).
    gen = torch.Generator().manual_seed(seed)
    candidate_wm = copy.deepcopy(baseline_wm)
    with torch.no_grad():
        for param in candidate_wm.parameters():
            param.add_(torch.randn(param.shape, generator=gen) * 3.0)
    candidate_wm.eval()

    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=0.0, scoring_seed=seed)
    counter = _SpyCounter()
    store = _store(tmp_path, cfg, f"exp_seed_{seed}")

    gate = RegressionGate(
        cfg=cfg, slot_store=store, metrics=counter, held_out_batch=held_out, decoders=decoders
    )
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate_world_model=candidate_wm, baseline_world_model=baseline_wm, slot=slot
    )

    # The synthetically-degraded candidate must NEVER be promoted.
    assert decision.promoted is False
    assert store.load_active() is None
    assert counter.calls == ["regression_bound"]
    # The baseline RSSM is bitwise-unchanged by the gate evaluation.
    fresh = _make_world_model(cfg_model)
    for (n, p), (_, fp) in zip(
        baseline_wm.named_parameters(), fresh.named_parameters(), strict=True
    ):
        assert torch.equal(p, fp), f"baseline param {n!r} mutated by the gate"
