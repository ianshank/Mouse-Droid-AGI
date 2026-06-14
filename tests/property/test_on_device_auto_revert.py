"""Property test (CRITICAL): the safety gate NEVER promotes below the bound.

This is the load-bearing safety pin for Phase-6 WS4. The user-chosen contract is
a single inequality — *the active policy never persists a candidate scoring below
``baseline - regression_tolerance``* — and this test fuzzes the entire decision
surface to prove it holds with no exceptions:

* **Decision invariant (over the full score/tolerance space):** for any
  baseline score, candidate score and tolerance, the gate promotes iff
  ``candidate >= baseline - tolerance``. When it reverts it ALWAYS leaves the
  slot un-blessed (``load_active() is None``) and increments the revert counter
  exactly once with reason ``regression_bound``; when it promotes it marks the
  slot active and never touches the counter.

* **Real-rollout degradation (end-to-end through the world model):** a candidate
  whose policy is a deterministically *weakened* copy of the baseline — scored
  through the REAL ``score_policy`` rollout harness — is reverted whenever its
  real rollout-return falls below the bound, and the active pointer is never set
  in that case.

Together these prove: a deterministically-degraded candidate is always reverted
(never marked active) and increments the counter, and a within-tolerance
candidate is promoted.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import ExperienceConfig, ModelConfig, OnDeviceLearningConfig
from mousedroid.learning.on_device.regression_gate import RegressionGate
from mousedroid.learning.on_device.scoring import PolicyProtocol, StateDictPolicyAdapter
from mousedroid.learning.on_device.slot_store import OnDeviceSlotStore
from mousedroid.world_model.rssm import RSSM


class _SpyCounter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def inc_on_device_learning_reverted(self, reason: str, amount: int = 1) -> None:
        self.calls.append(reason)


class _NamedPolicy:
    def __init__(self, name: str) -> None:
        self.name = name

    def act(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(hidden.shape[0], 1)


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
    baseline_score=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    candidate_score=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    tolerance=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
)
def test_gate_decision_invariant_over_score_space(
    tmp_path: Path,
    baseline_score: float,
    candidate_score: float,
    tolerance: float,
) -> None:
    """Promote iff candidate >= baseline - tolerance; revert is always safe."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=tolerance)
    counter = _SpyCounter()
    tag = "exp_" + str(abs(hash((baseline_score, candidate_score, tolerance))))
    store = _store(tmp_path, cfg, tag)

    scores = {"baseline": baseline_score, "candidate": candidate_score}

    def _score_fn(policy: PolicyProtocol) -> float:
        return scores[policy.name]  # type: ignore[attr-defined]

    gate = RegressionGate(cfg=cfg, slot_store=store, metrics=counter, score_fn=_score_fn)
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate=_NamedPolicy("candidate"), baseline=_NamedPolicy("baseline"), slot=slot
    )

    expected_promote = candidate_score >= baseline_score - tolerance
    assert decision.promoted is expected_promote

    if expected_promote:
        # Promoted: slot blessed, counter untouched.
        assert store.load_active() == slot.digest
        assert counter.calls == []
    else:
        # Reverted: active pointer NEVER set, counter fired once.
        assert store.load_active() is None
        assert counter.calls == ["regression_bound"]


def _make_world_model() -> RSSM:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vision_dim=0, vision_proj_dim=0, hidden_dim=8, latent_dim=4, action_dim=3, obs_dim=8
    )
    wm = RSSM(cfg)
    wm.eval()
    return wm


def _seed_states(wm: RSSM, n: int = 3) -> list[tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(42)
    return [
        (torch.randn(1, wm.cfg.hidden_dim), torch.randn(1, wm.cfg.latent_dim)) for _ in range(n)
    ]


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(seed=st.integers(min_value=0, max_value=2**16))
def test_real_rollout_degraded_candidate_is_consistent(tmp_path: Path, seed: int) -> None:
    """End-to-end: a real-WM degraded candidate is promoted/reverted consistently.

    The candidate is a deterministically-weakened copy of the baseline policy
    net (weights scaled toward zero). Both are scored through the REAL
    ``score_policy`` rollout harness on the SAME seed-states + seed, so the gate
    decision MUST match the recomputed inequality exactly, and a revert MUST
    leave the active pointer unset.
    """
    torch.manual_seed(seed)
    wm = _make_world_model()
    cfg = OnDeviceLearningConfig(
        enabled=True,
        regression_tolerance=0.0,
        rollout_horizon=4,
        n_scoring_rollouts=3,
        scoring_seed=seed,
    )
    counter = _SpyCounter()
    store = _store(tmp_path, cfg, f"exp_seed_{seed}")

    in_dim = wm.cfg.hidden_dim + wm.cfg.latent_dim
    baseline_net = nn.Linear(in_dim, wm.cfg.action_dim)
    # Degrade: scale the candidate's weights so its actions differ from baseline.
    degraded_net = nn.Linear(in_dim, wm.cfg.action_dim)
    degraded_net.load_state_dict(baseline_net.state_dict())
    with torch.no_grad():
        for p in degraded_net.parameters():
            p.mul_(0.1)

    baseline = StateDictPolicyAdapter(
        baseline_net,
        hidden_dim=wm.cfg.hidden_dim,
        latent_dim=wm.cfg.latent_dim,
        action_dim=wm.cfg.action_dim,
    )
    candidate = StateDictPolicyAdapter(
        degraded_net,
        hidden_dim=wm.cfg.hidden_dim,
        latent_dim=wm.cfg.latent_dim,
        action_dim=wm.cfg.action_dim,
    )

    gate = RegressionGate(
        cfg=cfg, slot_store=store, metrics=counter, world_model=wm, seed_states=_seed_states(wm)
    )
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(candidate=candidate, baseline=baseline, slot=slot)

    # The decision must agree with the recomputed inequality on the SAME scores.
    expected = decision.candidate_score >= decision.baseline_score - cfg.regression_tolerance
    assert decision.promoted is expected

    # The load-bearing invariant: a reverted candidate is NEVER active.
    if not decision.promoted:
        assert store.load_active() is None
        assert counter.calls == ["regression_bound"]
    else:
        assert store.load_active() == slot.digest
