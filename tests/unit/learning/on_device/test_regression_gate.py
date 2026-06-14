"""Unit tests for the WS4 safety-regression gate + auto-revert decision.

Pins the user-chosen promote/revert contract:

* candidate is PROMOTED iff ``candidate_return >= baseline_return - tolerance``;
* on PROMOTE the slot is marked active via the slot store;
* on REVERT the live policy is left untouched (slot NOT marked active) AND the
  ``inc_on_device_learning_reverted("regression_bound")`` counter increments;
* the decision is evaluated at the tolerance boundary (within/at/below);
* both scores + delta + decision are returned + structlogged.

The gate's ``score_fn`` is injectable so these decision-logic tests pin exact,
ordered scores without depending on the stochastic world-model rollout (that is
covered separately in ``test_scoring.py``). The default ``score_fn`` (the real
``score_policy``) is exercised end-to-end by the integration test.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mousedroid.config.schema import ExperienceConfig, OnDeviceLearningConfig
from mousedroid.learning.on_device.regression_gate import GateDecision, RegressionGate
from mousedroid.learning.on_device.scoring import PolicyProtocol
from mousedroid.learning.on_device.slot_store import OnDeviceSlotStore


class _SpyCounter:
    """Records ``inc_on_device_learning_reverted`` calls (RevertCounter shape)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def inc_on_device_learning_reverted(self, reason: str, amount: int = 1) -> None:
        self.calls.append(reason)


class _NamedPolicy:
    """A trivial policy stand-in identified by name for the injectable scorer."""

    def __init__(self, name: str) -> None:
        self.name = name

    def act(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(hidden.shape[0], 1)


def _make_store(tmp_path: Path, cfg: OnDeviceLearningConfig) -> OnDeviceSlotStore:
    experience = ExperienceConfig(path=str(tmp_path / "experience_root"))
    return OnDeviceSlotStore(experience_cfg=experience, on_device_cfg=cfg)


def _make_gate(
    tmp_path: Path,
    cfg: OnDeviceLearningConfig,
    counter: _SpyCounter,
    scores: dict[str, float],
) -> tuple[RegressionGate, OnDeviceSlotStore]:
    """Build a gate whose score_fn returns a pinned score per policy name."""
    store = _make_store(tmp_path, cfg)

    def _score_fn(policy: PolicyProtocol) -> float:
        return scores[policy.name]  # type: ignore[attr-defined]

    gate = RegressionGate(
        cfg=cfg,
        slot_store=store,
        metrics=counter,
        score_fn=_score_fn,
    )
    return gate, store


def test_promote_when_candidate_not_worse(tmp_path: Path) -> None:
    """An equal-or-better candidate is promoted; slot marked active."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=0.01)
    counter = _SpyCounter()
    gate, store = _make_gate(tmp_path, cfg, counter, {"baseline": 1.0, "candidate": 1.0})
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate=_NamedPolicy("candidate"), baseline=_NamedPolicy("baseline"), slot=slot
    )

    assert decision.promoted is True
    assert counter.calls == []
    assert store.load_active() == slot.digest


def test_revert_when_candidate_worse_beyond_tolerance(tmp_path: Path) -> None:
    """A worse candidate is reverted; counter increments; baseline untouched."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=0.0)
    counter = _SpyCounter()
    gate, store = _make_gate(tmp_path, cfg, counter, {"baseline": 1.0, "candidate": 0.5})
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate=_NamedPolicy("candidate"), baseline=_NamedPolicy("baseline"), slot=slot
    )

    assert decision.promoted is False
    assert counter.calls == ["regression_bound"]
    assert store.load_active() is None  # no active pointer on revert


def test_boundary_equal_to_tolerance_promotes(tmp_path: Path) -> None:
    """At exactly ``baseline - tolerance`` the candidate is promoted (>=)."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=1.0)
    counter = _SpyCounter()
    gate, store = _make_gate(tmp_path, cfg, counter, {"baseline": 2.0, "candidate": 1.0})
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate=_NamedPolicy("candidate"), baseline=_NamedPolicy("baseline"), slot=slot
    )

    assert decision.promoted is True
    assert counter.calls == []


def test_just_below_boundary_reverts(tmp_path: Path) -> None:
    """Just past the tolerance the candidate is reverted."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=1.0)
    counter = _SpyCounter()
    gate, store = _make_gate(tmp_path, cfg, counter, {"baseline": 2.0, "candidate": 0.99})
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate=_NamedPolicy("candidate"), baseline=_NamedPolicy("baseline"), slot=slot
    )

    assert decision.promoted is False
    assert counter.calls == ["regression_bound"]


def test_decision_carries_scores_and_delta(tmp_path: Path) -> None:
    """The returned decision exposes both scores + the delta."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=1.0)
    counter = _SpyCounter()
    gate, store = _make_gate(tmp_path, cfg, counter, {"baseline": 2.0, "candidate": 1.5})
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate=_NamedPolicy("candidate"), baseline=_NamedPolicy("baseline"), slot=slot
    )

    assert isinstance(decision, GateDecision)
    assert decision.candidate_score == 1.5
    assert decision.baseline_score == 2.0
    assert decision.delta == 1.5 - 2.0
    assert decision.promoted is True


def test_default_score_fn_requires_world_model_and_seed_states(tmp_path: Path) -> None:
    """Omitting both ``score_fn`` and the default scorer's deps raises."""
    import pytest

    cfg = OnDeviceLearningConfig(enabled=True)
    store = _make_store(tmp_path, cfg)

    with pytest.raises(ValueError, match="world_model and seed_states"):
        RegressionGate(cfg=cfg, slot_store=store, metrics=None)


def test_metrics_none_revert_does_not_raise(tmp_path: Path) -> None:
    """A revert with no metrics counter wired is a safe no-op (still reverts)."""
    cfg = OnDeviceLearningConfig(enabled=True, regression_tolerance=0.0)
    store = _make_store(tmp_path, cfg)

    def _score_fn(policy: PolicyProtocol) -> float:
        return {"baseline": 1.0, "candidate": 0.0}[policy.name]  # type: ignore[attr-defined]

    gate = RegressionGate(cfg=cfg, slot_store=store, metrics=None, score_fn=_score_fn)
    slot = store.persist({"w": torch.zeros(2)})

    decision = gate.evaluate(
        candidate=_NamedPolicy("candidate"), baseline=_NamedPolicy("baseline"), slot=slot
    )

    assert decision.promoted is False
    assert store.load_active() is None
