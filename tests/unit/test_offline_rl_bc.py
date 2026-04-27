"""Tests for the Phase 2.1 behavior-cloning auxiliary loss on offline RL trainers."""

from __future__ import annotations

import pytest
import torch

from mousedroid.learning.offline_rl import CQLTrainer, IQLTrainer

_STATE_DIM = 8
_ACTION_DIM = 3
_BATCH = 16
_HIDDEN = 32


def _make_cql() -> CQLTrainer:
    return CQLTrainer(
        state_dim=_STATE_DIM,
        action_dim=_ACTION_DIM,
        hidden_dim=_HIDDEN,
    )


def _make_iql() -> IQLTrainer:
    return IQLTrainer(
        state_dim=_STATE_DIM,
        action_dim=_ACTION_DIM,
        hidden_dim=_HIDDEN,
    )


@pytest.mark.parametrize("factory", [_make_cql, _make_iql])
def test_bc_update_zero_weight_is_noop(factory: object) -> None:
    """weight=0.0 must be byte-identical to skipping the call."""
    trainer = factory()  # type: ignore[operator]
    states = torch.randn(_BATCH, _STATE_DIM)
    actions = torch.randn(_BATCH, _ACTION_DIM)

    snapshot = {k: v.detach().clone() for k, v in trainer.policy.state_dict().items()}
    out = trainer.bc_update(states, actions, weight=0.0)

    assert out == {"bc_loss": 0.0}
    for k, v in trainer.policy.state_dict().items():
        assert torch.equal(v, snapshot[k]), f"policy weights changed for {k}"


@pytest.mark.parametrize("factory", [_make_cql, _make_iql])
def test_bc_update_positive_weight_reduces_loss(factory: object) -> None:
    """A few BC steps on a fixed batch must drive loss strictly down."""
    torch.manual_seed(0)
    trainer = factory()  # type: ignore[operator]
    states = torch.randn(_BATCH, _STATE_DIM)
    actions = torch.randn(_BATCH, _ACTION_DIM)

    initial = trainer.bc_update(states, actions, weight=1.0)["bc_loss"]
    for _ in range(20):
        trainer.bc_update(states, actions, weight=1.0)
    final = trainer.bc_update(states, actions, weight=1.0)["bc_loss"]

    assert final < initial, f"BC loss did not decrease: {initial} -> {final}"


@pytest.mark.parametrize("factory", [_make_cql, _make_iql])
def test_bc_update_does_not_touch_q_network(factory: object) -> None:
    """BC is a pure actor regularizer; Q-network must remain frozen."""
    trainer = factory()  # type: ignore[operator]
    snapshot = {k: v.detach().clone() for k, v in trainer.q_network.state_dict().items()}
    states = torch.randn(_BATCH, _STATE_DIM)
    actions = torch.randn(_BATCH, _ACTION_DIM)

    trainer.bc_update(states, actions, weight=1.0)

    for k, v in trainer.q_network.state_dict().items():
        assert torch.equal(v, snapshot[k]), f"Q-network weights changed for {k}"
