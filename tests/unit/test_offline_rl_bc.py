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


# -- Phase 2.1: dedicated bc_optimizer wiring ----------------------------------


def _make_cql_with_bc_lr(bc_lr: float | None) -> CQLTrainer:
    return CQLTrainer(
        state_dim=_STATE_DIM,
        action_dim=_ACTION_DIM,
        hidden_dim=_HIDDEN,
        bc_lr=bc_lr,
    )


@pytest.mark.parametrize("factory", [_make_cql, _make_iql])
def test_bc_optimizer_shared_when_lr_none(factory: object) -> None:
    """``bc_lr=None`` must alias ``bc_optimizer`` to ``policy_optimizer``."""
    trainer = factory()  # type: ignore[operator]
    assert trainer.bc_optimizer is trainer.policy_optimizer


def test_bc_optimizer_separate_when_lr_set() -> None:
    """``bc_lr`` set ⇒ dedicated Adam over policy params with that LR."""
    trainer = _make_cql_with_bc_lr(bc_lr=5e-4)
    assert trainer.bc_optimizer is not trainer.policy_optimizer
    assert trainer.bc_optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_bc_update_steps_only_bc_optimizer_when_dedicated() -> None:
    """When ``bc_lr`` is set, policy_optimizer state must not change on a BC step."""
    trainer = _make_cql_with_bc_lr(bc_lr=1e-3)
    states = torch.randn(_BATCH, _STATE_DIM)
    actions = torch.randn(_BATCH, _ACTION_DIM)

    # Force a real Adam internal state by running one step then snapshotting.
    trainer.policy_optimizer.zero_grad()
    fake_loss = trainer.policy(states).sum()
    fake_loss.backward()
    trainer.policy_optimizer.step()
    policy_state_before = {
        k: {kk: vv.clone() if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
        for k, v in trainer.policy_optimizer.state.items()
    }

    trainer.bc_update(states, actions, weight=1.0)

    # policy_optimizer's internal Adam state (exp_avg, exp_avg_sq, step) must
    # be untouched — BC only steps the dedicated bc_optimizer.
    for param, state in trainer.policy_optimizer.state.items():
        prior = policy_state_before[param]
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state and key in prior:
                assert torch.equal(
                    state[key], prior[key]
                ), f"policy_optimizer.{key} mutated by BC step despite bc_lr set"


def test_legacy_checkpoint_loads_without_bc_state(tmp_path: object) -> None:
    """A checkpoint saved by a shared-optimizer trainer must load into a
    dedicated-bc trainer without a key error (backwards compat)."""
    from pathlib import Path

    tmp = Path(str(tmp_path))
    legacy = _make_cql()  # bc_lr=None → shared optimizer
    legacy_path = tmp / "legacy.pt"
    legacy.save(str(legacy_path))
    # Verify checkpoint does NOT contain bc_optimizer key.
    blob = torch.load(legacy_path, map_location="cpu", weights_only=True)
    assert "bc_optimizer" not in blob

    # Loading into a trainer with a dedicated bc_optimizer must succeed and
    # leave the bc_optimizer in its initial state (no state restored).
    fresh = _make_cql_with_bc_lr(bc_lr=1e-3)
    fresh.load(str(legacy_path))


def test_dedicated_bc_checkpoint_round_trip(tmp_path: object) -> None:
    """A trainer with ``bc_lr`` set must round-trip its bc_optimizer state."""
    from pathlib import Path

    tmp = Path(str(tmp_path))
    trainer = _make_cql_with_bc_lr(bc_lr=1e-3)
    states = torch.randn(_BATCH, _STATE_DIM)
    actions = torch.randn(_BATCH, _ACTION_DIM)
    trainer.bc_update(states, actions, weight=1.0)

    ckpt = tmp / "with_bc.pt"
    trainer.save(str(ckpt))
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert "bc_optimizer" in blob
    assert blob["bc_lr"] == pytest.approx(1e-3)

    restored = _make_cql_with_bc_lr(bc_lr=1e-3)
    restored.load(str(ckpt))
    # Both bc_optimizer states should now have at least one populated entry.
    assert len(restored.bc_optimizer.state) == len(trainer.bc_optimizer.state)


def test_bc_update_empty_batch_is_noop() -> None:
    """An empty (batch=0) tensor pair short-circuits without an optimizer step."""
    trainer = _make_cql()
    snapshot = {k: v.detach().clone() for k, v in trainer.policy.state_dict().items()}
    empty_states = torch.empty(0, _STATE_DIM)
    empty_actions = torch.empty(0, _ACTION_DIM)

    out = trainer.bc_update(empty_states, empty_actions, weight=1.0)

    assert out == {"bc_loss": 0.0}
    for k, v in trainer.policy.state_dict().items():
        assert torch.equal(v, snapshot[k]), f"policy weights changed on empty batch at {k}"


def test_bc_update_shape_mismatch_raises() -> None:
    """Mismatched batch dimensions must raise rather than silently broadcast."""
    trainer = _make_cql()
    states = torch.randn(_BATCH, _STATE_DIM)
    actions = torch.randn(_BATCH + 1, _ACTION_DIM)

    with pytest.raises((RuntimeError, ValueError)):
        trainer.bc_update(states, actions, weight=1.0)
