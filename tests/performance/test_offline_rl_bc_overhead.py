"""Phase 2.1 — performance budget regression for the BC auxiliary loss.

The BC update adds one forward + backward pass per training step. This test
quantifies that overhead on a tiny CPU-only trainer and asserts it stays
within a configurable budget. CI defaults to ``BC_OVERHEAD_BUDGET=2.5`` —
i.e. BC-active training may take up to 2.5x the wall-clock of BC-disabled.

Marked ``slow`` so the default ``pytest`` invocation skips it; ``scripts/ci.sh``
performance stage picks it up. The budget is operator-tunable via the
``MOUSEDROID_BC_OVERHEAD_BUDGET`` env var so Jetson runs (slower per-step) can
relax the bound without touching code.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

from mousedroid.learning.offline_rl import CQLTrainer

_STATE_DIM = 8
_ACTION_DIM = 3
_BATCH = 32
_HIDDEN = 64
_STEPS = 50
_DEFAULT_BUDGET = 2.5
_BUDGET_ENV = "MOUSEDROID_BC_OVERHEAD_BUDGET"


def _resolve_budget() -> float:
    """Return the operator-tuned BC overhead budget multiplier."""
    raw = os.environ.get(_BUDGET_ENV)
    if raw is None:
        return _DEFAULT_BUDGET
    parsed = float(raw)
    if parsed <= 1.0:
        msg = f"{_BUDGET_ENV} must be > 1.0 (got {parsed!r})"
        raise ValueError(msg)
    return parsed


def _build_trainer(*, bc_lr: float | None) -> CQLTrainer:
    return CQLTrainer(
        state_dim=_STATE_DIM,
        action_dim=_ACTION_DIM,
        hidden_dim=_HIDDEN,
        device=torch.device("cpu"),
        bc_lr=bc_lr,
    )


@pytest.mark.slow
def test_bc_active_wall_clock_within_budget() -> None:
    """BC-active training step must stay within the configured overhead budget."""
    torch.manual_seed(0)
    states = torch.randn(_BATCH, _STATE_DIM)
    actions = torch.randn(_BATCH, _ACTION_DIM)
    rewards = torch.randn(_BATCH)
    next_states = torch.randn(_BATCH, _STATE_DIM)
    dones = torch.zeros(_BATCH)

    baseline = _build_trainer(bc_lr=None)
    bc_on = _build_trainer(bc_lr=1e-3)

    # Warmup once to avoid first-call JIT/Adam initialization skew.
    baseline.update_step(states, actions, rewards, next_states, dones)
    bc_on.update_step(states, actions, rewards, next_states, dones)
    bc_on.bc_update(states, actions, weight=1.0)

    # Baseline: update_step only
    t0 = time.perf_counter()
    for _ in range(_STEPS):
        baseline.update_step(states, actions, rewards, next_states, dones)
    baseline_elapsed = time.perf_counter() - t0

    # BC-active: update_step + bc_update on the same batch
    t0 = time.perf_counter()
    for _ in range(_STEPS):
        bc_on.update_step(states, actions, rewards, next_states, dones)
        bc_on.bc_update(states, actions, weight=1.0)
    bc_on_elapsed = time.perf_counter() - t0

    budget = _resolve_budget()
    ratio = bc_on_elapsed / max(baseline_elapsed, 1e-6)
    assert ratio <= budget, (
        f"BC-active training is {ratio:.2f}x baseline (budget {budget:.2f}x). "
        f"baseline={baseline_elapsed:.3f}s, bc_on={bc_on_elapsed:.3f}s. "
        f"Tune via env {_BUDGET_ENV} if Jetson hardware is slower."
    )


@pytest.mark.slow
def test_dedicated_bc_optimizer_state_size_bounded() -> None:
    """The dedicated bc_optimizer state must not exceed Adam's known footprint
    (two tensors per param: exp_avg + exp_avg_sq)."""
    trainer = _build_trainer(bc_lr=1e-3)
    states = torch.randn(_BATCH, _STATE_DIM)
    actions = torch.randn(_BATCH, _ACTION_DIM)

    # Step once to populate Adam internal state.
    trainer.bc_update(states, actions, weight=1.0)

    policy_param_count = sum(1 for _ in trainer.policy.parameters())
    bc_state_entries = len(trainer.bc_optimizer.state)
    assert bc_state_entries == policy_param_count, (
        f"Adam state entries ({bc_state_entries}) != policy params "
        f"({policy_param_count}) — memory accounting drift."
    )
