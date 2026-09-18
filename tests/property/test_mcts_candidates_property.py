"""Hypothesis property tests for MCTS candidate-action generation.

The example-based unit tier spot-checks the default 9x3 candidate set. These
fuzz the invariants that must hold for *any* ``(n_action_candidates,
action_dim)`` an operator can configure, since both are free schema fields:

- ``per_axis`` never emits a duplicate candidate (a duplicate double-weights
  that action in the tree policy and wastes a simulation budget slot);
- ``per_axis`` always yields exactly ``n_action_candidates`` rows, in bounds;
- ``per_axis`` always keeps the stop action, whatever the truncation;
- ``shared_axis`` stays rank-deficient, which is the documented legacy
  behaviour the default must preserve.
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import MCTSConfig
from mousedroid.constants import DEFAULT_ACTION_LIMIT
from mousedroid.world_model.mcts import MCTSPlanner

_CPU = torch.device("cpu")


class _StubWorldModel:
    """Identity latent dynamics — candidate generation never consults it."""

    def imagine_step(
        self, action: torch.Tensor, h: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return h, z, torch.zeros(1)


def _candidates(strategy: str, n: int, action_dim: int) -> torch.Tensor:
    cfg = MCTSConfig(n_action_candidates=n, action_candidate_strategy=strategy)  # type: ignore[arg-type]
    planner = MCTSPlanner(cfg, _StubWorldModel(), action_dim=action_dim)
    return planner._generate_candidate_actions(_CPU)


@given(
    n=st.integers(min_value=1, max_value=48),
    action_dim=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=60, deadline=None)
def test_per_axis_shape_and_bounds_hold_for_any_config(n: int, action_dim: int) -> None:
    """Exactly ``n`` rows of width ``action_dim``, all within the limit."""
    candidates = _candidates("per_axis", n, action_dim)
    assert candidates.shape == (n, action_dim)
    assert bool((candidates.abs() <= DEFAULT_ACTION_LIMIT + 1e-6).all())
    assert bool(torch.isfinite(candidates).all())


@given(
    n=st.integers(min_value=1, max_value=48),
    action_dim=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=60, deadline=None)
def test_per_axis_never_emits_duplicates(n: int, action_dim: int) -> None:
    """De-duplication holds across the whole configurable surface."""
    candidates = _candidates("per_axis", n, action_dim)
    assert torch.unique(candidates, dim=0).shape[0] == n


@given(
    n=st.integers(min_value=1, max_value=48),
    action_dim=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=60, deadline=None)
def test_per_axis_always_keeps_the_stop_action(n: int, action_dim: int) -> None:
    """Stop is ordered first, so no truncation can remove it."""
    candidates = _candidates("per_axis", n, action_dim)
    assert bool((candidates.abs().sum(dim=1) == 0).any())


@given(
    n=st.integers(min_value=1, max_value=48),
    action_dim=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=60, deadline=None)
def test_per_axis_is_reproducible(n: int, action_dim: int) -> None:
    """Two independently built planners agree exactly."""
    assert torch.equal(
        _candidates("per_axis", n, action_dim), _candidates("per_axis", n, action_dim)
    )


@given(
    n=st.integers(min_value=2, max_value=48),
    action_dim=st.integers(min_value=2, max_value=6),
)
@settings(max_examples=40, deadline=None)
def test_shared_axis_stays_rank_one(n: int, action_dim: int) -> None:
    """Characterisation pin: the legacy default is rank-deficient by design."""
    candidates = _candidates("shared_axis", n, action_dim)
    assert int(torch.linalg.matrix_rank(candidates)) == 1
