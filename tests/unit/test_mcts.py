from __future__ import annotations

import pytest
import torch

from mousedroid.config.schema import MCTSConfig
from mousedroid.world_model.mcts import MCTSPlanner


class MockWorldModel:
    def observe_step(self, observation, prev_action, h, z):
        return h, z, h, 0.0

    def imagine_step(self, action, h, z):
        new_h = h.clone()
        new_z = z.clone()
        reward = torch.zeros(1, 1)
        return new_h, new_z, reward


@pytest.fixture
def cfg() -> MCTSConfig:
    return MCTSConfig(n_simulations_base=4, rollout_depth=2, n_action_candidates=3)


@pytest.fixture
def planner(cfg: MCTSConfig) -> MCTSPlanner:
    return MCTSPlanner(cfg, MockWorldModel())


def test_constructor(planner: MCTSPlanner) -> None:
    assert planner._cfg is not None
    assert planner._world_model is not None


def test_plan_returns_tensor(planner: MCTSPlanner) -> None:
    h = torch.zeros(1, 256)
    z = torch.zeros(1, 64)
    action = planner.plan(h, z)
    assert isinstance(action, torch.Tensor)


def test_plan_action_in_range(planner: MCTSPlanner) -> None:
    h = torch.zeros(1, 256)
    z = torch.zeros(1, 64)
    action = planner.plan(h, z)
    assert (action >= -1.0).all()
    assert (action <= 1.0).all()


def test_plan_action_shape(planner: MCTSPlanner) -> None:
    h = torch.zeros(1, 256)
    z = torch.zeros(1, 64)
    action = planner.plan(h, z)
    # plan returns (1, action_dim)
    assert action.shape[-1] == 3


def test_different_simulation_budgets() -> None:
    for n_sims in [2, 5, 10]:
        cfg = MCTSConfig(n_simulations_base=n_sims, rollout_depth=1, n_action_candidates=3)
        planner = MCTSPlanner(cfg, MockWorldModel())
        h = torch.zeros(1, 256)
        z = torch.zeros(1, 64)
        action = planner.plan(h, z)
        assert (action >= -1.0).all()
        assert (action <= 1.0).all()


def test_plan_no_grad(planner: MCTSPlanner) -> None:
    h = torch.zeros(1, 256)
    z = torch.zeros(1, 64)
    action = planner.plan(h, z)
    assert not action.requires_grad


def test_generate_candidate_actions(planner: MCTSPlanner) -> None:
    actions = planner._generate_candidate_actions(torch.device("cpu"))
    assert actions.shape == (planner._cfg.n_action_candidates, 3)


def test_plan_with_explicit_n_simulations(planner: MCTSPlanner) -> None:
    """plan() uses the n_simulations kwarg when provided."""
    h = torch.zeros(1, 256)
    z = torch.zeros(1, 64)
    action = planner.plan(h, z, n_simulations=8)
    assert isinstance(action, torch.Tensor)
    assert (action >= -1.0).all()
    assert (action <= 1.0).all()


def test_plan_n_simulations_none_uses_config(planner: MCTSPlanner) -> None:
    """plan() falls back to cfg.n_simulations_base when n_simulations is None."""
    h = torch.zeros(1, 256)
    z = torch.zeros(1, 64)
    # Default fixture has n_simulations_base=4; should not raise
    action = planner.plan(h, z, n_simulations=None)
    assert isinstance(action, torch.Tensor)


def test_ucb1_unvisited_returns_inf(planner: MCTSPlanner) -> None:
    from mousedroid.world_model.mcts import _Node

    node = _Node(
        action=torch.zeros(1, 3),
        h=torch.zeros(1, 256),
        z=torch.zeros(1, 64),
        visit_count=0,
    )
    score = planner._ucb1(node, parent_visits=10)
    assert score == float("inf")


def test_candidate_actions_independent_per_dimension() -> None:
    """Candidate actions should have independent values per dimension, not identical."""
    cfg = MCTSConfig(n_simulations_base=4, rollout_depth=2, n_action_candidates=8)
    planner = MCTSPlanner(cfg, MockWorldModel())
    actions = planner._generate_candidate_actions(torch.device("cpu"))
    assert actions.shape == (8, 3)
    # With independent permutations, columns should NOT all be identical.
    # If they were identical (the old bug), every row would have the same value
    # across all dims, i.e., actions[:, 0] == actions[:, 1] for all rows.
    all_same = torch.all(actions[:, 0] == actions[:, 1]).item()
    assert not all_same, "Candidate actions have identical values across dimensions (old bug)"


def test_candidate_actions_cover_range() -> None:
    """Candidate actions should cover [-1, 1] range in each dimension."""
    cfg = MCTSConfig(n_simulations_base=4, rollout_depth=2, n_action_candidates=16)
    planner = MCTSPlanner(cfg, MockWorldModel())
    actions = planner._generate_candidate_actions(torch.device("cpu"))
    # Each column should contain values from linspace(-1, 1, 16), just permuted
    for dim in range(3):
        col = actions[:, dim].sort().values
        expected = torch.linspace(-1.0, 1.0, 16)
        assert torch.allclose(col, expected), f"Dimension {dim} doesn't cover full range"
