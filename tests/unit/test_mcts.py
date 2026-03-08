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
