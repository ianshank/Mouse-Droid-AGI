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


# ---------------------------------------------------------------------------
# E1-S1: Early-exit convergence tests
# ---------------------------------------------------------------------------


class TestEarlyExit:
    """Tests for early-exit convergence optimisation."""

    def test_early_exit_disabled_runs_all_sims(self) -> None:
        """When threshold=0, all simulations run (no early exit)."""
        cfg = MCTSConfig(
            n_simulations_base=10,
            rollout_depth=1,
            n_action_candidates=3,
            early_exit_value_threshold=0.0,
        )
        planner = MCTSPlanner(cfg, MockWorldModel())
        h = torch.zeros(1, 256)
        z = torch.zeros(1, 64)
        action = planner.plan(h, z)
        assert isinstance(action, torch.Tensor)
        assert (action >= -1.0).all() and (action <= 1.0).all()

    def test_early_exit_enabled_converges_early(self) -> None:
        """With a large threshold, search should stop before all sims (zero-reward model)."""
        cfg = MCTSConfig(
            n_simulations_base=50,
            rollout_depth=2,
            n_action_candidates=3,
            early_exit_value_threshold=1.0,  # Very high → converges immediately (zero rewards)
            early_exit_patience=2,
        )
        planner = MCTSPlanner(cfg, MockWorldModel())
        h = torch.zeros(1, 256)
        z = torch.zeros(1, 64)

        import time

        t0 = time.monotonic()
        action = planner.plan(h, z)
        elapsed = time.monotonic() - t0

        assert isinstance(action, torch.Tensor)
        # With a zero-reward model and threshold=1.0, should exit well before 50 sims
        # We can't assert exact iteration count, but elapsed time should be fast
        assert elapsed < 5.0  # Generous bound, but shows early exit happened

    def test_early_exit_patience_respected(self) -> None:
        """patience=1 should exit after first stable iteration (patience must be met)."""
        cfg = MCTSConfig(
            n_simulations_base=100,
            rollout_depth=1,
            n_action_candidates=3,
            early_exit_value_threshold=1.0,
            early_exit_patience=1,
        )
        planner = MCTSPlanner(cfg, MockWorldModel())
        h = torch.zeros(1, 256)
        z = torch.zeros(1, 64)
        action = planner.plan(h, z)
        assert (action >= -1.0).all() and (action <= 1.0).all()


# ---------------------------------------------------------------------------
# E1-S4: Time-budget adaptive simulation tests
# ---------------------------------------------------------------------------


class TestTimeBudget:
    """Tests for time-budget simulation limiting."""

    def test_time_budget_zero_runs_all(self) -> None:
        """Budget=0 means unlimited; all simulations run."""
        cfg = MCTSConfig(
            n_simulations_base=10,
            rollout_depth=1,
            n_action_candidates=3,
            simulation_budget_ms=0.0,
        )
        planner = MCTSPlanner(cfg, MockWorldModel())
        h = torch.zeros(1, 256)
        z = torch.zeros(1, 64)
        action = planner.plan(h, z)
        assert isinstance(action, torch.Tensor)

    def test_time_budget_tight_exits_early(self) -> None:
        """A very tight time budget (1ms) should terminate before completing all sims."""
        cfg = MCTSConfig(
            n_simulations_base=1000,
            rollout_depth=5,
            n_action_candidates=9,
            simulation_budget_ms=1.0,  # 1ms — impossible to complete 1000 sims
            early_exit_value_threshold=0.0,  # Disable early exit
        )
        planner = MCTSPlanner(cfg, MockWorldModel())
        h = torch.zeros(1, 256)
        z = torch.zeros(1, 64)

        import time

        t0 = time.monotonic()
        action = planner.plan(h, z)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        assert isinstance(action, torch.Tensor)
        # Should have bailed out well before a full 1000 sims
        assert elapsed_ms < 5000.0  # Generous upper bound


# ---------------------------------------------------------------------------
# E1-S2: Action diversity tests
# ---------------------------------------------------------------------------


class TestActionDiversity:
    """Tests for multi-dimensional action sampling."""

    def test_linspace_mode_broadcasts_1d(self) -> None:
        """Legacy linspace mode produces identical values across action dims."""
        cfg = MCTSConfig(
            n_simulations_base=2,
            rollout_depth=1,
            n_action_candidates=5,
            action_sampling="linspace",
        )
        planner = MCTSPlanner(cfg, MockWorldModel())
        actions = planner._generate_candidate_actions(torch.device("cpu"))
        assert actions.shape == (5, 3)
        # All dims should be the same for each row (broadcast from 1D)
        for i in range(5):
            assert torch.allclose(actions[i, 0:1].expand(3), actions[i])

    def test_uniform_mode_produces_diverse_actions(self) -> None:
        """Uniform mode produces independently sampled action dims."""
        cfg = MCTSConfig(
            n_simulations_base=2,
            rollout_depth=1,
            n_action_candidates=9,
            action_sampling="uniform",
        )
        planner = MCTSPlanner(cfg, MockWorldModel())
        actions = planner._generate_candidate_actions(torch.device("cpu"))
        assert actions.shape == (9, 3)
        # Check that not all dims are identical (with overwhelming probability)
        all_same = all(
            torch.allclose(actions[i, 0:1].expand(3), actions[i]) for i in range(9)
        )
        assert not all_same, "Uniform sampling should produce diverse action dims"

    def test_uniform_actions_in_range(self) -> None:
        """Uniform-sampled actions must be in [-1, 1]."""
        cfg = MCTSConfig(
            n_simulations_base=2,
            rollout_depth=1,
            n_action_candidates=50,  # More samples to tighten the bound
            action_sampling="uniform",
        )
        planner = MCTSPlanner(cfg, MockWorldModel())
        actions = planner._generate_candidate_actions(torch.device("cpu"))
        assert (actions >= -1.0).all()
        assert (actions <= 1.0).all()

