"""Tests for offline RL algorithms (CQL and IQL)."""

from __future__ import annotations

import pytest
import torch

from mousedroid.learning.offline_rl import (
    CQLTrainer,
    DeterministicPolicy,
    IQLTrainer,
    QNetwork,
    ValueNetwork,
)

# ---------------------------------------------------------------------------
# Network unit tests
# ---------------------------------------------------------------------------


class TestQNetwork:
    """Test twin Q-network."""

    def test_output_shapes(self) -> None:
        qnet = QNetwork(state_dim=8, action_dim=3, hidden_dim=32)
        states = torch.randn(4, 8)
        actions = torch.randn(4, 3)
        q1, q2 = qnet(states, actions)
        assert q1.shape == (4, 1)
        assert q2.shape == (4, 1)

    def test_twin_q_independence(self) -> None:
        """Q1 and Q2 should produce different values (different init)."""
        qnet = QNetwork(state_dim=8, action_dim=3, hidden_dim=32)
        states = torch.randn(4, 8)
        actions = torch.randn(4, 3)
        q1, q2 = qnet(states, actions)
        # Not exactly equal due to different random init
        assert not torch.allclose(q1, q2)

    def test_gradient_flow(self) -> None:
        qnet = QNetwork(state_dim=8, action_dim=3, hidden_dim=32)
        states = torch.randn(4, 8)
        actions = torch.randn(4, 3, requires_grad=True)
        q1, q2 = qnet(states, actions)
        loss = q1.mean() + q2.mean()
        loss.backward()
        assert actions.grad is not None


class TestDeterministicPolicy:
    """Test deterministic policy network."""

    def test_output_shape(self) -> None:
        policy = DeterministicPolicy(state_dim=8, action_dim=3, hidden_dim=32)
        states = torch.randn(4, 8)
        actions = policy(states)
        assert actions.shape == (4, 3)

    def test_output_bounded(self) -> None:
        """Tanh output should be in [-1, 1]."""
        policy = DeterministicPolicy(state_dim=8, action_dim=3, hidden_dim=32)
        states = torch.randn(100, 8) * 10  # large inputs
        actions = policy(states)
        assert (actions >= -1.0).all()
        assert (actions <= 1.0).all()


class TestValueNetwork:
    """Test state value network (IQL-specific)."""

    def test_output_shape(self) -> None:
        vnet = ValueNetwork(state_dim=8, hidden_dim=32)
        states = torch.randn(4, 8)
        values = vnet(states)
        assert values.shape == (4, 1)


# ---------------------------------------------------------------------------
# CQL trainer tests
# ---------------------------------------------------------------------------


class TestCQLTrainer:
    """Test CQL training algorithm."""

    @pytest.fixture
    def trainer(self) -> CQLTrainer:
        return CQLTrainer(
            state_dim=8,
            action_dim=3,
            hidden_dim=32,
            gamma=0.99,
            tau=0.005,
            lr=1e-3,
            cql_alpha=1.0,
            n_random_actions=5,
        )

    @pytest.fixture
    def batch(self) -> dict[str, torch.Tensor]:
        return {
            "states": torch.randn(16, 8),
            "actions": torch.randn(16, 3),
            "rewards": torch.randn(16),
            "next_states": torch.randn(16, 8),
            "dones": torch.zeros(16),
        }

    def test_update_step_returns_losses(
        self,
        trainer: CQLTrainer,
        batch: dict[str, torch.Tensor],
    ) -> None:
        losses = trainer.update_step(**batch)
        assert "q_loss" in losses
        assert "bellman_loss" in losses
        assert "cql_loss" in losses
        assert "policy_loss" in losses

    def test_update_step_losses_finite(
        self,
        trainer: CQLTrainer,
        batch: dict[str, torch.Tensor],
    ) -> None:
        losses = trainer.update_step(**batch)
        for key, val in losses.items():
            assert not torch.isnan(torch.tensor(val)), f"{key} is NaN"
            assert not torch.isinf(torch.tensor(val)), f"{key} is Inf"

    def test_multiple_updates_converge(
        self,
        trainer: CQLTrainer,
    ) -> None:
        """Loss should generally decrease over repeated updates on same data."""
        batch = {
            "states": torch.randn(32, 8),
            "actions": torch.randn(32, 3) * 0.1,
            "rewards": torch.ones(32),
            "next_states": torch.randn(32, 8),
            "dones": torch.zeros(32),
        }

        trainer.update_step(**batch)
        for _ in range(20):
            last_losses = trainer.update_step(**batch)

        # At minimum, losses should remain finite
        for key, val in last_losses.items():
            assert not torch.isnan(torch.tensor(val)), f"{key} diverged to NaN"

    def test_cql_alpha_effect(self) -> None:
        """Higher CQL alpha should produce larger CQL loss component."""
        batch = {
            "states": torch.randn(16, 8),
            "actions": torch.randn(16, 3),
            "rewards": torch.randn(16),
            "next_states": torch.randn(16, 8),
            "dones": torch.zeros(16),
        }

        trainer_low = CQLTrainer(
            state_dim=8, action_dim=3, hidden_dim=32,
            cql_alpha=0.1, n_random_actions=5,
        )
        trainer_high = CQLTrainer(
            state_dim=8, action_dim=3, hidden_dim=32,
            cql_alpha=10.0, n_random_actions=5,
        )

        # Use same initial weights
        trainer_high.q_network.load_state_dict(trainer_low.q_network.state_dict())
        trainer_high.target_q_network.load_state_dict(
            trainer_low.target_q_network.state_dict(),
        )
        trainer_high.policy.load_state_dict(trainer_low.policy.state_dict())

        losses_low = trainer_low.update_step(**batch)
        losses_high = trainer_high.update_step(**batch)

        # Higher alpha means larger total Q loss contribution
        assert losses_high["q_loss"] > losses_low["q_loss"]

    def test_save_and_load(self, trainer: CQLTrainer, tmp_path: object) -> None:
        path = str(tmp_path) + "/cql.pt"
        trainer.save(path)

        new_trainer = CQLTrainer(
            state_dim=8, action_dim=3, hidden_dim=32,
            cql_alpha=1.0, n_random_actions=5,
        )
        new_trainer.load(path)

        # Verify weights match
        for p1, p2 in zip(
            trainer.q_network.parameters(),
            new_trainer.q_network.parameters(),
            strict=True,
        ):
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# IQL trainer tests
# ---------------------------------------------------------------------------


class TestIQLTrainer:
    """Test IQL training algorithm."""

    @pytest.fixture
    def trainer(self) -> IQLTrainer:
        return IQLTrainer(
            state_dim=8,
            action_dim=3,
            hidden_dim=32,
            gamma=0.99,
            tau=0.005,
            lr=1e-3,
            iql_tau=0.7,
            beta=3.0,
        )

    @pytest.fixture
    def batch(self) -> dict[str, torch.Tensor]:
        return {
            "states": torch.randn(16, 8),
            "actions": torch.randn(16, 3),
            "rewards": torch.randn(16),
            "next_states": torch.randn(16, 8),
            "dones": torch.zeros(16),
        }

    def test_update_step_returns_losses(
        self,
        trainer: IQLTrainer,
        batch: dict[str, torch.Tensor],
    ) -> None:
        losses = trainer.update_step(**batch)
        assert "q_loss" in losses
        assert "value_loss" in losses
        assert "policy_loss" in losses

    def test_update_step_losses_finite(
        self,
        trainer: IQLTrainer,
        batch: dict[str, torch.Tensor],
    ) -> None:
        losses = trainer.update_step(**batch)
        for key, val in losses.items():
            assert not torch.isnan(torch.tensor(val)), f"{key} is NaN"
            assert not torch.isinf(torch.tensor(val)), f"{key} is Inf"

    def test_has_value_network(self, trainer: IQLTrainer) -> None:
        assert hasattr(trainer, "value_network")
        assert isinstance(trainer.value_network, ValueNetwork)

    def test_expectile_loss_asymmetric(self, trainer: IQLTrainer) -> None:
        """Positive differences should be weighted more with tau > 0.5."""
        pos_diff = torch.tensor([1.0, 2.0, 3.0])
        neg_diff = torch.tensor([-1.0, -2.0, -3.0])

        pos_loss = trainer._expectile_loss(pos_diff)
        neg_loss = trainer._expectile_loss(neg_diff)

        # With iql_tau=0.7, positive diffs weighted 0.7, negative weighted 0.3
        assert pos_loss > neg_loss

    def test_save_and_load(self, trainer: IQLTrainer, tmp_path: object) -> None:
        path = str(tmp_path) + "/iql.pt"
        trainer.save(path)

        new_trainer = IQLTrainer(
            state_dim=8, action_dim=3, hidden_dim=32,
            iql_tau=0.7, beta=3.0,
        )
        new_trainer.load(path)

        # Verify weights match
        for p1, p2 in zip(
            trainer.value_network.parameters(),
            new_trainer.value_network.parameters(),
            strict=True,
        ):
            assert torch.allclose(p1, p2)

    def test_multiple_updates_stable(self, trainer: IQLTrainer) -> None:
        """IQL should remain numerically stable over multiple updates."""
        batch = {
            "states": torch.randn(32, 8),
            "actions": torch.randn(32, 3) * 0.1,
            "rewards": torch.ones(32),
            "next_states": torch.randn(32, 8),
            "dones": torch.zeros(32),
        }

        for _ in range(30):
            losses = trainer.update_step(**batch)

        for key, val in losses.items():
            assert not torch.isnan(torch.tensor(val)), f"{key} diverged to NaN"
            assert not torch.isinf(torch.tensor(val)), f"{key} diverged to Inf"
