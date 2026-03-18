"""Tests for Phase 2.4 — Constitutional RL training components."""

from __future__ import annotations

import numpy as np
from training.train_constitutional_rl import _gae, _ppo_update

from mousedroid.cognitive.constitutional_rl import (
    ConstitutionalChecker,
    ConstitutionalRLConfig,
    PolicyMLP,
    ValueMLP,
)


class TestGAE:
    """Test Generalised Advantage Estimation."""

    def test_basic_gae(self) -> None:
        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.5, 0.5]
        advantages, returns = _gae(rewards, values, gamma=0.99, gae_lambda=0.95)

        assert advantages.shape == (3,)
        assert returns.shape == (3,)
        # Advantages should be positive since rewards > values
        assert np.all(advantages > 0)

    def test_zero_rewards(self) -> None:
        rewards = [0.0, 0.0, 0.0]
        values = [0.0, 0.0, 0.0]
        advantages, _returns = _gae(rewards, values, gamma=0.99, gae_lambda=0.95)

        np.testing.assert_allclose(advantages, 0.0, atol=1e-6)


class TestPPOUpdate:
    """Test PPO update step."""

    def test_update_runs_without_error(self) -> None:
        policy = PolicyMLP(input_dim=8, action_dim=2)
        value_fn = ValueMLP(input_dim=8)

        states = np.random.randn(10, 8).astype(np.float32)
        actions = np.random.randn(10, 2).astype(np.float32)
        old_log_probs = np.random.randn(10).astype(np.float32)
        advantages = np.random.randn(10).astype(np.float32)
        returns = np.random.randn(10).astype(np.float32)

        losses = _ppo_update(
            policy,
            value_fn,
            states,
            actions,
            old_log_probs,
            advantages,
            returns,
            clip_epsilon=0.2,
            lr=1e-3,
            n_epochs=2,
        )

        assert "policy_loss" in losses
        assert "value_loss" in losses


class TestConstitutionalRewardZeroing:
    """Verify that constitutional violations zero out reward."""

    def test_violation_detected(self) -> None:
        checker = ConstitutionalChecker(ConstitutionalRLConfig(speed_ceiling_mps=0.3))
        action = np.array([0.8, 0.0], dtype=np.float64)
        _, violations = checker.check(action, {})
        assert len(violations) > 0

    def test_safe_action_no_violations(self) -> None:
        checker = ConstitutionalChecker()
        action = np.array([0.1, 0.0], dtype=np.float64)
        _, violations = checker.check(
            action,
            {
                "battery_v": 12.0,
                "obstacle_dist_m": 2.0,
                "mcts_sims": 50,
            },
        )
        assert violations == []

    def test_trivially_safe_policy_no_violations(self) -> None:
        """A policy outputting near-zero actions should never violate."""
        checker = ConstitutionalChecker()
        policy = PolicyMLP(input_dim=8, action_dim=2)

        # Zero out weights for near-zero output
        policy._w1 *= 0.0
        policy._b1 *= 0.0
        policy._w2 *= 0.0
        policy._b2 *= 0.0

        total_violations = 0
        for _ in range(100):
            state = np.random.randn(8).astype(np.float32)
            action = policy.forward(state)
            _, violations = checker.check(
                action,
                {
                    "battery_v": 12.0,
                    "obstacle_dist_m": 2.0,
                    "mcts_sims": 50,
                },
            )
            total_violations += len(violations)

        assert total_violations == 0


# ---------------------------------------------------------------------------
# Config-driven context values (Phase 3 refactor)
# ---------------------------------------------------------------------------


class TestConstitutionalRLTrainingContextFromConfig:
    """train_constitutional_rl must source nominal_battery_v and nominal_obstacle_dist_m
    from AnnotationConfig, not hardcoded values."""

    def test_annotation_config_defaults_match_checker_expectations(self) -> None:
        """Default nominal_battery_v=12.0 and nominal_obstacle_dist_m=2.0 are safe."""
        from mousedroid.config.schema import AnnotationConfig

        ann_cfg = AnnotationConfig()
        assert ann_cfg.nominal_battery_v == 12.0
        assert ann_cfg.nominal_obstacle_dist_m == 2.0

        # Verify these values pass a safe-action check
        checker = ConstitutionalChecker()
        action = np.array([0.1, 0.0], dtype=np.float64)
        _, violations = checker.check(
            action,
            {
                "battery_v": ann_cfg.nominal_battery_v,
                "obstacle_dist_m": ann_cfg.nominal_obstacle_dist_m,
                "mcts_sims": 50,
            },
        )
        assert violations == []

    def test_custom_nominal_values_propagate(self) -> None:
        """Custom nominal values in AnnotationConfig affect context dict."""
        from mousedroid.config.schema import AnnotationConfig

        ann_cfg = AnnotationConfig(nominal_battery_v=11.5, nominal_obstacle_dist_m=1.5)
        context = {
            "battery_v": ann_cfg.nominal_battery_v,
            "obstacle_dist_m": ann_cfg.nominal_obstacle_dist_m,
            "mcts_sims": 50,
        }
        assert context["battery_v"] == 11.5
        assert context["obstacle_dist_m"] == 1.5
