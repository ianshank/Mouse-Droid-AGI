"""Tests for Phase 2.2 — MCTS policy warm-start."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mousedroid.cognitive.constitutional_rl import PolicyMLP, ValueMLP


class TestPolicyMLPSaveLoad:
    """Test save/load for PolicyMLP."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        policy = PolicyMLP(input_dim=64, action_dim=3)
        path = tmp_path / "policy.npz"
        policy.save(path)
        assert path.exists()

    def test_roundtrip_preserves_weights(self, tmp_path: Path) -> None:
        policy = PolicyMLP(input_dim=64, action_dim=3)
        path = tmp_path / "policy.npz"
        policy.save(path)

        policy2 = PolicyMLP(input_dim=64, action_dim=3)
        policy2.load(path)

        np.testing.assert_array_equal(policy._w1, policy2._w1)
        np.testing.assert_array_equal(policy._b1, policy2._b1)
        np.testing.assert_array_equal(policy._w2, policy2._w2)
        np.testing.assert_array_equal(policy._b2, policy2._b2)

    def test_loaded_policy_same_output(self, tmp_path: Path) -> None:
        policy = PolicyMLP(input_dim=64, action_dim=3)
        state = np.random.randn(64).astype(np.float32)
        out1 = policy.forward(state)

        path = tmp_path / "policy.npz"
        policy.save(path)

        policy2 = PolicyMLP(input_dim=64, action_dim=3)
        policy2.load(path)
        out2 = policy2.forward(state)

        np.testing.assert_allclose(out1, out2, atol=1e-6)


class TestValueMLPSaveLoad:
    """Test save/load for ValueMLP."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        value_fn = ValueMLP(input_dim=64)
        path = tmp_path / "value.npz"
        value_fn.save(path)
        assert path.exists()

    def test_roundtrip_preserves_weights(self, tmp_path: Path) -> None:
        value_fn = ValueMLP(input_dim=64)
        path = tmp_path / "value.npz"
        value_fn.save(path)

        value_fn2 = ValueMLP(input_dim=64)
        value_fn2.load(path)

        np.testing.assert_array_equal(value_fn._w1, value_fn2._w1)
        np.testing.assert_array_equal(value_fn._b1, value_fn2._b1)

    def test_loaded_value_same_output(self, tmp_path: Path) -> None:
        value_fn = ValueMLP(input_dim=64)
        state = np.random.randn(64).astype(np.float32)
        out1 = value_fn.forward(state)

        path = tmp_path / "value.npz"
        value_fn.save(path)

        value_fn2 = ValueMLP(input_dim=64)
        value_fn2.load(path)
        out2 = value_fn2.forward(state)

        assert abs(out1 - out2) < 1e-6


class TestWarmstartPolicy:
    """Test policy weight initialization from latent statistics."""

    def test_warmstart_produces_valid_policy(self) -> None:
        from training.warmstart_policy import warmstart_policy

        latent_mean = np.random.randn(64).astype(np.float32)
        latent_std = np.abs(np.random.randn(64).astype(np.float32)) + 0.1

        policy = warmstart_policy(latent_mean, latent_std, input_dim=64, action_dim=3)
        state = np.random.randn(64).astype(np.float32)
        action = policy.forward(state)

        assert action.shape == (3,)
        assert np.all(np.abs(action) <= 1.0)  # tanh output


# ---------------------------------------------------------------------------
# tune_ucb — config-driven candidates and episode cap (Phase 3 refactor)
# ---------------------------------------------------------------------------


class TestTuneUcbConfigDriven:
    """tune_ucb() must read ucb_candidates and warmstart_n_episodes from MCTSConfig."""

    def _make_dummy_rssm(self):
        """Return a minimal RSSM-like stub that satisfies tune_ucb's interface."""
        from unittest.mock import MagicMock
        import torch

        rssm = MagicMock()
        rssm._cfg.hidden_dim = 32
        rssm._cfg.latent_dim = 16

        # imagine_step returns (h, z, reward_scalar)
        def fake_imagine(action, h, z):
            new_h = torch.zeros_like(h)
            new_z = torch.zeros_like(z)
            reward = torch.tensor(0.5)
            return new_h, new_z, reward

        rssm.imagine_step.side_effect = fake_imagine
        return rssm

    def _make_mcts_planner(self):
        """Patch MCTSPlanner so tune_ucb doesn't need a real world model."""
        from unittest.mock import patch, MagicMock
        import torch

        planner = MagicMock()
        planner.plan.return_value = torch.zeros(3)
        return planner

    def test_ucb_candidates_from_config(self) -> None:
        """tune_ucb must iterate over base_cfg.ucb_candidates, not a hardcoded list."""
        from unittest.mock import patch, MagicMock
        import torch
        from training.warmstart_policy import tune_ucb
        from mousedroid.config.schema import MCTSConfig

        rssm = self._make_dummy_rssm()
        # Only 2 candidates so the test runs fast
        cfg = MCTSConfig(ucb_candidates=[0.5, 1.5], warmstart_n_episodes=2, n_simulations_max=3)

        with patch("training.warmstart_policy.MCTSPlanner") as MockPlanner:
            mock_planner = self._make_mcts_planner()
            MockPlanner.return_value = mock_planner

            best_ucb, results = tune_ucb(rssm, cfg, n_episodes=2, target_ms=9999.0)

        # Results dict should have exactly the 2 custom candidates
        candidate_keys = [k for k in results if k.startswith("ucb_")]
        assert len(candidate_keys) == 2
        assert "ucb_0.5" in results
        assert "ucb_1.5" in results
        assert best_ucb in (0.5, 1.5)

    def test_warmstart_n_episodes_caps_inner_loop(self) -> None:
        """tune_ucb inner loop must be capped by base_cfg.warmstart_n_episodes."""
        from unittest.mock import patch, MagicMock, call
        import torch
        from training.warmstart_policy import tune_ucb
        from mousedroid.config.schema import MCTSConfig

        rssm = self._make_dummy_rssm()
        cfg = MCTSConfig(ucb_candidates=[1.0], warmstart_n_episodes=3, n_simulations_max=2)

        call_counts = []

        with patch("training.warmstart_policy.MCTSPlanner") as MockPlanner:
            mock_planner = MagicMock()
            mock_planner.plan.return_value = torch.zeros(3)

            def count_and_return(*a, **kw):
                call_counts.append(1)
                return torch.zeros(3)

            mock_planner.plan.side_effect = count_and_return
            MockPlanner.return_value = mock_planner

            # n_episodes=1000 but warmstart_n_episodes=3 → only 3 episodes run
            tune_ucb(rssm, cfg, n_episodes=1000, target_ms=9999.0)

        # 3 episodes × 20 steps per episode = 60 plan() calls
        assert len(call_counts) == 3 * 20

    def test_n_simulations_defaults_to_config(self) -> None:
        """When n_simulations is omitted, n_simulations_max from cfg is used."""
        from unittest.mock import patch, MagicMock
        import torch
        from training.warmstart_policy import tune_ucb
        from mousedroid.config.schema import MCTSConfig

        rssm = self._make_dummy_rssm()
        # Custom max — tune_ucb should forward it to MCTSConfig(n_simulations_max=...)
        cfg = MCTSConfig(ucb_candidates=[1.0], warmstart_n_episodes=1, n_simulations_max=7)

        captured_cfgs: list = []

        def capture_planner(mcts_cfg, *a, **kw):
            captured_cfgs.append(mcts_cfg)
            m = MagicMock()
            m.plan.return_value = torch.zeros(3)
            return m

        with patch("training.warmstart_policy.MCTSPlanner", side_effect=capture_planner):
            tune_ucb(rssm, cfg, n_episodes=1, target_ms=9999.0)

        assert len(captured_cfgs) == 1
        assert captured_cfgs[0].n_simulations_max == 7
