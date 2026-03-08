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
