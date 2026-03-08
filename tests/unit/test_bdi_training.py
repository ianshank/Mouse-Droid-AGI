"""Tests for Phase 2.3 — BDI weight training."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mousedroid.cognitive.bdi_model import NeuralBDI
from training.collect_annotations import INTENTION_LABELS, label_intention
from training.train_bdi import (
    train_affect_estimator,
    train_belief_encoder,
    train_desire_encoder,
    train_intention_predictor,
)


def _make_dummy_data(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Create small dummy dataset for fast tests."""
    rng = np.random.default_rng(99)
    observations = rng.standard_normal((n, 256)).astype(np.float32)
    intentions = rng.integers(0, 8, size=n).astype(np.int64)
    return observations, intentions


class TestBeliefEncoder:
    def test_loss_decreases(self) -> None:
        obs, _ = _make_dummy_data(100)
        # Just 5 epochs for speed
        weights = train_belief_encoder(obs, lr=1e-3, epochs=5, batch_size=32)
        assert "w1" in weights
        assert "b1" in weights
        assert weights["w1"].shape == (256, 128)

    def test_saves_loadable_npz(self, tmp_path: Path) -> None:
        obs, _ = _make_dummy_data(100)
        weights = train_belief_encoder(obs, lr=1e-3, epochs=2, batch_size=32)
        path = tmp_path / "belief.npz"
        np.savez(path, **weights)

        loaded = np.load(path)
        np.testing.assert_array_equal(weights["w1"], loaded["w1"])


class TestDesireEncoder:
    def test_output_shape(self) -> None:
        obs, _ = _make_dummy_data(100)
        belief_w = train_belief_encoder(obs, lr=1e-3, epochs=2, batch_size=32)
        desire_w = train_desire_encoder(obs, belief_w, lr=1e-3, epochs=2, batch_size=32)
        assert desire_w["w1"].shape == (128, 64)


class TestIntentionPredictor:
    def test_output_shape(self) -> None:
        obs, intentions = _make_dummy_data(100)
        belief_w = train_belief_encoder(obs, lr=1e-3, epochs=2, batch_size=32)
        desire_w = train_desire_encoder(obs, belief_w, lr=1e-3, epochs=2, batch_size=32)
        intent_w = train_intention_predictor(
            obs, intentions, belief_w, desire_w, lr=1e-3, epochs=2, batch_size=32,
        )
        assert intent_w["w1"].shape == (64, 8)


class TestAffectEstimator:
    def test_output_shape(self) -> None:
        obs, intentions = _make_dummy_data(100)
        belief_w = train_belief_encoder(obs, lr=1e-3, epochs=2, batch_size=32)
        desire_w = train_desire_encoder(obs, belief_w, lr=1e-3, epochs=2, batch_size=32)
        intent_w = train_intention_predictor(
            obs, intentions, belief_w, desire_w, lr=1e-3, epochs=2, batch_size=32,
        )
        affect_w = train_affect_estimator(
            obs, belief_w, desire_w, intent_w, lr=1e-3, epochs=2, batch_size=32,
        )
        assert affect_w["w1"].shape == (72, 2)


class TestNeuralBDILoadsTrained:
    """Verify that trained weights are loadable by NeuralBDI."""

    def test_full_pipeline_roundtrip(self, tmp_path: Path) -> None:
        obs, intentions = _make_dummy_data(100)

        belief_w = train_belief_encoder(obs, lr=1e-3, epochs=2, batch_size=32)
        desire_w = train_desire_encoder(obs, belief_w, lr=1e-3, epochs=2, batch_size=32)
        intent_w = train_intention_predictor(
            obs, intentions, belief_w, desire_w, lr=1e-3, epochs=2, batch_size=32,
        )
        affect_w = train_affect_estimator(
            obs, belief_w, desire_w, intent_w, lr=1e-3, epochs=2, batch_size=32,
        )

        # Save in the format NeuralBDI expects
        np.savez(tmp_path / "belief.npz", **belief_w)
        np.savez(tmp_path / "desire.npz", **desire_w)
        np.savez(tmp_path / "intention.npz", **intent_w)
        np.savez(tmp_path / "affect.npz", **affect_w)

        # NeuralBDI should load without error
        bdi = NeuralBDI(weights_dir=tmp_path)
        result = bdi.infer(obs[0])

        assert "belief" in result
        assert "desire" in result
        assert "intentions" in result
        assert "affect" in result
        assert result["belief"].shape == (128,)
        assert result["desire"].shape == (64,)
        assert result["intentions"].shape == (8,)
        assert result["affect"].shape == (2,)


class TestLabelIntention:
    def test_all_labels_valid(self) -> None:
        assert len(INTENTION_LABELS) == 8

    def test_low_battery_returns_charge(self) -> None:
        from mousedroid.sensing.bundle import MouseDroidObservationBundle

        obs = MouseDroidObservationBundle(
            _motor_state=np.array([0, 0, 0, 9.0], dtype=np.float32),
        )
        action = np.array([0.1, 0.1, 0.0], dtype=np.float32)
        assert label_intention(action, obs) == 6  # charge

    def test_close_obstacle_returns_avoid(self) -> None:
        from mousedroid.sensing.bundle import MouseDroidObservationBundle

        obs = MouseDroidObservationBundle(
            _distance_m=0.1,
            _motor_state=np.array([0, 0, 0, 12.0], dtype=np.float32),
        )
        action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        assert label_intention(action, obs) == 2  # avoid_obstacle
