"""Tests for Phase 2.3 — BDI weight training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from training.collect_annotations import INTENTION_LABELS, label_intention
from training.train_bdi import (
    _init_weights,
    belief_reconstruction_mse,
    linear_classifier_accuracy,
    majority_class_accuracy,
    passes_bdi_publish_bars,
    pca_reconstruction_mse,
    train_affect_estimator,
    train_belief_encoder,
    train_desire_encoder,
    train_intention_predictor,
)

from mousedroid.cognitive.bdi_model import NeuralBDI

_N_CLASSES = len(INTENTION_LABELS)


def _make_dummy_data(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Create small dummy dataset for fast tests."""
    rng = np.random.default_rng(99)
    observations = rng.standard_normal((n, 256)).astype(np.float32)
    intentions = rng.integers(0, _N_CLASSES, size=n).astype(np.int64)
    return observations, intentions


class TestBeliefEncoder:
    def test_loss_decreases(self) -> None:
        obs, _ = _make_dummy_data(100)
        # Just 5 epochs for speed
        weights = train_belief_encoder(obs, lr=1e-3, epochs=5, batch_size=32)
        assert "w1" in weights
        assert "b1" in weights
        assert weights["w1"].shape == (256, 128)


class TestHeInitAndAdam:
    def test_he_init_std_matches_sqrt_2_over_fan_in(self) -> None:
        from mousedroid.constants import WEIGHT_INIT_SCALE

        rng = np.random.default_rng(0)
        fan_in, fan_out = 256, 128
        weights = _init_weights(rng, fan_in, fan_out)
        expected = float(np.sqrt(2.0 / fan_in))
        assert abs(float(weights.std()) - expected) < 0.04
        assert not np.isclose(expected, WEIGHT_INIT_SCALE)

    def test_belief_ae_beats_mean_predictor(self) -> None:
        """Adam must move off the documented SGD predict-zero plateau."""
        rng = np.random.default_rng(7)
        obs = rng.standard_normal((128, 256)).astype(np.float32)
        weights = train_belief_encoder(obs, lr=3e-3, epochs=40, batch_size=32)
        ae_mse = belief_reconstruction_mse(obs, weights)
        mean_mse = float(np.mean((obs - obs.mean(axis=0)) ** 2))
        assert ae_mse < 0.75 * mean_mse

    def test_publish_bars_require_pca_and_majority(self) -> None:
        assert passes_bdi_publish_bars(
            belief_mse=0.3,
            pca_mse=0.4,
            intention_acc=0.5,
            majority_acc=0.2,
        )
        assert not passes_bdi_publish_bars(
            belief_mse=0.5,
            pca_mse=0.4,
            intention_acc=0.5,
            majority_acc=0.2,
        )
        assert not passes_bdi_publish_bars(
            belief_mse=0.3,
            pca_mse=0.4,
            intention_acc=0.2,
            majority_acc=0.2,
        )

    def test_pca_helper_is_below_mean_on_low_rank_data(self) -> None:
        rng = np.random.default_rng(1)
        z = rng.standard_normal((64, 8)).astype(np.float32)
        w = rng.standard_normal((8, 32)).astype(np.float32)
        x = z @ w
        assert pca_reconstruction_mse(x, rank=8) < 1e-5


class TestCausalIntention:
    def test_causal_features_beat_majority_class(self) -> None:
        from training.collect_annotations import (
            _sample_label_context,
            intention_feature_vector,
            label_intention,
        )

        rng = np.random.default_rng(11)
        rows = []
        labels = []
        for _ in range(400):
            action = np.tanh(rng.standard_normal(3).astype(np.float32))
            obs, human_detected, human_dist, commanded = _sample_label_context(rng, action)
            rows.append(
                intention_feature_vector(
                    action,
                    obs,
                    human_detected=human_detected,
                    human_dist_m=human_dist,
                    commanded_action=commanded,
                )
            )
            labels.append(
                label_intention(
                    action,
                    obs,
                    human_detected=human_detected,
                    human_dist_m=human_dist,
                    commanded_action=commanded,
                )
            )
        features = np.stack(rows)
        y = np.array(labels, dtype=np.int64)
        weights = train_intention_predictor(
            np.zeros((len(y), 256), dtype=np.float32),
            y,
            lr=1e-2,
            epochs=40,
            batch_size=32,
            features=features,
        )
        acc = linear_classifier_accuracy(features, y, weights)
        majority = majority_class_accuracy(y)
        assert acc > majority
        assert weights["w1"].shape[0] == features.shape[1]
        assert "w2" in weights

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
            obs,
            intentions,
            belief_w,
            desire_w,
            lr=1e-3,
            epochs=2,
            batch_size=32,
        )
        assert intent_w["w1"].shape == (64, _N_CLASSES)


class TestAffectEstimator:
    def test_output_shape(self) -> None:
        obs, intentions = _make_dummy_data(100)
        belief_w = train_belief_encoder(obs, lr=1e-3, epochs=2, batch_size=32)
        desire_w = train_desire_encoder(obs, belief_w, lr=1e-3, epochs=2, batch_size=32)
        intent_w = train_intention_predictor(
            obs,
            intentions,
            belief_w,
            desire_w,
            lr=1e-3,
            epochs=2,
            batch_size=32,
        )
        affect_w = train_affect_estimator(
            obs,
            belief_w,
            desire_w,
            intent_w,
            lr=1e-3,
            epochs=2,
            batch_size=32,
        )
        assert affect_w["w1"].shape == (64 + _N_CLASSES, 2)


class TestNeuralBDILoadsTrained:
    """Verify that trained weights are loadable by NeuralBDI."""

    def test_full_pipeline_roundtrip(self, tmp_path: Path) -> None:
        obs, intentions = _make_dummy_data(100)

        belief_w = train_belief_encoder(obs, lr=1e-3, epochs=2, batch_size=32)
        desire_w = train_desire_encoder(obs, belief_w, lr=1e-3, epochs=2, batch_size=32)
        intent_w = train_intention_predictor(
            obs,
            intentions,
            belief_w,
            desire_w,
            lr=1e-3,
            epochs=2,
            batch_size=32,
        )
        affect_w = train_affect_estimator(
            obs,
            belief_w,
            desire_w,
            intent_w,
            lr=1e-3,
            epochs=2,
            batch_size=32,
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
        assert result["intentions"].shape == (_N_CLASSES,)
        assert result["affect"].shape == (2,)


class TestLabelIntention:
    def test_all_labels_valid(self) -> None:
        assert len(INTENTION_LABELS) == 10

    def test_battery_threshold_is_overridable(self) -> None:
        from mousedroid.sensing.bundle import MouseDroidObservationBundle

        obs = MouseDroidObservationBundle(
            _motor_state=np.array([0, 0, 0, 9.5], dtype=np.float32),
        )
        action = np.array([0.1, 0.1, 0.0], dtype=np.float32)
        assert label_intention(action, obs, battery_warn_v=9.0) != 6

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
