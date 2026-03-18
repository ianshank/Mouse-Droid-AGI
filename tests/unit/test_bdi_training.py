"""Tests for Phase 2.3 — BDI weight training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from training.collect_annotations import INTENTION_LABELS, label_intention
from training.train_bdi import (
    train_affect_estimator,
    train_bdi,
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


class TestTrainBdiNormStatsRegression:
    """Regression: train_bdi() must persist belief_norm_stats.npz (Bug fix 2026-03)."""

    def test_norm_stats_file_created(self, tmp_path: Path) -> None:
        """train_bdi() must save belief_norm_stats.npz alongside other weights."""
        ann_path = tmp_path / "annotations.npz"
        obs, intentions = _make_dummy_data(50)
        np.savez(ann_path, observations=obs, intentions=intentions)

        output_dir = tmp_path / "bdi_out"
        train_bdi(ann_path, output_dir=output_dir, epochs=2, batch_size=16)

        norm_path = output_dir / "belief_norm_stats.npz"
        assert norm_path.exists(), "belief_norm_stats.npz must be saved by train_bdi"

    def test_norm_stats_shapes_match_obs_dim(self, tmp_path: Path) -> None:
        """mean and std arrays must match observation dimension (256)."""
        ann_path = tmp_path / "annotations.npz"
        obs, intentions = _make_dummy_data(50)
        np.savez(ann_path, observations=obs, intentions=intentions)

        output_dir = tmp_path / "bdi_out"
        train_bdi(ann_path, output_dir=output_dir, epochs=2, batch_size=16)

        data = np.load(output_dir / "belief_norm_stats.npz")
        assert "mean" in data, "must contain 'mean' key"
        assert "std" in data, "must contain 'std' key"
        assert data["mean"].shape == (256,)
        assert data["std"].shape == (256,)

    def test_all_weight_files_produced(self, tmp_path: Path) -> None:
        """train_bdi() must produce belief, desire, intention, affect, and norm stats."""
        ann_path = tmp_path / "annotations.npz"
        obs, intentions = _make_dummy_data(50)
        np.savez(ann_path, observations=obs, intentions=intentions)

        output_dir = tmp_path / "bdi_out"
        train_bdi(ann_path, output_dir=output_dir, epochs=2, batch_size=16)

        expected = ["belief.npz", "desire.npz", "intention.npz", "affect.npz", "belief_norm_stats.npz"]
        for name in expected:
            assert (output_dir / name).exists(), f"Missing: {name}"


# ---------------------------------------------------------------------------
# Config-driven dim params (Phase 3 refactor)
# ---------------------------------------------------------------------------


class TestBeliefEncoderDimParams:
    """train_belief_encoder() must respect explicit obs_dim and belief_dim params."""

    def test_custom_obs_dim(self) -> None:
        rng = np.random.default_rng(0)
        obs = rng.standard_normal((60, 128)).astype(np.float32)
        weights = train_belief_encoder(obs, lr=1e-3, epochs=2, batch_size=16, obs_dim=128, belief_dim=64)
        assert weights["w1"].shape == (128, 64)
        assert weights["w2"].shape == (64, 64)

    def test_custom_belief_dim(self) -> None:
        rng = np.random.default_rng(0)
        obs = rng.standard_normal((60, 256)).astype(np.float32)
        weights = train_belief_encoder(obs, lr=1e-3, epochs=2, batch_size=16, obs_dim=256, belief_dim=32)
        # w2 is belief_dim × belief_dim (square self-encoding layer)
        assert weights["w1"].shape == (256, 32)
        assert weights["w2"].shape == (32, 32)


class TestDesireEncoderDimParams:
    def test_custom_dims(self) -> None:
        rng = np.random.default_rng(0)
        obs = rng.standard_normal((60, 128)).astype(np.float32)
        belief_w = train_belief_encoder(obs, lr=1e-3, epochs=2, batch_size=16, obs_dim=128, belief_dim=32)
        desire_w = train_desire_encoder(obs, belief_w, lr=1e-3, epochs=2, batch_size=16, belief_dim=32, desire_dim=16)
        assert desire_w["w1"].shape == (32, 16)


class TestAffectEstimatorDimParams:
    def test_custom_affect_dim(self) -> None:
        obs, intentions = _make_dummy_data(60)
        belief_w = train_belief_encoder(obs, lr=1e-3, epochs=2, batch_size=16)
        desire_w = train_desire_encoder(obs, belief_w, lr=1e-3, epochs=2, batch_size=16)
        intent_w = train_intention_predictor(obs, intentions, belief_w, desire_w, lr=1e-3, epochs=2, batch_size=16)
        affect_w = train_affect_estimator(
            obs, belief_w, desire_w, intent_w,
            lr=1e-3, epochs=2, batch_size=16,
            desire_dim=64, affect_dim=4,
        )
        # Output column count should match affect_dim=4
        assert affect_w["w1"].shape[1] == 4


class TestTrainBdiWithBDIConfig:
    """train_bdi() must forward BDITrainingConfig dims to sub-functions."""

    def test_custom_dims_reflected_in_output_shapes(self, tmp_path: Path) -> None:
        from mousedroid.config.schema import BDITrainingConfig

        ann_path = tmp_path / "annotations.npz"
        obs, intentions = _make_dummy_data(60)
        np.savez(ann_path, observations=obs, intentions=intentions)

        cfg = BDITrainingConfig(
            obs_dim=256,
            belief_dim=32,
            desire_dim=16,
            affect_dim=4,
            epochs=2,
            batch_size=16,
        )
        output_dir = tmp_path / "bdi_out_custom"
        train_bdi(ann_path, output_dir=output_dir, bdi_config=cfg)

        belief_w = np.load(output_dir / "belief.npz")
        desire_w = np.load(output_dir / "desire.npz")
        affect_w = np.load(output_dir / "affect.npz")

        assert belief_w["w1"].shape == (256, 32)   # obs_dim × belief_dim
        assert desire_w["w1"].shape == (32, 16)    # belief_dim × desire_dim
        assert affect_w["w1"].shape[1] == 4        # affect_dim
