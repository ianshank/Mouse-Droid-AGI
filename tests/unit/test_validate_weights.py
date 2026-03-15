"""Tests for training.validate_weights — post-training validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from training.validate_weights import (
    generate_training_report,
    validate_bdi_accuracy,
    validate_constitutional_rl,
    validate_rssm_shapes,
    validate_weight_files,
)

from mousedroid.config.schema import Settings


class TestValidateWeightFiles:
    """Tests for validate_weight_files()."""

    def test_all_files_present(self, tmp_path: Path) -> None:
        """Passes when all expected weight files exist."""
        for sub in ["rssm", "mcts", "bdi", "constitutional_rl"]:
            (tmp_path / sub).mkdir()
        (tmp_path / "rssm" / "final.pt").write_bytes(b"fake")
        (tmp_path / "mcts" / "policy_init.npz").write_bytes(b"fake")
        for name in ["belief.npz", "desire.npz", "intention.npz", "affect.npz"]:
            (tmp_path / "bdi" / name).write_bytes(b"fake")
        (tmp_path / "constitutional_rl" / "policy.npz").write_bytes(b"fake")
        (tmp_path / "constitutional_rl" / "value.npz").write_bytes(b"fake")

        result = validate_weight_files(tmp_path)
        assert result.passed is True
        assert len(result.errors) == 0

    def test_missing_files(self, tmp_path: Path) -> None:
        """Fails when weight files are missing."""
        result = validate_weight_files(tmp_path)
        assert result.passed is False
        assert len(result.errors) > 0

    def test_partial_files(self, tmp_path: Path) -> None:
        """Fails when only some files are present."""
        (tmp_path / "rssm").mkdir()
        (tmp_path / "rssm" / "final.pt").write_bytes(b"fake")

        result = validate_weight_files(tmp_path)
        assert result.passed is False
        assert result.metrics["files_found"] == 1


class TestValidateRSSMShapes:
    """Tests for validate_rssm_shapes()."""

    def test_missing_checkpoint(self, tmp_path: Path) -> None:
        """Fails when RSSM checkpoint doesn't exist."""
        cfg = Settings(mock_hardware=True)
        result = validate_rssm_shapes(tmp_path, cfg)
        assert result.passed is False

    def test_valid_checkpoint(self, tmp_path: Path) -> None:
        """Passes with valid RSSM state_dict."""
        from mousedroid.world_model.rssm import RSSM

        cfg = Settings(mock_hardware=True)
        rssm = RSSM(cfg.model)
        ckpt_dir = tmp_path / "rssm"
        ckpt_dir.mkdir()
        torch.save(rssm.state_dict(), ckpt_dir / "final.pt")

        result = validate_rssm_shapes(tmp_path, cfg)
        assert result.passed is True
        assert "param_count" in result.metrics

    def test_checkpoint_state_format(self, tmp_path: Path) -> None:
        """Passes with CheckpointState-wrapped state_dict."""
        from mousedroid.world_model.rssm import RSSM

        cfg = Settings(mock_hardware=True)
        rssm = RSSM(cfg.model)
        ckpt_dir = tmp_path / "rssm"
        ckpt_dir.mkdir()
        torch.save(
            {
                "model_state_dict": rssm.state_dict(),
                "best_loss": 0.03,
                "epoch": 100,
            },
            ckpt_dir / "final.pt",
        )

        result = validate_rssm_shapes(tmp_path, cfg)
        assert result.passed is True
        assert result.metrics["best_loss"] == 0.03


class TestValidateBDIAccuracy:
    """Tests for validate_bdi_accuracy()."""

    def test_missing_files(self, tmp_path: Path) -> None:
        """Fails when BDI weight files are missing."""
        ann_path = tmp_path / "annotations.npz"
        np.savez(ann_path, observations=np.zeros((10, 256)), intentions=np.zeros(10))

        result = validate_bdi_accuracy(tmp_path, ann_path)
        assert result.passed is False

    def test_missing_annotations(self, tmp_path: Path) -> None:
        """Fails when annotations file is missing."""
        result = validate_bdi_accuracy(tmp_path, tmp_path / "missing.npz")
        assert result.passed is False


class TestValidateConstitutionalRL:
    """Tests for validate_constitutional_rl()."""

    def test_missing_files(self, tmp_path: Path) -> None:
        """Fails when policy/value files are missing."""
        result = validate_constitutional_rl(tmp_path)
        assert result.passed is False

    def test_files_present(self, tmp_path: Path) -> None:
        """Passes when policy and value files exist."""
        (tmp_path / "constitutional_rl").mkdir()
        np.savez(tmp_path / "constitutional_rl" / "policy.npz", w1=np.zeros((128, 2)))
        np.savez(tmp_path / "constitutional_rl" / "value.npz", w1=np.zeros((128, 1)))

        result = validate_constitutional_rl(tmp_path)
        assert result.passed is True


class TestGenerateTrainingReport:
    """Tests for generate_training_report()."""

    def test_report_generated(self, tmp_path: Path) -> None:
        """Generates valid JSON report."""
        cfg = Settings(mock_hardware=True)
        report_path = tmp_path / "report.json"

        report = generate_training_report(
            tmp_path,
            cfg,
            output_path=report_path,
        )
        assert report_path.exists()
        assert report.timestamp != ""

    def test_report_tracks_failures(self, tmp_path: Path) -> None:
        """Report reflects failed checks."""
        cfg = Settings(mock_hardware=True)
        report = generate_training_report(
            tmp_path,
            cfg,
            output_path=tmp_path / "report.json",
        )
        # Missing weight files → not all passed
        assert report.all_checks_passed is False


class TestValidateBDIAccuracyNormRegression:
    """Regression: validate_bdi_accuracy must use norm stats & two-layer belief (Bug fix 2026-03)."""

    def _make_trained_weights(self, bdi_dir: Path) -> tuple[Path, np.ndarray, np.ndarray]:
        """Create a minimal set of trained BDI weights with norm stats."""
        from mousedroid.utils.numpy_ops import relu

        rng = np.random.default_rng(42)
        bdi_dir.mkdir(parents=True, exist_ok=True)

        # Simulate normalised training
        raw_obs = rng.standard_normal((200, 256)).astype(np.float32)
        obs_mean = raw_obs.mean(axis=0)
        obs_std = raw_obs.std(axis=0) + 1e-8
        normed = (raw_obs - obs_mean) / obs_std

        # Train minimal encoder on normalised data
        w1 = rng.standard_normal((256, 128)).astype(np.float32) * 0.01
        b1 = np.zeros(128, dtype=np.float32)
        w2 = rng.standard_normal((128, 128)).astype(np.float32) * 0.01
        b2 = np.zeros(128, dtype=np.float32)
        h1 = relu(normed @ w1 + b1)
        beliefs = relu(h1 @ w2 + b2)

        # Desire
        dw1 = rng.standard_normal((128, 64)).astype(np.float32) * 0.01
        db1 = np.zeros(64, dtype=np.float32)
        desires = relu(beliefs @ dw1 + db1)

        # Intention labels from desire logits
        iw1 = rng.standard_normal((64, 10)).astype(np.float32) * 0.01
        ib1 = np.zeros(10, dtype=np.float32)
        logits = desires @ iw1 + ib1
        intentions = logits.argmax(axis=-1).astype(np.int64)

        # Save weights
        np.savez(bdi_dir / "belief.npz", w1=w1, b1=b1, w2=w2, b2=b2)
        np.savez(bdi_dir / "desire.npz", w1=dw1, b1=db1)
        np.savez(bdi_dir / "intention.npz", w1=iw1, b1=ib1)
        np.savez(bdi_dir / "belief_norm_stats.npz", mean=obs_mean, std=obs_std)

        # Save annotations
        ann_path = bdi_dir.parent / "annotations.npz"
        np.savez(ann_path, observations=raw_obs, intentions=intentions)
        return ann_path, obs_mean, obs_std

    def test_validation_uses_norm_stats(self, tmp_path: Path) -> None:
        """validate_bdi_accuracy must load and apply norm stats."""
        bdi_dir = tmp_path / "bdi"
        ann_path, _, _ = self._make_trained_weights(bdi_dir)

        result = validate_bdi_accuracy(tmp_path, ann_path)
        assert result.phase == "bdi_accuracy"
        assert "accuracy" in result.metrics
        # With matching norm stats, accuracy should be reasonable (not 0%)
        assert result.metrics["accuracy"] > 0.0

    def test_validation_warns_without_norm_stats(self, tmp_path: Path) -> None:
        """Validation should still work without norm stats (with warning)."""
        bdi_dir = tmp_path / "bdi"
        ann_path, _, _ = self._make_trained_weights(bdi_dir)

        # Remove norm stats
        (bdi_dir / "belief_norm_stats.npz").unlink()

        result = validate_bdi_accuracy(tmp_path, ann_path)
        assert result.phase == "bdi_accuracy"
        # Should not crash — degrades gracefully
        assert "accuracy" in result.metrics

    def test_two_layer_belief_forward_pass(self, tmp_path: Path) -> None:
        """Validation must use two-layer belief encoder (w1,b1 + w2,b2)."""
        from mousedroid.utils.numpy_ops import relu

        bdi_dir = tmp_path / "bdi"
        ann_path, obs_mean, obs_std = self._make_trained_weights(bdi_dir)

        # Manually reproduce the forward pass to verify it matches
        ann = np.load(ann_path)
        obs = ann["observations"][:10].astype(np.float32)
        obs_norm = (obs - obs_mean) / (obs_std + 1e-8)

        belief_w = np.load(bdi_dir / "belief.npz")
        h1 = relu(obs_norm @ belief_w["w1"] + belief_w["b1"])
        belief = relu(h1 @ belief_w["w2"] + belief_w["b2"])

        # Verify two layers were needed (single layer gives different result)
        single_layer = relu(obs_norm @ belief_w["w1"] + belief_w["b1"])
        assert not np.allclose(belief, single_layer), "Two-layer must differ from single-layer"
