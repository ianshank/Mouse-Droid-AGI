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
        for sub in ["rssm", "mcts", "bdi"]:
            (tmp_path / sub).mkdir()
        (tmp_path / "rssm" / "final.pt").write_bytes(b"fake")
        (tmp_path / "mcts" / "policy_init.npz").write_bytes(b"fake")
        for name in ["belief.npz", "desire.npz", "intention.npz", "affect.npz"]:
            (tmp_path / "bdi" / name).write_bytes(b"fake")
        (tmp_path / "policy.npz").write_bytes(b"fake")
        (tmp_path / "value.npz").write_bytes(b"fake")

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
        np.savez(tmp_path / "policy.npz", w1=np.zeros((128, 2)))
        np.savez(tmp_path / "value.npz", w1=np.zeros((128, 1)))

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
