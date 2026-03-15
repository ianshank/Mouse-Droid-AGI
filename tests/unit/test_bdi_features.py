"""Tests for E2 Sprint 2 BDI Training capabilities.

Covers:
- BeliefEncoder normalisation (E2-S3)
- Class balance audit and oversampling (E2-S2)
- CI accuracy gate script (E2-S4)
- BDITrainingConfig backward compatibility (E2-S1)
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from mousedroid.cognitive.bdi_model import BeliefEncoder
from mousedroid.config.schema import BDITrainingConfig, Settings
from training.collect_annotations import (
    INTENTION_LABELS,
    audit_class_balance,
    balance_classes,
    label_intention,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def sample_observations(rng: np.random.Generator) -> np.ndarray:
    """100 random observations."""
    return rng.standard_normal((100, 256)).astype(np.float32)


@pytest.fixture()
def sample_intentions(rng: np.random.Generator) -> np.ndarray:
    """Imbalanced intentions: classes 0-4 get 15 each, 5-9 get 5 each."""
    labels = np.concatenate(
        [np.full(15, i, dtype=np.int64) for i in range(5)]
        + [np.full(5, i, dtype=np.int64) for i in range(5, 10)]
    )
    return rng.permutation(labels)


@pytest.fixture()
def tmp_weights(tmp_path: Path, rng: np.random.Generator) -> Path:
    """Minimal belief encoder weights."""
    w1 = rng.standard_normal((256, 128)).astype(np.float32) * 0.01
    b1 = np.zeros(128, dtype=np.float32)
    w2 = rng.standard_normal((128, 128)).astype(np.float32) * 0.01
    b2 = np.zeros(128, dtype=np.float32)
    weights_path = tmp_path / "belief.npz"
    np.savez(weights_path, w1=w1, b1=b1, w2=w2, b2=b2)
    return weights_path


@pytest.fixture()
def tmp_weights_with_norm(tmp_weights: Path, rng: np.random.Generator) -> Path:
    """Belief weights with companion norm stats file."""
    norm_path = tmp_weights.parent / "belief_norm_stats.npz"
    mean = rng.standard_normal(256).astype(np.float32)
    std = np.abs(rng.standard_normal(256)).astype(np.float32) + 0.1
    np.savez(norm_path, mean=mean, std=std)
    return tmp_weights


# ---------------------------------------------------------------------------
# belief normalisation
# ---------------------------------------------------------------------------


class TestBeliefEncoderNormalisation:
    def test_default_no_normalise(self) -> None:
        enc = BeliefEncoder()
        assert enc._normalise is False
        assert enc._norm_mean is None

    def test_normalise_without_weights_is_noop(self) -> None:
        enc = BeliefEncoder(normalise=True)
        x = np.random.default_rng().standard_normal(256).astype(np.float32)
        result = enc.forward(x)
        assert result.shape == (128,)

    def test_normalise_with_stats(self, tmp_weights_with_norm: Path) -> None:
        enc_norm = BeliefEncoder(tmp_weights_with_norm, normalise=True)
        enc_raw = BeliefEncoder(tmp_weights_with_norm, normalise=False)
        x = np.random.default_rng().standard_normal(256).astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            out_raw = enc_raw.forward(x)

        out_norm = enc_norm.forward(x)
        assert not np.allclose(out_raw, out_norm, atol=1e-6)

    def test_deprecation_warning(self, tmp_weights_with_norm: Path) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BeliefEncoder(tmp_weights_with_norm, normalise=False)
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_no_warning_without_stats(self, tmp_weights: Path) -> None:
        norm_file = tmp_weights.parent / "belief_norm_stats.npz"
        if norm_file.exists():
            norm_file.unlink()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BeliefEncoder(tmp_weights, normalise=False)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0


# ---------------------------------------------------------------------------
# class balance
# ---------------------------------------------------------------------------


class TestClassBalance:
    def test_audit_returns_all_classes(self, sample_intentions: np.ndarray) -> None:
        counts = audit_class_balance(sample_intentions)
        assert len(counts) == len(INTENTION_LABELS)

    def test_balance_increases_minority(
        self,
        sample_observations: np.ndarray,
        sample_intentions: np.ndarray,
    ) -> None:
        obs, intentions = balance_classes(sample_observations, sample_intentions)
        for cls_idx in range(5, 10):
            orig_cnt = int(np.sum(sample_intentions == cls_idx))
            new_cnt = int(np.sum(intentions == cls_idx))
            assert new_cnt > orig_cnt

    def test_balance_within_ratio(
        self,
        sample_observations: np.ndarray,
        sample_intentions: np.ndarray,
    ) -> None:
        _, intentions = balance_classes(
            sample_observations, sample_intentions, max_ratio=1.2
        )
        counts = np.array([int(np.sum(intentions == i)) for i in range(10)])
        assert (counts.max() / counts[counts > 0].min()) <= 1.25

    def test_balance_empty_class(self) -> None:
        obs = np.zeros((10, 256), dtype=np.float32)
        intentions = np.zeros(10, dtype=np.int64)
        _, balanced_int = balance_classes(obs, intentions)
        for i in range(1, 10):
            assert int(np.sum(balanced_int == i)) == 0


class TestLabelIntentionHelpers:
    def test_protect_human(self) -> None:
        obs = MagicMock(distance_m=1.0, motor_state=np.array([0, 0, 0, 12.0]))
        assert label_intention(np.array([0.5, 0]), obs, human_detected=True, human_dist_m=0.3) == 8

    def test_obey_command(self) -> None:
        obs = MagicMock(distance_m=1.0, motor_state=np.array([0, 0, 0, 12.0]))
        assert label_intention(np.array([0.5, 0]), obs, commanded_action=np.array([1, 0])) == 9


class TestBDITrainingConfig:
    def test_defaults(self) -> None:
        cfg = BDITrainingConfig()
        assert cfg.epochs == 200
        assert hasattr(Settings(mock_hardware=True), "bdi_training")


class TestCheckReportScript:
    def test_pass_report(self, tmp_path: Path) -> None:
        rep = tmp_path / "rep.json"
        rep.write_text(json.dumps({"phases": {"bdi": {"status": "pass", "accuracy": 0.65}}}))
        res = subprocess.run([
            sys.executable, "scripts/check_report.py",
            "--report", str(rep), "--phase", "bdi", "--must-pass"
        ], capture_output=True, cwd=str(Path(__file__).resolve().parents[2]))
        assert res.returncode == 0

    def test_fail_report(self, tmp_path: Path) -> None:
        rep = tmp_path / "rep.json"
        rep.write_text(json.dumps({"phases": {"bdi": {"status": "fail"}}}))
        res = subprocess.run([
            sys.executable, "scripts/check_report.py",
            "--report", str(rep), "--phase", "bdi", "--must-pass"
        ], capture_output=True, cwd=str(Path(__file__).resolve().parents[2]))
        assert res.returncode == 1

    def test_missing_phase(self, tmp_path: Path) -> None:
        rep = tmp_path / "rep.json"
        rep.write_text(json.dumps({"phases": {}}))
        res = subprocess.run([
            sys.executable, "scripts/check_report.py",
            "--report", str(rep), "--phase", "bdi", "--must-pass"
        ], capture_output=True, cwd=str(Path(__file__).resolve().parents[2]))
        assert res.returncode == 0
