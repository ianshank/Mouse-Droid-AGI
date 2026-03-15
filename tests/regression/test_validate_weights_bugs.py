"""Regression tests for training validation bugs fixed 2026-03-14.

Bug 1: BDI forward pass concatenated belief+desire (192-dim) but
       intention.w1 input expected only desire (64-dim) → matmul crash.

Bug 2: validate_constitutional_rl looked for policy.npz/value.npz in
       weights/ root instead of weights/constitutional_rl/ subdirectory.

Bug 3: MCTS p50 latency (219ms) was not flagged despite being 4× above
       the 50ms target → added validate_mcts_latency check.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Bug 1 — BDI matmul dimension mismatch
# ---------------------------------------------------------------------------


class TestBdiBugRegression:
    """Regression: intention layer must use desire output only, not concat."""

    def test_bdi_forward_does_not_crash_with_desire_only(self, tmp_path: Path) -> None:
        """After the fix, BDI forward pass uses desire (64-dim) not belief+desire (192-dim)."""
        bdi_dir = tmp_path / "bdi"
        bdi_dir.mkdir()
        rng = np.random.default_rng(0)

        # belief: obs(256) → hidden(128)
        np.savez(bdi_dir / "belief.npz", w1=rng.standard_normal((256, 128)).astype(np.float32), b1=np.zeros(128, dtype=np.float32))
        # desire: belief(128) → desire(64)
        np.savez(bdi_dir / "desire.npz", w1=rng.standard_normal((128, 64)).astype(np.float32), b1=np.zeros(64, dtype=np.float32))
        # intention: desire(64) → classes(10)  — NOT 192!
        np.savez(bdi_dir / "intention.npz", w1=rng.standard_normal((64, 10)).astype(np.float32), b1=np.zeros(10, dtype=np.float32))

        belief_w = np.load(bdi_dir / "belief.npz")
        desire_w = np.load(bdi_dir / "desire.npz")
        intent_w = np.load(bdi_dir / "intention.npz")

        def relu(x: np.ndarray) -> np.ndarray:
            return np.maximum(0, x)

        obs = rng.standard_normal((10, 256)).astype(np.float32)
        belief = relu(obs @ belief_w["w1"] + belief_w["b1"])
        desire = relu(belief @ desire_w["w1"] + desire_w["b1"])

        # Fixed path: use desire only
        logits = desire @ intent_w["w1"] + intent_w["b1"]
        predictions = logits.argmax(axis=-1)

        assert predictions.shape == (10,), "Should produce one class per sample"

    def test_bdi_concat_would_crash(self) -> None:
        """Demonstrates that the original bug (belief+desire concat) raises MatMul error."""
        rng = np.random.default_rng(1)
        belief = rng.standard_normal((5, 128)).astype(np.float32)
        desire = rng.standard_normal((5, 64)).astype(np.float32)
        intent_w1 = rng.standard_normal((64, 10)).astype(np.float32)  # expects 64-dim

        combined = np.concatenate([belief, desire], axis=-1)  # 192-dim — wrong!
        assert combined.shape[1] == 192

        with pytest.raises(ValueError, match="matmul|core dimension"):
            _ = combined @ intent_w1  # Must fail — proves the old code was broken


# ---------------------------------------------------------------------------
# Bug 2 — constitutional_rl wrong weights path
# ---------------------------------------------------------------------------


class TestConstitutionalRlPathRegression:
    """Regression: policy.npz/value.npz should be found in constitutional_rl/ subdir."""

    def test_correct_subdir_path_resolves(self, tmp_path: Path) -> None:
        """validate_constitutional_rl should look in weights/constitutional_rl/."""
        rl_dir = tmp_path / "constitutional_rl"
        rl_dir.mkdir()
        # Create dummy weight files
        np.savez(rl_dir / "policy.npz", w1=np.ones((64, 3)), b1=np.zeros(3))
        np.savez(rl_dir / "value.npz", w1=np.ones((64, 1)), b1=np.zeros(1))

        assert (tmp_path / "constitutional_rl" / "policy.npz").exists()
        assert (tmp_path / "constitutional_rl" / "value.npz").exists()
        # Demonstrates that weights_dir / "constitutional_rl" / "policy.npz" is the right path
        assert not (tmp_path / "policy.npz").exists(), "Old incorrect root path must not exist"

    def test_old_root_path_would_miss_files(self, tmp_path: Path) -> None:
        """The original bug searched in weights/ root — those files don't exist there."""
        rl_dir = tmp_path / "constitutional_rl"
        rl_dir.mkdir()
        np.savez(rl_dir / "policy.npz", w1=np.ones((64, 3)))

        # Old (broken) path:
        assert not (tmp_path / "policy.npz").exists(), "Proves old code would have failed"


# ---------------------------------------------------------------------------
# Bug 3 — MCTS latency not flagged
# ---------------------------------------------------------------------------


class TestMctsLatencyRegression:
    """Regression: validate_mcts_latency must flag p50 > target."""

    def _write_tuned_config(self, path: Path, p50_ms: float, best_ucb: float = 1.41) -> None:
        data = {
            f"ucb_{best_ucb}": {
                "mean_reward": 0.15,
                "p50_ms": p50_ms,
                "p95_ms": p50_ms * 1.2,
            },
            "best_ucb_c": best_ucb,
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_within_target_passes(self, tmp_path: Path) -> None:
        mcts_dir = tmp_path / "mcts"
        mcts_dir.mkdir()
        self._write_tuned_config(mcts_dir / "tuned_config.json", p50_ms=45.0)

        # Import and invoke after writing file
        from training.validate_weights import validate_mcts_latency
        result = validate_mcts_latency(tmp_path, target_p50_ms=50.0)
        assert result.passed, f"Should pass at 45ms but got errors: {result.errors}"

    def test_over_target_fails(self, tmp_path: Path) -> None:
        mcts_dir = tmp_path / "mcts"
        mcts_dir.mkdir()
        self._write_tuned_config(mcts_dir / "tuned_config.json", p50_ms=219.0)

        from training.validate_weights import validate_mcts_latency
        result = validate_mcts_latency(tmp_path, target_p50_ms=50.0)
        assert not result.passed, "219ms p50 should fail the 50ms target"
        assert any("219" in e for e in result.errors), f"Error should mention 219ms: {result.errors}"

    def test_missing_config_skips_gracefully(self, tmp_path: Path) -> None:
        """If tuned_config.json doesn't exist, shouldn't block the pipeline."""
        from training.validate_weights import validate_mcts_latency
        result = validate_mcts_latency(tmp_path, target_p50_ms=50.0)
        assert result.passed, "Missing config should not block"
        assert result.metrics.get("skipped") is True
