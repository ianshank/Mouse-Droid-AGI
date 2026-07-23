"""Unit tests for ``training/drift_reduction.py`` + the pretrainer drift seam.

Pins the parity discipline (``corruption_prob=0`` ⇒ augmented arm EXACTLY
reproduces the baseline arm), input validation, and the
``RSSMPretrainer`` byte-identical legacy path (``drift=None`` / omitted /
disabled all construct the same training behaviour).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mousedroid.common.torch_device import resolve_device
from mousedroid.config.schema import DriftTrainingConfig, ModelConfig
from mousedroid.training.drift_reduction import train_pair_and_compare
from mousedroid.training.rssm_pretrainer import RSSMPretrainer
from mousedroid.training.sim_episode_generator import EpisodeBatch
from mousedroid.world_model.rssm import RSSM
from tests.unit._rssm_drift_helpers import seq_batch, tiny_rssm_cfg

_B = 2
_T = 16


def _tiny_cfg() -> ModelConfig:
    return tiny_rssm_cfg()


def _batch(mcfg: ModelConfig, seed: int = 0) -> dict[str, torch.Tensor]:
    return seq_batch(mcfg, episodes=_B, seq_len=_T, seed=seed)


def _drift_cfg(**overrides: object) -> DriftTrainingConfig:
    return DriftTrainingConfig.model_validate(
        {"eval_context_steps": 4, "eval_horizon": 8, **overrides}
    )


class TestTrainPairAndCompare:
    def test_corruption_prob_zero_reproduces_baseline_exactly(self) -> None:
        mcfg = _tiny_cfg()
        result = train_pair_and_compare(
            mcfg,
            _drift_cfg(corruption_prob=0.0, residual_head=False),
            [_batch(mcfg, seed=0), _batch(mcfg, seed=1)],
            _batch(mcfg, seed=9),
            steps=4,
            learning_rate=1e-3,
        )
        assert result.corrupted_batches == 0
        # Both arms re-seed identically per step, so p=0 ⇒ EXACT reproduction.
        assert result.baseline.per_step_mse == result.augmented.per_step_mse
        assert result.baseline_final_train_loss == result.augmented_final_train_loss

    def test_corruption_produces_finite_divergent_reports(self) -> None:
        mcfg = _tiny_cfg()
        result = train_pair_and_compare(
            mcfg,
            _drift_cfg(corruption_prob=1.0),
            [_batch(mcfg, seed=0), _batch(mcfg, seed=1)],
            _batch(mcfg, seed=9),
            steps=4,
            learning_rate=1e-3,
        )
        assert result.corrupted_batches == 4
        channel = result.baseline.headline_channel
        assert result.baseline.mean(channel) >= 0.0
        assert result.augmented.mean(channel) >= 0.0
        # With every batch corrupted, the arms must have diverged.
        assert result.baseline.per_step_mse != result.augmented.per_step_mse
        # The residual-head channel exists only on the augmented arm.
        assert "motor_corrected" in result.augmented.channels()
        assert "motor_corrected" not in result.baseline.channels()

    def test_empty_batches_rejected(self) -> None:
        mcfg = _tiny_cfg()
        with pytest.raises(ValueError, match="non-empty"):
            train_pair_and_compare(
                mcfg, _drift_cfg(), [], _batch(mcfg), steps=1, learning_rate=1e-3
            )

    def test_nonpositive_steps_rejected(self) -> None:
        mcfg = _tiny_cfg()
        with pytest.raises(ValueError, match="steps"):
            train_pair_and_compare(
                mcfg, _drift_cfg(), [_batch(mcfg)], _batch(mcfg), steps=0, learning_rate=1e-3
            )


def _episode_batch(mcfg: ModelConfig, seed: int = 0) -> EpisodeBatch:
    tensors = _batch(mcfg, seed=seed)
    return EpisodeBatch(
        motor=tensors["motor"],
        ultrasonic=tensors["ultrasonic"],
        lidar=torch.zeros(_B, _T, 0),
        valid_mask=tensors["valid_mask"],
        action=tensors["action"],
        reward=torch.zeros(_B, _T, 1),
        vision=torch.zeros(_B, _T, 0),
    )


def _pretrain_history(
    mcfg: ModelConfig, tmp_path: Path, drift: DriftTrainingConfig | None, tag: str
) -> list[float]:
    torch.manual_seed(0)
    model = RSSM(mcfg)
    torch.manual_seed(1)
    trainer = RSSMPretrainer(
        model,
        lr=1e-3,
        grad_clip=100.0,
        amp=False,
        device=torch.device("cpu"),
        drift=drift,
    )
    torch.manual_seed(2)
    return trainer.train([_episode_batch(mcfg)], epochs=2, checkpoint_path=tmp_path / f"{tag}.pt")


class TestPretrainerDriftSeam:
    def test_none_and_disabled_paths_byte_identical(self, tmp_path: Path) -> None:
        mcfg = _tiny_cfg()
        history_none = _pretrain_history(mcfg, tmp_path, None, "none")
        history_disabled = _pretrain_history(mcfg, tmp_path, DriftTrainingConfig(), "disabled")
        assert history_none == history_disabled

    def test_enabled_drift_changes_training(self, tmp_path: Path) -> None:
        mcfg = _tiny_cfg()
        history_none = _pretrain_history(mcfg, tmp_path, None, "none2")
        history_drift = _pretrain_history(
            mcfg,
            tmp_path,
            _drift_cfg(enabled=True, corruption_prob=1.0),
            "drift",
        )
        assert history_none != history_drift


class TestDeviceAgnostic:
    def test_default_device_resolves_to_available_hardware(self) -> None:
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert resolve_device(None).type == expected

    def test_explicit_cpu_passthrough(self) -> None:
        assert resolve_device("cpu").type == "cpu"
        assert resolve_device(torch.device("cpu")).type == "cpu"

    def test_cpu_batches_train_on_explicit_cpu(self) -> None:
        """Explicit device= threads end-to-end (CPU host executes the cpu leg)."""
        mcfg = _tiny_cfg()
        result = train_pair_and_compare(
            mcfg,
            _drift_cfg(corruption_prob=0.0, residual_head=False),
            [_batch(mcfg, seed=0)],
            _batch(mcfg, seed=9),
            steps=2,
            learning_rate=1e-3,
            device="cpu",
        )
        assert result.steps == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only pair run")
    def test_cuda_pair_run_produces_finite_reports(self) -> None:
        mcfg = _tiny_cfg()
        result = train_pair_and_compare(
            mcfg,
            _drift_cfg(corruption_prob=1.0),
            [_batch(mcfg, seed=0)],  # CPU batches, moved internally
            _batch(mcfg, seed=9),
            steps=2,
            learning_rate=1e-3,
            device="cuda",
        )
        channel = result.baseline.headline_channel
        assert result.baseline.mean(channel) >= 0.0
        assert result.augmented.mean(channel) >= 0.0
