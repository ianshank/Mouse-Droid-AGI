"""Unit tests for batch tuner module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mousedroid.config.schema import TrainingPipelineConfig
from mousedroid.training.batch_tuner import VRAMBatchTuner


@pytest.fixture
def config() -> TrainingPipelineConfig:
    """Create default pipeline config."""
    return TrainingPipelineConfig(vram_headroom_mb=512)


@pytest.fixture
def tuner(config: TrainingPipelineConfig) -> VRAMBatchTuner:
    """Create a batch tuner instance."""
    return VRAMBatchTuner(config)


def test_scales_down_when_vram_constrained(config: TrainingPipelineConfig) -> None:
    """Batch size should scale down when VRAM is limited."""
    tuner = VRAMBatchTuner(config)
    # Mock: 2048 MB free, 8192 MB total -> usable = 2048-512=1536
    # ratio = 1536/8192 ~= 0.1875 -> 16 * 0.1875 = 3
    with patch.object(tuner, "_get_vram_info", return_value=(2048, 8192)):
        result = tuner.tune_batch_size("rssm", 16)
    assert result < 16
    assert result >= 1


def test_returns_base_size_when_vram_plentiful(config: TrainingPipelineConfig) -> None:
    """Batch size should equal base_size when VRAM is abundant."""
    tuner = VRAMBatchTuner(config)
    # Mock: 7680 MB free, 8192 MB total -> usable = 7680-512=7168
    # ratio = 7168/8192 ~= 0.875 -> 16 * 0.875 = 14 (less than 16)
    # Actually need ratio >= 1.0 to get full base_size back
    # 8704 free, 8192 total -> usable = 8704-512=8192, ratio=8192/8192=1.0
    with patch.object(tuner, "_get_vram_info", return_value=(8704, 8192)):
        result = tuner.tune_batch_size("rssm", 16)
    assert result == 16


def test_floor_at_one(config: TrainingPipelineConfig) -> None:
    """Batch size should never go below 1."""
    tuner = VRAMBatchTuner(config)
    # Mock: 100 MB free, 8192 total -> usable = max(0, 100-512)=0
    # ratio = 0/8192 = 0 -> max(1, 16*0) = 1
    with patch.object(tuner, "_get_vram_info", return_value=(100, 8192)):
        result = tuner.tune_batch_size("rssm", 16)
    assert result == 1


def test_cap_at_base_size(config: TrainingPipelineConfig) -> None:
    """Batch size should never exceed base_size."""
    tuner = VRAMBatchTuner(config)
    # Even with absurdly high free memory the result caps at base.
    with patch.object(tuner, "_get_vram_info", return_value=(100000, 8192)):
        result = tuner.tune_batch_size("rssm", 32)
    assert result == 32


def test_no_cuda_returns_base_size(config: TrainingPipelineConfig) -> None:
    """When CUDA is unavailable, return base_size for CPU training."""
    tuner = VRAMBatchTuner(config)
    with patch.object(tuner, "_get_vram_info", return_value=(0, 0)):
        result = tuner.tune_batch_size("bdi", 32)
    assert result == 32
