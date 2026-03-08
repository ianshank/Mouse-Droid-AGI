"""Tests for Phase 2.1 — Synthetic data generation components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mousedroid.sensing.bundle import MouseDroidObservationBundle
from training.data_generator import _bundle_to_tensors


class TestBundleToTensors:
    """Test observation bundle → tensor conversion."""

    def test_output_keys(self) -> None:
        obs = MouseDroidObservationBundle()
        tensors = _bundle_to_tensors(obs)
        assert "vision" in tensors
        assert "ultrasonic" in tensors
        assert "motor_state" in tensors
        assert "valid_mask" in tensors

    def test_output_shapes(self) -> None:
        obs = MouseDroidObservationBundle()
        tensors = _bundle_to_tensors(obs)
        assert tensors["vision"].shape == (256,)
        assert tensors["ultrasonic"].shape == (1,)
        assert tensors["motor_state"].shape == (4,)
        assert tensors["valid_mask"].shape == (3,)

    def test_output_dtype(self) -> None:
        obs = MouseDroidObservationBundle()
        tensors = _bundle_to_tensors(obs)
        for v in tensors.values():
            assert v.dtype == torch.float32


class TestRSSMSequenceDataset:
    """Test PyTorch dataset wrapper."""

    def test_dataset_length(self, tmp_path: Path) -> None:
        from training.rssm_dataset import RSSMSequenceDataset

        # Create minimal fake data
        episodes = []
        for _ in range(5):
            ep = []
            for _ in range(10):
                ep.append({
                    "vision": torch.randn(16),
                    "ultrasonic": torch.randn(1),
                    "motor_state": torch.randn(4),
                    "valid_mask": torch.ones(3),
                    "action": torch.randn(3),
                })
            episodes.append(ep)

        data_path = tmp_path / "sequences.pt"
        torch.save(episodes, data_path)

        ds = RSSMSequenceDataset(data_path, seq_len=10)
        assert len(ds) == 5

    def test_dataset_item_shapes(self, tmp_path: Path) -> None:
        from training.rssm_dataset import RSSMSequenceDataset

        episodes = [[{
            "vision": torch.randn(16),
            "ultrasonic": torch.randn(1),
            "motor_state": torch.randn(4),
            "valid_mask": torch.ones(3),
            "action": torch.randn(3),
        } for _ in range(10)]]

        data_path = tmp_path / "sequences.pt"
        torch.save(episodes, data_path)

        ds = RSSMSequenceDataset(data_path, seq_len=8)
        vision, ultrasonic, motor_state, valid_mask, actions = ds[0]

        assert vision.shape == (8, 16)
        assert ultrasonic.shape == (8, 1)
        assert motor_state.shape == (8, 4)
        assert valid_mask.shape == (8, 3)
        assert actions.shape == (8, 3)

    def test_dataset_padding_short_episode(self, tmp_path: Path) -> None:
        from training.rssm_dataset import RSSMSequenceDataset

        # Episode with only 3 steps, but seq_len=10
        episodes = [[{
            "vision": torch.ones(16),
            "ultrasonic": torch.ones(1),
            "motor_state": torch.ones(4),
            "valid_mask": torch.ones(3),
            "action": torch.ones(3),
        } for _ in range(3)]]

        data_path = tmp_path / "sequences.pt"
        torch.save(episodes, data_path)

        ds = RSSMSequenceDataset(data_path, seq_len=10)
        vision, _, _, _, _ = ds[0]

        # First 3 timesteps should be ones, rest zeros (padding)
        assert torch.all(vision[:3] == 1.0)
        assert torch.all(vision[3:] == 0.0)
