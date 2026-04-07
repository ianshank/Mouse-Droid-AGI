"""Tests for mousedroid.common.actions — action normalisation utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mousedroid.common.actions import normalize_action_numpy, normalize_action_tensor


class TestNormalizeActionTensor:
    """Tests for normalize_action_tensor."""

    def test_exact_dim_passes_through(self) -> None:
        action = torch.tensor([0.5, -0.3, 0.1])
        result = normalize_action_tensor(action, expected_dim=3)
        assert result.shape == (3,)
        torch.testing.assert_close(result, torch.tensor([0.5, -0.3, 0.1]))

    def test_pads_when_too_small(self) -> None:
        action = torch.tensor([0.5])
        result = normalize_action_tensor(action, expected_dim=3)
        assert result.shape == (3,)
        assert result[0].item() == 0.5
        assert result[1].item() == 0.0
        assert result[2].item() == 0.0

    def test_truncates_when_too_large(self) -> None:
        action = torch.tensor([0.5, -0.3, 0.1, 0.9])
        result = normalize_action_tensor(action, expected_dim=2)
        assert result.shape == (2,)
        torch.testing.assert_close(result, torch.tensor([0.5, -0.3]))

    def test_clamps_to_range(self) -> None:
        action = torch.tensor([2.0, -3.0, 0.5])
        result = normalize_action_tensor(action, expected_dim=3)
        assert result[0].item() == 1.0
        assert result[1].item() == -1.0
        assert result[2].item() == 0.5

    def test_multidim_flattened(self) -> None:
        action = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        result = normalize_action_tensor(action, expected_dim=4)
        assert result.shape == (4,)

    def test_detaches_gradient(self) -> None:
        action = torch.tensor([0.5], requires_grad=True)
        result = normalize_action_tensor(action, expected_dim=1)
        assert not result.requires_grad


class TestNormalizeActionNumpy:
    """Tests for normalize_action_numpy."""

    def test_converts_numpy_to_tensor(self) -> None:
        arr = np.array([0.5, -0.3, 0.1], dtype=np.float32)
        result = normalize_action_numpy(arr, expected_dim=3)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)

    def test_pads_numpy_input(self) -> None:
        arr = np.array([0.5], dtype=np.float64)
        result = normalize_action_numpy(arr, expected_dim=3)
        assert result.shape == (3,)
        assert result[0].item() == pytest.approx(0.5, abs=1e-5)

    def test_clamps_numpy_input(self) -> None:
        arr = np.array([5.0, -5.0], dtype=np.float32)
        result = normalize_action_numpy(arr, expected_dim=2)
        assert result[0].item() == 1.0
        assert result[1].item() == -1.0
