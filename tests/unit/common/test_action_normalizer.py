"""Tests for mousedroid.common.actions — action normalisation utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mousedroid.common.actions import normalize_action_numpy, normalize_action_tensor


class TestNormalizeActionTensor:
    """Tests for normalize_action_tensor."""

    def test_exact_dim_passthrough(self):
        action = torch.tensor([0.5, -0.3, 0.8])
        result = normalize_action_tensor(action, 3)
        assert result.shape == (3,)
        torch.testing.assert_close(result, action)

    def test_pads_when_too_small(self):
        action = torch.tensor([0.5])
        result = normalize_action_tensor(action, 3)
        assert result.shape == (3,)
        assert result[0].item() == pytest.approx(0.5)
        assert result[1].item() == 0.0
        assert result[2].item() == 0.0

    def test_truncates_when_too_large(self):
        action = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        result = normalize_action_tensor(action, 3)
        assert result.shape == (3,)
        torch.testing.assert_close(result, torch.tensor([0.1, 0.2, 0.3]))

    def test_clamps_to_unit_range(self):
        action = torch.tensor([2.0, -3.0, 0.5])
        result = normalize_action_tensor(action, 3)
        assert result[0].item() == pytest.approx(1.0)
        assert result[1].item() == pytest.approx(-1.0)
        assert result[2].item() == pytest.approx(0.5)

    def test_2d_input_flattened(self):
        action = torch.tensor([[0.5, -0.3, 0.8]])
        result = normalize_action_tensor(action, 3)
        assert result.shape == (3,)

    def test_empty_pads_to_dim(self):
        action = torch.tensor([])
        result = normalize_action_tensor(action, 3)
        assert result.shape == (3,)
        assert (result == 0.0).all()

    def test_detaches_gradient(self):
        action = torch.tensor([0.5, -0.3, 0.8], requires_grad=True)
        result = normalize_action_tensor(action, 3)
        assert not result.requires_grad


class TestNormalizeActionNumpy:
    """Tests for normalize_action_numpy."""

    def test_basic_conversion(self):
        action = np.array([0.5, -0.3, 0.8], dtype=np.float32)
        result = normalize_action_numpy(action, 3)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)

    def test_pads_and_clamps(self):
        action = np.array([2.0], dtype=np.float32)
        result = normalize_action_numpy(action, 3)
        assert result[0].item() == pytest.approx(1.0)
        assert result[1].item() == 0.0

    def test_float64_input(self):
        action = np.array([0.5, -0.3], dtype=np.float64)
        result = normalize_action_numpy(action, 2)
        assert result.dtype == torch.float32
