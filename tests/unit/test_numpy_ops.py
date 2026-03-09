"""Tests for shared numpy operations utility module."""

from __future__ import annotations

import numpy as np

from mousedroid.utils.numpy_ops import layer_norm, relu, softmax


class TestRelu:
    def test_positive_values_unchanged(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(relu(x), x)

    def test_negative_values_zeroed(self) -> None:
        x = np.array([-1.0, -0.5, -100.0])
        np.testing.assert_array_equal(relu(x), np.zeros(3))

    def test_mixed_values(self) -> None:
        x = np.array([-1.0, 0.0, 1.0, -0.5, 0.5])
        expected = np.array([0.0, 0.0, 1.0, 0.0, 0.5])
        np.testing.assert_array_equal(relu(x), expected)

    def test_zero_input(self) -> None:
        x = np.zeros(5)
        np.testing.assert_array_equal(relu(x), np.zeros(5))

    def test_preserves_dtype(self) -> None:
        x = np.array([-1.0, 1.0], dtype=np.float32)
        assert relu(x).dtype == np.float32  # preserves input dtype


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert abs(float(np.sum(result)) - 1.0) < 1e-6

    def test_large_logits_stable(self) -> None:
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert np.all(np.isfinite(result))
        assert abs(float(np.sum(result)) - 1.0) < 1e-6

    def test_all_zeros_uniform(self) -> None:
        x = np.zeros(4)
        result = softmax(x)
        np.testing.assert_allclose(result, 0.25, atol=1e-6)

    def test_batch_softmax(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = softmax(x, axis=-1)
        for row in result:
            assert abs(float(np.sum(row)) - 1.0) < 1e-6

    def test_negative_logits(self) -> None:
        x = np.array([-10.0, -20.0, -30.0])
        result = softmax(x)
        assert np.all(result > 0)
        assert abs(float(np.sum(result)) - 1.0) < 1e-6


class TestLayerNorm:
    def test_zero_mean(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = layer_norm(x)
        assert abs(float(np.mean(result))) < 1e-6

    def test_unit_variance(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = layer_norm(x)
        assert abs(float(np.var(result)) - 1.0) < 1e-4

    def test_constant_input(self) -> None:
        x = np.ones(5) * 3.0
        result = layer_norm(x)
        # Constant input → all zeros (no variance)
        assert np.all(np.isfinite(result))

    def test_custom_eps(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        result = layer_norm(x, eps=1e-3)
        assert np.all(np.isfinite(result))
