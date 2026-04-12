"""Property-based tests for action normalisation utilities."""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.common.actions import normalize_action_numpy, normalize_action_tensor

# ---------------------------------------------------------------------------
# Bounds: output is always clamped to [-1, 1]
# ---------------------------------------------------------------------------


@given(
    values=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=16,
    ),
    expected_dim=st.integers(min_value=1, max_value=32),
)
@settings(max_examples=200)
def test_normalize_tensor_bounds(values: list[float], expected_dim: int) -> None:
    """Normalised tensor values must lie in [-1, 1]."""
    tensor = torch.tensor(values, dtype=torch.float32)
    result = normalize_action_tensor(tensor, expected_dim)
    assert result.shape == (expected_dim,)
    assert (result >= -1.0).all()
    assert (result <= 1.0).all()


@given(
    values=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=16,
    ),
    expected_dim=st.integers(min_value=1, max_value=32),
)
@settings(max_examples=200)
def test_normalize_numpy_bounds(values: list[float], expected_dim: int) -> None:
    """Numpy-to-tensor normalisation also keeps values in [-1, 1]."""
    arr = np.array(values, dtype=np.float32)
    result = normalize_action_numpy(arr, expected_dim)
    assert result.shape == (expected_dim,)
    assert (result >= -1.0).all()
    assert (result <= 1.0).all()


# ---------------------------------------------------------------------------
# Idempotence: normalising already-clamped values should be a no-op
# ---------------------------------------------------------------------------


@given(
    values=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=16,
    ),
)
@settings(max_examples=200)
def test_normalize_idempotent_for_clamped_input(values: list[float]) -> None:
    """Values already in [-1, 1] should be unchanged after normalisation."""
    dim = len(values)
    tensor = torch.tensor(values, dtype=torch.float32)
    result = normalize_action_tensor(tensor, dim)
    torch.testing.assert_close(result, tensor, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Padding: shorter input gets zero-padded
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=1, max_value=8),
    pad=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100)
def test_padding_appends_zeros(n: int, pad: int) -> None:
    """When input is shorter than expected_dim, trailing entries are zero."""
    expected_dim = n + pad
    tensor = torch.ones(n, dtype=torch.float32) * 0.5
    result = normalize_action_tensor(tensor, expected_dim)
    assert result.shape == (expected_dim,)
    # Trailing padded entries must be zero
    assert (result[n:] == 0.0).all()


# ---------------------------------------------------------------------------
# Truncation: longer input is truncated
# ---------------------------------------------------------------------------


@given(
    expected_dim=st.integers(min_value=1, max_value=8),
    extra=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100)
def test_truncation_keeps_prefix(expected_dim: int, extra: int) -> None:
    """When input is longer than expected_dim, only the first entries survive."""
    full_dim = expected_dim + extra
    values = torch.arange(full_dim, dtype=torch.float32) * 0.1
    result = normalize_action_tensor(values, expected_dim)
    assert result.shape == (expected_dim,)
    # Should match the clamped prefix
    expected = torch.clamp(values[:expected_dim], -1.0, 1.0)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
