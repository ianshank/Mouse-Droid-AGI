"""Property-based tests for numpy_ops using Hypothesis."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mousedroid.common.math.numpy_ops import layer_norm, relu, softmax

# Safe float strategy that avoids NaN/Inf to preserve mathematical properties
_safe_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------


@given(
    x=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=64),
        elements=_safe_floats,
    )
)
@settings(max_examples=200)
def test_softmax_sums_to_one(x: np.ndarray) -> None:
    """softmax output along the last axis must sum to 1.0 within tolerance."""
    result = softmax(x)
    assert np.isclose(result.sum(), 1.0, atol=1e-5), (
        f"softmax sum {result.sum()} is not close to 1.0 for input {x}"
    )


@given(
    x=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=64),
        elements=_safe_floats,
    )
)
@settings(max_examples=200)
def test_softmax_all_nonnegative(x: np.ndarray) -> None:
    """All softmax outputs must be non-negative."""
    result = softmax(x)
    assert np.all(result >= 0.0), f"softmax produced negative values for input {x}"


@given(
    x=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=64),
        elements=_safe_floats,
    )
)
@settings(max_examples=200)
def test_softmax_shift_invariant(x: np.ndarray) -> None:
    """softmax(x) must equal softmax(x - max(x)) — shift invariance."""
    shifted = x - np.max(x)
    result_original = softmax(x)
    result_shifted = softmax(shifted)
    assert np.allclose(result_original, result_shifted, atol=1e-6), (
        f"softmax is not shift-invariant for input {x}"
    )


@given(
    x=arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=1, max_value=8),
            st.integers(min_value=1, max_value=16),
        ),
        elements=_safe_floats,
    )
)
@settings(max_examples=100)
def test_softmax_2d_each_row_sums_to_one(x: np.ndarray) -> None:
    """For 2-D input each row (axis=-1) must sum to 1.0."""
    result = softmax(x, axis=-1)
    row_sums = result.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), (
        f"2-D softmax row sums {row_sums} not close to 1 for input shape {x.shape}"
    )


# ---------------------------------------------------------------------------
# relu
# ---------------------------------------------------------------------------


@given(
    x=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=128),
        elements=_safe_floats,
    )
)
@settings(max_examples=200)
def test_relu_all_nonnegative(x: np.ndarray) -> None:
    """ReLU output must be non-negative for all inputs."""
    result = relu(x)
    assert np.all(result >= 0.0), f"relu produced negative values for input {x}"


@given(
    x=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=128),
        elements=st.floats(min_value=1e-9, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=200)
def test_relu_identity_for_positive(x: np.ndarray) -> None:
    """relu(x) == x when all elements of x are strictly positive."""
    result = relu(x)
    assert np.allclose(result, x), f"relu did not act as identity on strictly positive input {x}"


@given(
    x=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=128),
        elements=st.floats(min_value=-1e6, max_value=-1e-9, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=200)
def test_relu_zeros_for_negative(x: np.ndarray) -> None:
    """relu(x) == 0 when all elements of x are strictly negative."""
    result = relu(x)
    assert np.all(result == 0.0), f"relu did not zero out strictly negative input {x}"


# ---------------------------------------------------------------------------
# layer_norm
# ---------------------------------------------------------------------------


@given(
    x=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=128),
        elements=_safe_floats,
    )
)
@settings(max_examples=200)
def test_layer_norm_near_zero_mean(x: np.ndarray) -> None:
    """layer_norm output must have near-zero mean."""
    # Assume away constant arrays — normalisation is undefined for zero variance
    assume(not np.all(x == x[0]))
    result = layer_norm(x)
    mean = float(np.mean(result))
    assert abs(mean) < 1e-5, f"layer_norm mean {mean} is not near zero for input {x}"


@given(
    x=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=128),
        elements=_safe_floats,
    )
)
@settings(max_examples=200)
def test_layer_norm_output_shape_preserved(x: np.ndarray) -> None:
    """layer_norm must return an array with the same shape as the input."""
    result = layer_norm(x)
    assert result.shape == x.shape, f"layer_norm changed shape from {x.shape} to {result.shape}"


@given(
    x=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=128),
        elements=_safe_floats,
    ),
    shift=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_layer_norm_translation_invariant(x: np.ndarray, shift: float) -> None:
    """layer_norm(x + c) must equal layer_norm(x) for any constant shift c.

    Adding a constant to all elements does not change the variance, so the
    normalised output is identical regardless of the fixed eps stabilisation.
    """
    assume(not np.all(x == x[0]))
    result_orig = layer_norm(x)
    result_shifted = layer_norm(x + shift)
    assert np.allclose(result_orig, result_shifted, atol=1e-5), (
        f"layer_norm is not translation-invariant for input {x} shifted by {shift}"
    )
