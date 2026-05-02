"""Property-based tests for private math helpers.

Closes Priority 6.2 from docs/planning/NEXT_STEPS.md by adding Hypothesis
``@given`` tests for the two private helpers that have unique invariants:

- ``mousedroid.cognitive.bdi_model._bayesian_normalise`` — distribution
  normaliser used by the Neural BDI pipeline; safe against all-zero inputs
  via ``_BAYESIAN_SUM_EPS``.
- ``mousedroid.common.tools.motor_tools._clamp`` — saturating clamp used by
  the motor-control tool dispatcher.

``_safe_softmax`` is intentionally not duplicated here: it is a thin
delegator over ``mousedroid.common.math.numpy_ops.softmax``, whose
properties are already covered by ``test_numpy_ops_property.py``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from mousedroid.cognitive.bdi_model import _BAYESIAN_SUM_EPS, _bayesian_normalise
from mousedroid.common.tools.motor_tools import _clamp

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Non-negative finite floats — `_bayesian_normalise` is documented to expect
# a non-negative input. We exercise the upper edge generously so the
# `_BAYESIAN_SUM_EPS` correction stays vanishingly small relative to the sum.
_nonneg_floats = st.floats(
    min_value=0.0,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)

# Bounded floats for `_clamp` — keep the range broad enough to cover both
# in-range and out-of-range cases while avoiding NaN/Inf.
_finite_floats = st.floats(
    min_value=-1e9,
    max_value=1e9,
    allow_nan=False,
    allow_infinity=False,
)


# ---------------------------------------------------------------------------
# _bayesian_normalise
# ---------------------------------------------------------------------------


@given(
    values=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=64),
        elements=_nonneg_floats,
    ),
)
@settings(max_examples=200)
def test_bayesian_normalise_sums_to_approximately_one(
    values: NDArray[np.floating[Any]],
) -> None:
    """Normalised output sums to ~1 when the input has non-trivial mass."""
    # Skip the all-zero edge case here — it has its own dedicated test below.
    assume(float(np.sum(values)) > 1.0)

    result = _bayesian_normalise(values)
    total = float(np.sum(result))

    # The eps correction means the sum is `sum / (sum + eps)` which is
    # vanishingly close to but strictly below 1.0 for any positive sum.
    assert math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6), (
        f"normalised sum {total} not within tolerance of 1.0 "
        f"for input with sum {float(np.sum(values))}"
    )


@given(
    values=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=64),
        elements=_nonneg_floats,
    ),
)
@settings(max_examples=200)
def test_bayesian_normalise_preserves_shape(
    values: NDArray[np.floating[Any]],
) -> None:
    """Output shape matches input shape for any valid 1-D input."""
    result = _bayesian_normalise(values)
    assert result.shape == values.shape


@given(
    values=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=64),
        elements=_nonneg_floats,
    ),
)
@settings(max_examples=200)
def test_bayesian_normalise_nonnegative_input_yields_nonnegative_output(
    values: NDArray[np.floating[Any]],
) -> None:
    """Non-negative input produces non-negative, finite output."""
    result = _bayesian_normalise(values)
    assert np.all(result >= 0.0), f"normalisation produced negatives: {result}"
    assert np.all(np.isfinite(result)), f"normalisation produced non-finite: {result}"


@given(
    n=st.integers(min_value=1, max_value=64),
)
@settings(max_examples=20)
def test_bayesian_normalise_handles_all_zeros_without_nan(n: int) -> None:
    """All-zero input must not produce NaN/Inf — the eps guard kicks in."""
    values = np.zeros(n, dtype=np.float64)

    result = _bayesian_normalise(values)

    assert np.all(np.isfinite(result)), f"all-zero input produced non-finite: {result}"
    # With sum == 0, the divisor is exactly _BAYESIAN_SUM_EPS, so each output
    # is `0 / eps == 0`. The whole array stays at zero, which is the documented
    # safe behaviour ("safe against all-zeros").
    assert np.all(result == 0.0), f"all-zero input should map to all-zero output: {result}"


@given(
    values=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=64),
        elements=_nonneg_floats,
    ),
    scale=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_bayesian_normalise_scale_invariant_when_sum_is_large(
    values: NDArray[np.floating[Any]],
    scale: float,
) -> None:
    """Scaling input by a positive constant leaves the distribution unchanged.

    This holds modulo the eps correction, which is negligible when the input
    sum is large compared to ``_BAYESIAN_SUM_EPS``.
    """
    assume(float(np.sum(values)) > 1.0)

    out_orig = _bayesian_normalise(values)
    out_scaled = _bayesian_normalise(values * scale)

    # Tolerance proportional to the eps correction's relative size.
    sum_orig = float(np.sum(values))
    sum_scaled = sum_orig * scale
    eps_drift = _BAYESIAN_SUM_EPS * (1.0 / sum_orig + 1.0 / sum_scaled)
    tol = max(1e-6, eps_drift)

    assert np.allclose(out_orig, out_scaled, atol=tol), (
        f"scale invariance violated for scale={scale}: max delta="
        f"{float(np.max(np.abs(out_orig - out_scaled)))}, tol={tol}"
    )


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------


@given(
    value=_finite_floats,
    bounds=st.tuples(_finite_floats, _finite_floats),
)
@settings(max_examples=200)
def test_clamp_output_within_bounds(value: float, bounds: tuple[float, float]) -> None:
    """Output is always within the (sorted) bounds."""
    lower, upper = sorted(bounds)
    result = _clamp(value, lower=lower, upper=upper)
    assert lower <= result <= upper, (
        f"clamp({value}, [{lower}, {upper}]) -> {result} escaped the bounds"
    )


@given(
    value=_finite_floats,
    half_width=st.floats(min_value=1e-9, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_clamp_identity_when_in_range(value: float, half_width: float) -> None:
    """Inputs already within ``[value-half_width, value+half_width]`` pass through."""
    lower = value - half_width
    upper = value + half_width
    assert _clamp(value, lower=lower, upper=upper) == value


@given(
    value=_finite_floats,
    bounds=st.tuples(_finite_floats, _finite_floats),
)
@settings(max_examples=200)
def test_clamp_idempotent(value: float, bounds: tuple[float, float]) -> None:
    """Clamping twice yields the same result as clamping once."""
    lower, upper = sorted(bounds)
    once = _clamp(value, lower=lower, upper=upper)
    twice = _clamp(once, lower=lower, upper=upper)
    assert once == twice, f"clamp not idempotent: once={once}, twice={twice}"


@given(
    value=_finite_floats,
    bounds=st.tuples(_finite_floats, _finite_floats),
)
@settings(max_examples=200)
def test_clamp_saturates_below_lower(value: float, bounds: tuple[float, float]) -> None:
    """Inputs strictly below ``lower`` saturate to ``lower``."""
    lower, upper = sorted(bounds)
    assume(value < lower)
    assert _clamp(value, lower=lower, upper=upper) == lower


@given(
    value=_finite_floats,
    bounds=st.tuples(_finite_floats, _finite_floats),
)
@settings(max_examples=200)
def test_clamp_saturates_above_upper(value: float, bounds: tuple[float, float]) -> None:
    """Inputs strictly above ``upper`` saturate to ``upper``."""
    lower, upper = sorted(bounds)
    assume(value > upper)
    assert _clamp(value, lower=lower, upper=upper) == upper


@given(
    value=_finite_floats,
    point=_finite_floats,
)
@settings(max_examples=100)
def test_clamp_degenerate_bounds_return_the_point(value: float, point: float) -> None:
    """When ``lower == upper``, output is always exactly that point."""
    assert _clamp(value, lower=point, upper=point) == point
