"""Property-based tests for reward aggregation utilities."""

from __future__ import annotations

import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mousedroid.reward.aggregator import pareto_dominates, pareto_weighted_sum

# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

_pos_float = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)
_score_float = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)


def _objectives(names: list[str], value: float) -> dict[str, torch.Tensor]:
    """Build an objectives dict where every key maps to the same scalar tensor."""
    return {n: torch.tensor(value) for n in names}


# ---------------------------------------------------------------------------
# Monotonicity: increasing all component scores should not decrease aggregate
# ---------------------------------------------------------------------------


@given(
    base=_score_float,
    delta=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    w=_pos_float,
)
@settings(max_examples=200)
def test_monotonicity_single_objective(base: float, delta: float, w: float) -> None:
    """Increasing the single component score must not decrease the aggregate."""
    names = ["obj"]
    weights = {"obj": w}
    low = pareto_weighted_sum(_objectives(names, base), weights)
    high = pareto_weighted_sum(_objectives(names, base + delta), weights)
    assert high.item() >= low.item() - 1e-6


@given(
    base=_score_float,
    delta=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    w1=_pos_float,
    w2=_pos_float,
)
@settings(max_examples=200)
def test_monotonicity_two_objectives(base: float, delta: float, w1: float, w2: float) -> None:
    """When both objectives increase, the aggregate must not decrease."""
    names = ["a", "b"]
    weights = {"a": w1, "b": w2}
    low = pareto_weighted_sum(
        {n: torch.tensor(base) for n in names},
        weights,
    )
    high = pareto_weighted_sum(
        {n: torch.tensor(base + delta) for n in names},
        weights,
    )
    assert high.item() >= low.item() - 1e-6


# ---------------------------------------------------------------------------
# Boundedness: aggregate is bounded by min/max of components (weighted avg)
# ---------------------------------------------------------------------------


@given(
    vals=st.lists(
        _score_float,
        min_size=1,
        max_size=5,
    ),
    raw_weights=st.lists(
        _pos_float,
        min_size=1,
        max_size=5,
    ),
)
@settings(max_examples=200)
def test_aggregate_bounded_by_components(vals: list[float], raw_weights: list[float]) -> None:
    """Weighted average must lie between min and max component values."""
    n = min(len(vals), len(raw_weights))
    assume(n >= 1)
    names = [f"o{i}" for i in range(n)]
    objectives = {names[i]: torch.tensor(vals[i]) for i in range(n)}
    weights = {names[i]: raw_weights[i] for i in range(n)}

    agg = pareto_weighted_sum(objectives, weights).item()
    lo = min(vals[:n])
    hi = max(vals[:n])
    assert agg >= lo - 1e-5
    assert agg <= hi + 1e-5


# ---------------------------------------------------------------------------
# Zero weights: result should be zero
# ---------------------------------------------------------------------------


def test_zero_weights_returns_zero() -> None:
    """When all weights are zero, the aggregate should be zero."""
    objectives = {"a": torch.tensor(5.0), "b": torch.tensor(3.0)}
    weights = {"a": 0.0, "b": 0.0}
    result = pareto_weighted_sum(objectives, weights)
    assert result.item() == 0.0


# ---------------------------------------------------------------------------
# Pareto dominance transitivity
# ---------------------------------------------------------------------------


@given(
    a_val=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    b_val=st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_pareto_dominance_strict(a_val: float, b_val: float) -> None:
    """If a is strictly better on every objective, a dominates b."""
    assume(a_val > b_val)
    a = {"x": a_val, "y": a_val}
    b = {"x": b_val, "y": b_val}
    assert pareto_dominates(a, b)
    assert not pareto_dominates(b, a)


def test_pareto_dominance_equal_not_dominant() -> None:
    """Equal solutions do not dominate each other."""
    a = {"x": 1.0, "y": 2.0}
    assert not pareto_dominates(a, a)
