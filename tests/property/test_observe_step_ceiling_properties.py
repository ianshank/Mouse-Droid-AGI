"""Hypothesis property tests for the observe_step consumer-ceiling arithmetic.

The example-based unit tier (``tests/unit/scripts/test_analyze_observe_step_ceiling``)
pins the closed form at known points and pins the ``imagine_step`` call count against
the shipped ``MCTSConfig`` defaults. These fuzz the invariants the Amdahl bound must
satisfy for *any* share and *any* stage speedup, because both are free inputs — a
share is whatever the rover eventually measures, and a stage speedup is whatever the
FP16 / IO-binding / engine-cache work eventually delivers:

- the bound is monotone non-decreasing in the stage speedup ``k`` (a faster stage is
  never worse end-to-end);
- the bound is monotone non-decreasing in the stage share ``s`` (the same speedup
  applied to a larger slice is never worse);
- the bound never exceeds ``1 / (1 - s)``, which is the whole point of the ceiling:
  no ``k``, however large, buys more than the serial remainder allows;
- the bound is always ``>= 1`` — accelerating a stage cannot slow the system down.

Plus the two structural invariants of the call-count derivation, which the ceiling's
low end is computed from.

Needs neither ``torch`` nor ``onnxruntime``: the script under test imports only
``mousedroid.constants`` and ``mousedroid.logging.setup`` at module scope and defers
``load_settings`` into ``resolve_config``, which nothing here calls.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._script_loader import load_script_module

_mod = load_script_module("analyze_observe_step_ceiling")

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# The full admissible share domain, both edges included: s = 0 (the stage is free)
# and s = 1 (the stage is everything) are the two cases the closed form has to
# special-case, so they must be reachable rather than assumed away.
_shares = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Stage speedups from "no change" upward. Capped well below float overflow so the
# reciprocals stay in a regime where the tolerance below is meaningful.
_stage_speedups = st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False)

_positive_counts = st.integers(min_value=1, max_value=10_000)

# Slack for one comparison of two independently-rounded reciprocals. The closed form
# is exact in the reals; in float64 two evaluations that should be equal can differ by
# a couple of ulps, so a strict `>=` would fail on arithmetic noise rather than on a
# broken invariant.
_TOLERANCE = 1e-9


def _at_least(actual: float, lower_bound: float) -> bool:
    """``actual >= lower_bound`` with float64 slack; ``inf`` compares exactly."""
    if math.isinf(actual) or math.isinf(lower_bound):
        return actual >= lower_bound
    return actual >= lower_bound - _TOLERANCE * max(1.0, abs(lower_bound))


# ---------------------------------------------------------------------------
# Amdahl bound
# ---------------------------------------------------------------------------


@given(share=_shares, speedup=_stage_speedups)
@settings(max_examples=300)
def test_bound_is_never_below_one(share: float, speedup: float) -> None:
    """Accelerating a stage cannot make the whole worse."""
    assert _at_least(_mod.amdahl_speedup(share, speedup), 1.0)


@given(share=_shares)
@settings(max_examples=100)
def test_ceiling_is_never_below_one(share: float) -> None:
    assert _at_least(_mod.amdahl_ceiling(share), 1.0)


@given(share=_shares, low=_stage_speedups, high=_stage_speedups)
@settings(max_examples=300)
def test_monotone_non_decreasing_in_stage_speedup(share: float, low: float, high: float) -> None:
    """A larger ``k`` never yields a smaller end-to-end speedup."""
    lower, upper = sorted((low, high))
    assert _at_least(_mod.amdahl_speedup(share, upper), _mod.amdahl_speedup(share, lower))


@given(low=_shares, high=_shares, speedup=_stage_speedups)
@settings(max_examples=300)
def test_monotone_non_decreasing_in_stage_share(low: float, high: float, speedup: float) -> None:
    """The same ``k`` applied to a larger slice never yields less."""
    lower, upper = sorted((low, high))
    assert _at_least(_mod.amdahl_speedup(upper, speedup), _mod.amdahl_speedup(lower, speedup))


@given(share=_shares, speedup=_stage_speedups)
@settings(max_examples=300)
def test_bound_never_exceeds_the_ceiling(share: float, speedup: float) -> None:
    """No finite ``k`` beats ``1 / (1 - s)``."""
    ceiling = _mod.amdahl_ceiling(share)
    assert _at_least(ceiling, _mod.amdahl_speedup(share, speedup))


@given(share=_shares)
@settings(max_examples=200)
def test_infinite_speedup_attains_the_ceiling(share: float) -> None:
    """``amdahl_ceiling`` is exactly the ``k -> inf`` limit, not an approximation."""
    assert _mod.amdahl_speedup(share, math.inf) == _mod.amdahl_ceiling(share)


@given(share=_shares)
@settings(max_examples=200)
def test_unit_speedup_is_the_identity(share: float) -> None:
    """``k = 1`` changes nothing, whatever the share."""
    assert _mod.amdahl_speedup(share, 1.0) == 1.0


@given(target=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_required_share_round_trips_through_the_ceiling(target: float) -> None:
    """``required_share_for_ceiling`` inverts ``amdahl_ceiling``."""
    share = _mod.required_share_for_ceiling(target)
    assert 0.0 <= share < 1.0
    assert _mod.amdahl_ceiling(share) == pytest.approx(target, rel=1e-6)


# ---------------------------------------------------------------------------
# Call-count derivation
# ---------------------------------------------------------------------------


@given(
    n_simulations=_positive_counts,
    n_action_candidates=_positive_counts,
    rollout_depth=_positive_counts,
)
@settings(max_examples=300)
def test_call_count_bracket_is_ordered_and_contains_the_rollout_leg(
    n_simulations: int, n_action_candidates: int, rollout_depth: int
) -> None:
    counts = _mod.imagine_calls_per_plan(
        n_simulations=n_simulations,
        n_action_candidates=n_action_candidates,
        rollout_depth=rollout_depth,
    )
    assert counts.rollout_calls < counts.total_min <= counts.total_max
    assert counts.total_min == counts.initial_expansion_calls + counts.rollout_calls
    assert counts.total_max == counts.total_min + counts.re_expansion_calls_max


@given(
    n_simulations=_positive_counts,
    n_action_candidates=_positive_counts,
    rollout_depth=_positive_counts,
)
@settings(max_examples=300)
def test_bracket_collapses_exactly_when_the_budget_fits_the_candidate_set(
    n_simulations: int, n_action_candidates: int, rollout_depth: int
) -> None:
    """mcts.py:318-319 can only fire once every root child has been visited once."""
    counts = _mod.imagine_calls_per_plan(
        n_simulations=n_simulations,
        n_action_candidates=n_action_candidates,
        rollout_depth=rollout_depth,
    )
    collapsed = counts.total_min == counts.total_max
    assert collapsed == (n_simulations <= n_action_candidates)


@given(
    n_simulations=_positive_counts,
    n_action_candidates=_positive_counts,
    rollout_depth=_positive_counts,
)
@settings(max_examples=300)
def test_equal_cost_share_is_a_valid_share_and_shrinks_with_the_call_count(
    n_simulations: int, n_action_candidates: int, rollout_depth: int
) -> None:
    """The sweep's derived low end must always be a usable Amdahl share."""
    counts = _mod.imagine_calls_per_plan(
        n_simulations=n_simulations,
        n_action_candidates=n_action_candidates,
        rollout_depth=rollout_depth,
    )
    ratio = _mod.call_count_ratio(counts)
    assert 0.0 < ratio.equal_cost_share_min <= ratio.equal_cost_share_max <= 1.0
    # A bigger planner leg can only dilute observe_step's structural share.
    assert _at_least(
        _mod.amdahl_ceiling(ratio.equal_cost_share_max),
        _mod.amdahl_ceiling(ratio.equal_cost_share_min),
    )


@given(
    low=st.floats(min_value=1e-6, max_value=0.4, allow_nan=False, allow_infinity=False),
    high=st.floats(min_value=0.4, max_value=1.0, allow_nan=False, allow_infinity=False),
    points=st.integers(min_value=2, max_value=32),
)
@settings(max_examples=200)
def test_sweep_is_ascending_and_inside_the_share_domain(
    low: float, high: float, points: int
) -> None:
    sweep = _mod.geometric_sweep(low, high, points)
    assert len(sweep) == points
    assert list(sweep) == sorted(sweep)
    for share in sweep:
        # Every swept value must be admissible input to the bound.
        assert _at_least(_mod.amdahl_ceiling(share), 1.0)
