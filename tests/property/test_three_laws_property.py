"""Property-based tests for Three Laws invariants using Hypothesis."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.safety.three_laws import RoboticsLaw, RoboticsLawChecker


def _make_checker() -> RoboticsLawChecker:
    return RoboticsLawChecker()


@given(
    speed=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    human_dist=st.floats(min_value=0.01, max_value=0.4, allow_nan=False),
)
@settings(max_examples=50)
def test_law1_never_allows_harm(speed: float, human_dist: float) -> None:
    """When a human is close, forward speed must be zero or negative."""
    checker = _make_checker()
    action = np.array([speed, 0.0], dtype=np.float64)
    ctx = {"human_detected": True, "human_dist_m": human_dist}
    safe, _ = checker.check(action, ctx)
    if speed > 0.05:  # Only check if original action was forward motion
        assert safe[0] <= 0.05, f"Forward motion {safe[0]} near human at {human_dist}m"


@given(
    speed=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    human_dist=st.floats(min_value=0.01, max_value=0.3, allow_nan=False),
    battery=st.floats(min_value=5.0, max_value=15.0, allow_nan=False),
)
@settings(max_examples=50)
def test_law_priority_invariant(
    speed: float,
    human_dist: float,
    battery: float,
) -> None:
    """Law 1 override always dominates Laws 2 and 3."""
    checker = _make_checker()
    action = np.array([speed, 0.0], dtype=np.float64)
    ctx = {
        "human_detected": True,
        "human_dist_m": human_dist,
        "battery_v": battery,
        "commanded_action": np.array([1.0, 0.0]),
    }
    _, violations = checker.check(action, ctx)

    law_nums = [v.law for v in violations]
    if RoboticsLaw.FIRST in law_nums:
        # If Law 1 triggered, it must be first
        assert violations[0].law == RoboticsLaw.FIRST


@given(
    action=st.lists(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
        min_size=1,
        max_size=3,
    ),
)
@settings(max_examples=50)
def test_safe_action_always_returned(action: list[float]) -> None:
    """Output is always a valid numpy array."""
    checker = _make_checker()
    arr = np.array(action, dtype=np.float64)
    safe, _ = checker.check(arr, {})
    assert isinstance(safe, np.ndarray)
    assert safe.shape == arr.shape
    assert np.all(np.isfinite(safe))


@given(
    action=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=3,
    ),
)
@settings(max_examples=50)
def test_violations_always_list(action: list[float]) -> None:
    """Output violations is always a list."""
    checker = _make_checker()
    arr = np.array(action, dtype=np.float64)
    _, violations = checker.check(arr, {})
    assert isinstance(violations, list)


@given(
    speed=st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
    human_dist=st.floats(min_value=0.01, max_value=0.4, allow_nan=False),
)
@settings(max_examples=50)
def test_law3_never_overrides_law1(speed: float, human_dist: float) -> None:
    """Self-preservation must never cause forward motion toward a human."""
    checker = _make_checker()
    action = np.array([speed, 0.0], dtype=np.float64)
    ctx = {
        "human_detected": True,
        "human_dist_m": human_dist,
        "battery_v": 5.0,  # Law 3 would want to conserve
    }
    safe, _ = checker.check(action, ctx)
    assert safe[0] <= 0.05, "Law 3 must not cause forward motion toward human"


@given(
    speed=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=30)
def test_all_laws_idempotent(speed: float) -> None:
    """Checking a safe action twice gives the same result."""
    checker = _make_checker()
    action = np.array([speed, 0.0], dtype=np.float64)
    ctx = {"battery_v": 12.0, "obstacle_dist_m": 2.0}

    safe1, _v1 = checker.check(action, ctx)
    safe2, _v2 = checker.check(safe1, ctx)

    np.testing.assert_allclose(safe1, safe2, atol=1e-10)
