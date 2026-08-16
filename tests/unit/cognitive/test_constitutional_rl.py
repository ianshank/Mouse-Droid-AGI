from __future__ import annotations

import numpy as np

from mousedroid.cognitive.constitutional_rl import (
    ConstitutionalChecker,
    ConstitutionalRLConfig,
    CuriosityAggregator,
    FlowCalculator,
)


def test_checker_constructor_default() -> None:
    checker = ConstitutionalChecker()
    assert checker is not None


def test_checker_constructor_custom_config() -> None:
    cfg = ConstitutionalRLConfig(speed_ceiling_mps=1.0)
    checker = ConstitutionalChecker(cfg)
    assert checker is not None


def test_check_returns_tuple() -> None:
    checker = ConstitutionalChecker()
    action = np.array([0.1, 0.0], dtype=np.float64)
    safe_action, violations = checker.check(action, {})
    assert isinstance(safe_action, np.ndarray)
    assert isinstance(violations, list)


def test_check_empty_violations_when_safe() -> None:
    checker = ConstitutionalChecker()
    action = np.array([0.1, 0.0], dtype=np.float64)
    _, violations = checker.check(action, {"battery_v": 12.0, "obstacle_dist_m": 5.0})
    assert violations == []


def test_check_speed_violation() -> None:
    checker = ConstitutionalChecker(ConstitutionalRLConfig(speed_ceiling_mps=0.5))
    action = np.array([1.0, 0.0], dtype=np.float64)
    safe_action, violations = checker.check(action, {})
    assert len(violations) > 0
    assert abs(safe_action[0]) <= 0.5


def test_check_battery_violation() -> None:
    checker = ConstitutionalChecker()
    action = np.array([0.3, 0.1], dtype=np.float64)
    _, violations = checker.check(action, {"battery_v": 5.0})
    assert any("battery" in v for v in violations)


def test_check_obstacle_violation() -> None:
    checker = ConstitutionalChecker()
    action = np.array([0.3, 0.1], dtype=np.float64)
    _safe_action, violations = checker.check(action, {"obstacle_dist_m": 0.05})
    assert any("obstacle" in v for v in violations)


def test_flow_calculator_zero_width_zero_value() -> None:
    fc = FlowCalculator()
    assert fc.compute(0.0, 0.0) == 1.0


def test_flow_calculator_zero_width_nonzero_value() -> None:
    fc = FlowCalculator()
    assert fc.compute(0.0, 1.0) == 0.0


def test_flow_calculator_normal_case() -> None:
    fc = FlowCalculator()
    score = fc.compute(10.0, 3.0)
    assert 0.0 <= score <= 1.0
    assert abs(score - 0.7) < 1e-6


def test_curiosity_aggregator_uniform() -> None:
    agg = CuriosityAggregator()
    scores = {"social": 0.5, "epistemic": 0.5, "perceptual": 0.5, "metacognitive": 0.5}
    result = agg.aggregate(scores)
    assert abs(result - 0.5) < 1e-6


def test_curiosity_aggregator_missing_channels() -> None:
    agg = CuriosityAggregator()
    scores = {"social": 1.0}
    result = agg.aggregate(scores)
    assert 0.0 <= result <= 1.0
