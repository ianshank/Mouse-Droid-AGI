"""Property-based tests for arm modules using Hypothesis."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.arm.environments.reward_shaping import RewardShaper
from mousedroid.arm.planning.pddl_domain import generate_problem, optimal_move_count
from mousedroid.config.schema import ArmTaskConfig, ArmTrainingConfig


class TestOptimalMoveCountProperties:
    """Property: optimal_move_count(n) == 2^n - 1 for all valid n."""

    @given(n=st.integers(min_value=1, max_value=20))
    def test_formula_matches(self, n: int) -> None:
        assert optimal_move_count(n) == (2**n) - 1

    @given(n=st.integers(min_value=1, max_value=20))
    def test_always_odd(self, n: int) -> None:
        assert optimal_move_count(n) % 2 == 1

    @given(n=st.integers(min_value=2, max_value=20))
    def test_monotonically_increasing(self, n: int) -> None:
        assert optimal_move_count(n) > optimal_move_count(n - 1)


class TestPDDLGenerationProperties:
    """Property: PDDL problem generation is valid for any disk count."""

    @given(n=st.integers(min_value=1, max_value=8))
    @settings(max_examples=20)
    def test_problem_contains_all_disks(self, n: int) -> None:
        positions = [[0.2 + i * 0.1, 0.0, 0.0] for i in range(3)]
        cfg = ArmTaskConfig(num_disks=n, num_pegs=3, peg_positions=positions)
        problem = generate_problem(cfg)
        for i in range(1, n + 1):
            assert f"disk_{i}" in problem

    @given(n=st.integers(min_value=1, max_value=8))
    @settings(max_examples=20)
    def test_problem_is_valid_pddl(self, n: int) -> None:
        positions = [[0.2 + i * 0.1, 0.0, 0.0] for i in range(3)]
        cfg = ArmTaskConfig(num_disks=n, num_pegs=3, peg_positions=positions)
        problem = generate_problem(cfg)
        # Basic PDDL structure checks
        assert "(define (problem" in problem
        assert ":domain tower-of-hanoi" in problem
        assert ":objects" in problem
        assert ":init" in problem
        assert ":goal" in problem


class TestRewardBoundedProperties:
    """Property: reward values are always bounded."""

    @given(
        achieved=st.lists(st.floats(-10, 10), min_size=3, max_size=3),
        desired=st.lists(st.floats(-10, 10), min_size=3, max_size=3),
    )
    @settings(max_examples=50)
    def test_reward_is_finite(self, achieved: list[float], desired: list[float]) -> None:
        shaper = RewardShaper(ArmTrainingConfig())
        a = np.array(achieved, dtype=np.float64)
        d = np.array(desired, dtype=np.float64)
        reward = shaper.compute(a, d, {})
        assert np.isfinite(reward)

    @given(
        is_success=st.booleans(),
        collision=st.booleans(),
        grasp=st.booleans(),
    )
    def test_reward_bounded_range(self, is_success: bool, collision: bool, grasp: bool) -> None:
        shaper = RewardShaper(ArmTrainingConfig())
        info = {
            "is_success": is_success,
            "collision": collision,
            "grasp_success": grasp,
        }
        reward = shaper.compute(np.zeros(3), np.zeros(3), info)
        # Max possible: complete + grasp + place = 1.0 + 0.1 + 0.2 = 1.3
        # Min possible: collision + wrong_disk = -0.5 + -0.1 = -0.6
        assert -10.0 < reward < 10.0
