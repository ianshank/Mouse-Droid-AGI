"""Tests for PDDL domain and problem generation."""

from __future__ import annotations

from mousedroid.arm.planning.pddl_domain import (
    generate_domain,
    generate_problem,
    optimal_move_count,
)
from mousedroid.config.schema import ArmTaskConfig


class TestOptimalMoveCount:
    """Test optimal move count calculation."""

    def test_1_disk(self) -> None:
        assert optimal_move_count(1) == 1

    def test_3_disks(self) -> None:
        assert optimal_move_count(3) == 7

    def test_5_disks(self) -> None:
        assert optimal_move_count(5) == 31

    def test_7_disks(self) -> None:
        assert optimal_move_count(7) == 127

    def test_10_disks(self) -> None:
        assert optimal_move_count(10) == 1023


class TestGenerateDomain:
    """Test PDDL domain generation."""

    def test_domain_contains_move_action(self) -> None:
        domain = generate_domain()
        assert ":action move" in domain

    def test_domain_contains_predicates(self) -> None:
        domain = generate_domain()
        assert "(on ?x ?y)" in domain
        assert "(clear ?x)" in domain
        assert "(smaller ?x ?y)" in domain
        assert "(disk ?x)" in domain
        assert "(peg ?x)" in domain

    def test_domain_contains_requirements(self) -> None:
        domain = generate_domain()
        assert ":requirements :strips" in domain


class TestGenerateProblem:
    """Test PDDL problem generation."""

    def test_3_disk_problem(self) -> None:
        cfg = ArmTaskConfig(num_disks=3, num_pegs=3)
        problem = generate_problem(cfg)
        assert "disk_1" in problem
        assert "disk_2" in problem
        assert "disk_3" in problem
        assert "peg_A" in problem
        assert "peg_B" in problem
        assert "peg_C" in problem
        assert ":domain tower-of-hanoi" in problem

    def test_5_disk_problem(self) -> None:
        cfg = ArmTaskConfig(
            num_disks=5,
            num_pegs=3,
            peg_positions=[[0.2, 0, 0], [0.3, 0, 0], [0.4, 0, 0]],
        )
        problem = generate_problem(cfg)
        assert "disk_5" in problem

    def test_problem_contains_initial_state(self) -> None:
        cfg = ArmTaskConfig(num_disks=3, num_pegs=3)
        problem = generate_problem(cfg)
        assert ":init" in problem
        assert "(on disk_3 peg_A)" in problem  # Bottom disk on first peg

    def test_problem_contains_goal(self) -> None:
        cfg = ArmTaskConfig(num_disks=3, num_pegs=3)
        problem = generate_problem(cfg)
        assert ":goal" in problem
        assert "(on disk_3 peg_C)" in problem  # All disks on last peg

    def test_problem_has_smaller_predicates(self) -> None:
        cfg = ArmTaskConfig(num_disks=3, num_pegs=3)
        problem = generate_problem(cfg)
        assert "(smaller disk_1 disk_2)" in problem
        assert "(smaller disk_1 peg_A)" in problem
