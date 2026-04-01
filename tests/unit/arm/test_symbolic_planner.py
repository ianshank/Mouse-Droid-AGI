"""Tests for symbolic planner."""

from __future__ import annotations

from mousedroid.arm.planning.pddl_domain import optimal_move_count
from mousedroid.arm.planning.symbolic_planner import SymbolicPlanner
from mousedroid.arm.protocols import SymbolicState
from mousedroid.config.schema import ArmPlanningConfig, ArmTaskConfig


def _make_planner(num_disks: int = 3) -> SymbolicPlanner:
    """Create a planner with default configs."""
    planning_cfg = ArmPlanningConfig()
    task_cfg = ArmTaskConfig(num_disks=num_disks, num_pegs=3)
    return SymbolicPlanner(planning_cfg, task_cfg)


def _make_initial_state() -> SymbolicState:
    """Create dummy initial state."""
    return SymbolicState(predicates=frozenset(), objects={})


def _make_goal_state() -> SymbolicState:
    """Create dummy goal state."""
    return SymbolicState(predicates=frozenset(), objects={})


class TestRecursiveSolver:
    """Test the recursive fallback solver (guaranteed optimal)."""

    def test_1_disk_produces_1_move(self) -> None:
        planner = _make_planner(num_disks=1)
        steps = planner._solve_recursive()
        assert len(steps) == 1
        assert steps[0].action == "move"

    def test_3_disks_produces_7_moves(self) -> None:
        planner = _make_planner(num_disks=3)
        steps = planner._solve_recursive()
        assert len(steps) == optimal_move_count(3)

    def test_5_disks_produces_31_moves(self) -> None:
        planner = _make_planner(num_disks=5)
        steps = planner._solve_recursive()
        assert len(steps) == optimal_move_count(5)

    def test_moves_reference_correct_pegs(self) -> None:
        planner = _make_planner(num_disks=3)
        steps = planner._solve_recursive()
        peg_names = {"peg_A", "peg_B", "peg_C"}
        for step in steps:
            assert step.args[1] in peg_names  # source peg
            assert step.args[2] in peg_names  # target peg

    def test_no_disk_moved_from_same_to_same(self) -> None:
        planner = _make_planner(num_disks=3)
        steps = planner._solve_recursive()
        for step in steps:
            assert step.args[1] != step.args[2]  # source != target


class TestSymbolicPlannerPlan:
    """Test the plan() method."""

    def test_plan_returns_steps(self) -> None:
        planner = _make_planner(num_disks=3)
        steps = planner.plan(_make_initial_state(), _make_goal_state())
        assert len(steps) > 0

    def test_replan_returns_steps(self) -> None:
        planner = _make_planner(num_disks=3)
        steps = planner.replan(_make_initial_state(), _make_goal_state(), "test error")
        assert len(steps) > 0
