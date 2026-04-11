"""Tests for symbolic planner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mousedroid.arm.planning.pddl_domain import optimal_move_count
from mousedroid.arm.planning.symbolic_planner import PlanningError, SymbolicPlanner
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


class TestPyperplanIntegration:
    """Test _solve_pddl and _parse_solution with mocked pyperplan."""

    def test_solve_pddl_with_mocked_pyperplan(self) -> None:
        planner = _make_planner(num_disks=2)

        mock_pyperplan = MagicMock()
        mock_pyperplan.solve.return_value = [
            "(move disk_1 peg_A peg_C)",
            "(move disk_2 peg_A peg_B)",
        ]

        with patch.dict("sys.modules", {"pyperplan": mock_pyperplan}):
            steps = planner._solve_pddl("(domain)", "(problem)")

        assert len(steps) == 2
        assert steps[0].action == "move"
        assert steps[0].args == ["disk_1", "peg_A", "peg_C"]
        assert steps[1].args == ["disk_2", "peg_A", "peg_B"]

    def test_solve_pddl_pyperplan_returns_none(self) -> None:
        from unittest.mock import MagicMock, patch

        planner = _make_planner(num_disks=2)

        mock_pyperplan = MagicMock()
        mock_pyperplan.solve.return_value = None

        with (
            patch.dict("sys.modules", {"pyperplan": mock_pyperplan}),
            pytest.raises(PlanningError, match="no solution"),
        ):
            planner._solve_pddl("(domain)", "(problem)")

    def test_plan_uses_pyperplan_when_available(self) -> None:
        planner = _make_planner(num_disks=2)

        mock_pyperplan = MagicMock()
        mock_pyperplan.solve.return_value = [
            "(move disk_1 peg_A peg_B)",
        ]

        with patch.dict("sys.modules", {"pyperplan": mock_pyperplan}):
            steps = planner.plan(_make_initial_state(), _make_goal_state())

        assert len(steps) == 1
        mock_pyperplan.solve.assert_called_once()

    def test_plan_wraps_pyperplan_exception(self) -> None:
        from unittest.mock import MagicMock, patch

        planner = _make_planner(num_disks=2)

        mock_pyperplan = MagicMock()
        mock_pyperplan.solve.side_effect = RuntimeError("solver crash")

        with (
            patch.dict("sys.modules", {"pyperplan": mock_pyperplan}),
            pytest.raises(PlanningError, match="Planning failed"),
        ):
            planner.plan(_make_initial_state(), _make_goal_state())


class TestParseSolution:
    """Test _parse_solution directly."""

    def test_parses_pddl_actions(self) -> None:
        planner = _make_planner(num_disks=2)
        solution = [
            "(move disk_1 peg_A peg_C)",
            "(move disk_2 peg_A peg_B)",
            "(move disk_1 peg_C peg_B)",
        ]
        steps = planner._parse_solution(solution)
        assert len(steps) == 3
        assert steps[0].action == "move"
        assert steps[2].args == ["disk_1", "peg_C", "peg_B"]

    def test_handles_empty_lines(self) -> None:
        planner = _make_planner(num_disks=2)
        solution = ["(move disk_1 peg_A peg_C)", "", "  "]
        steps = planner._parse_solution(solution)
        assert len(steps) == 1

    def test_handles_single_action(self) -> None:
        planner = _make_planner(num_disks=1)
        solution = ["(move disk_1 peg_A peg_C)"]
        steps = planner._parse_solution(solution)
        assert len(steps) == 1
        assert steps[0].action == "move"
