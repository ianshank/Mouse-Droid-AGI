"""Tests for symbolic planner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


class TestPyperplanIntegration:
    """Test _solve_pddl and _parse_solution against the pyperplan seam.

    The previous version of these tests patched ``sys.modules["pyperplan"]``
    to fake a ``pyperplan.solve(...)`` call that never existed. The real
    pyperplan (>=2.0) exposes ``pyperplan.planner.search_plan`` via a tiny
    seam (``_import_search_plan``) in the planner module. These tests patch
    that seam directly, which is both robust to package-layout changes and
    independent of import order.
    """

    _SEAM = "mousedroid.arm.planning.symbolic_planner._import_search_plan"

    def test_solve_pddl_with_mocked_pyperplan(self) -> None:
        planner = _make_planner(num_disks=2)

        mock_search_plan = MagicMock(
            return_value=[
                "(move disk_1 peg_A peg_C)",
                "(move disk_2 peg_A peg_B)",
            ]
        )
        with patch(
            self._SEAM,
            return_value=(mock_search_plan, MagicMock(), MagicMock()),
        ):
            steps = planner._solve_pddl("(domain)", "(problem)")

        assert len(steps) == 2
        assert steps[0].action == "move"
        assert steps[0].args == ["disk_1", "peg_A", "peg_C"]
        assert steps[1].args == ["disk_2", "peg_A", "peg_B"]
        mock_search_plan.assert_called_once()

    def test_solve_pddl_falls_back_when_pyperplan_returns_none(self) -> None:
        """When pyperplan returns no solution we fall back to the recursive
        solver instead of raising — empty :init fixtures are not a failure.
        """
        planner = _make_planner(num_disks=2)

        mock_search_plan = MagicMock(return_value=None)
        with patch(
            self._SEAM,
            return_value=(mock_search_plan, MagicMock(), MagicMock()),
        ):
            steps = planner._solve_pddl("(domain)", "(problem)")

        # 2 disks ⇒ optimal recursive plan has 2^2 - 1 = 3 moves.
        assert len(steps) == optimal_move_count(2)
        mock_search_plan.assert_called_once()

    def test_solve_pddl_falls_back_when_seam_import_fails(self) -> None:
        """If pyperplan's public surface shifted, fall back gracefully."""
        planner = _make_planner(num_disks=3)
        with patch(self._SEAM, side_effect=AttributeError("solve gone")):
            steps = planner._solve_pddl("(domain)", "(problem)")
        assert len(steps) == optimal_move_count(3)

    def test_solve_pddl_falls_back_when_pyperplan_not_installed(self) -> None:
        """``ImportError`` (and its subclass ``ModuleNotFoundError``) route to
        the recursive solver — this is the most likely failure on a CI host
        that doesn't ship pyperplan in its base image.
        """
        planner = _make_planner(num_disks=4)
        with patch(self._SEAM, side_effect=ImportError("No module named 'pyperplan'")):
            steps = planner._solve_pddl("(domain)", "(problem)")
        assert len(steps) == optimal_move_count(4)

    def test_solve_pddl_falls_back_when_module_not_found(self) -> None:
        """``ModuleNotFoundError`` is a subclass of ``ImportError``; exercising
        it explicitly documents the intent and guards against any future
        refactor that narrows the exception tuple.
        """
        planner = _make_planner(num_disks=2)
        err = ModuleNotFoundError("No module named 'pyperplan.search'")
        with patch(self._SEAM, side_effect=err):
            steps = planner._solve_pddl("(domain)", "(problem)")
        assert len(steps) == optimal_move_count(2)

    def test_solve_pddl_converts_operator_objects_to_strings(self) -> None:
        """``search_plan`` returns ``Operator`` objects, not strings — the
        seam converts each via ``str(op)``. Verifying the end-to-end
        conversion with a mock that has a custom ``__str__`` confirms the
        conversion path that the bare-string tests do not exercise.
        """

        class _FakeOperator:
            """Mimics pyperplan's ``Operator`` __str__: ``(action arg1 arg2 ...)``."""

            def __init__(self, action: str, args: tuple[str, ...]) -> None:
                self._action = action
                self._args = args

            def __str__(self) -> str:
                return f"({self._action} {' '.join(self._args)})"

        planner = _make_planner(num_disks=2)
        fake_solution = [
            _FakeOperator("move", ("disk_1", "peg_A", "peg_C")),
            _FakeOperator("move", ("disk_2", "peg_A", "peg_B")),
        ]
        mock_search_plan = MagicMock(return_value=fake_solution)
        with patch(
            self._SEAM,
            return_value=(mock_search_plan, MagicMock(), MagicMock()),
        ):
            steps = planner._solve_pddl("(domain)", "(problem)")

        assert len(steps) == 2
        assert steps[0].action == "move"
        assert steps[0].args == ["disk_1", "peg_A", "peg_C"]
        assert steps[1].args == ["disk_2", "peg_A", "peg_B"]

    def test_plan_uses_pyperplan_when_available(self) -> None:
        planner = _make_planner(num_disks=2)

        mock_search_plan = MagicMock(return_value=["(move disk_1 peg_A peg_B)"])
        with patch(
            self._SEAM,
            return_value=(mock_search_plan, MagicMock(), MagicMock()),
        ):
            steps = planner.plan(_make_initial_state(), _make_goal_state())

        assert len(steps) == 1
        mock_search_plan.assert_called_once()

    def test_solve_pddl_falls_back_when_pyperplan_exceeds_timeout(self) -> None:
        """A pyperplan call that exceeds ``planning_timeout_s`` is interrupted
        (best-effort) and the recursive solver takes over. This protects the
        sense-plan-act loop from a malformed PDDL that sends astar into an
        unbounded search.
        """
        # 2-disk config with an aggressive 100ms timeout — easy to trip with
        # a deliberately-slow mock without slowing the test.
        planning_cfg = ArmPlanningConfig(planning_timeout_s=0.1)
        task_cfg = ArmTaskConfig(num_disks=2, num_pegs=3)
        planner = SymbolicPlanner(planning_cfg, task_cfg)

        def _slow_search(*_args: object, **_kwargs: object) -> None:
            import time as _time

            _time.sleep(2.0)  # well past the 100ms budget
            return None  # never reached by the caller — future.result() times out first

        slow_mock = MagicMock(side_effect=_slow_search)
        with patch(
            self._SEAM,
            return_value=(slow_mock, MagicMock(), MagicMock()),
        ):
            steps = planner._solve_pddl("(domain)", "(problem)")

        # Recursive fallback produced the plan — caller never sees the hang.
        assert len(steps) == optimal_move_count(2)
        slow_mock.assert_called_once()

    def test_plan_falls_back_when_pyperplan_raises(self) -> None:
        """A pyperplan internal crash (parse error, search blowup, ...) routes
        to the deterministic recursive solver. Upstream callers see a successful
        plan rather than a PlanningError — that is the contract the replanner +
        BDI loop rely on. (Previously this test asserted exception wrapping,
        but with the recursive fallback in place that contract is obsolete.)
        """
        planner = _make_planner(num_disks=2)

        mock_search_plan = MagicMock(side_effect=RuntimeError("solver crash"))
        with patch(
            self._SEAM,
            return_value=(mock_search_plan, MagicMock(), MagicMock()),
        ):
            steps = planner.plan(_make_initial_state(), _make_goal_state())

        assert len(steps) == optimal_move_count(2)
        mock_search_plan.assert_called_once()


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
