"""Tests for symbolic planner (Protocol backends + recursive fallback)."""

from __future__ import annotations

import multiprocessing
import time
from unittest.mock import patch

import pytest

from mousedroid.arm.planning.pddl_domain import optimal_move_count
from mousedroid.arm.planning.symbolic_planner import (
    PlanningError,
    PyperplanBackend,
    RecursiveBackend,
    SymbolicPlanner,
    make_primary_backend,
    parse_solution,
    run_pyperplan_subprocess,
    solve_hanoi,
)
from mousedroid.arm.protocols import PlanStep, SymbolicPlannerBackend, SymbolicState
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


def _steps_runner(*ops: str) -> object:
    """Return a SearchRunner that yields the given operator string reprs."""

    def _runner(_domain: str, _problem: str, _timeout_s: float) -> list[str]:
        return list(ops)

    return _runner


class TestPureHelpers:
    """Module-level pure helpers shared by the backends + planner."""

    def test_parse_solution_parses_actions(self) -> None:
        steps = parse_solution(["(move disk_1 peg_A peg_C)", "(move disk_2 peg_A peg_B)"])
        assert [s.action for s in steps] == ["move", "move"]
        assert steps[0].args == ["disk_1", "peg_A", "peg_C"]

    def test_parse_solution_skips_blank_lines(self) -> None:
        assert len(parse_solution(["(move disk_1 peg_A peg_C)", "", "   "])) == 1

    def test_solve_hanoi_optimal_counts(self) -> None:
        for n in (1, 3, 5):
            assert len(solve_hanoi(n, 3)) == optimal_move_count(n)


class TestRecursiveBackend:
    """RecursiveBackend is total — always returns an optimal plan."""

    def test_returns_optimal_plan(self) -> None:
        backend = RecursiveBackend(ArmTaskConfig(num_disks=3, num_pegs=3))
        steps = backend.search("(domain)", "(problem)")
        assert steps is not None
        assert len(steps) == optimal_move_count(3)

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(RecursiveBackend(ArmTaskConfig()), SymbolicPlannerBackend)


class TestPyperplanBackendRunnerInjection:
    """PyperplanBackend delegates to an injected runner (no real subprocess)."""

    def test_parses_runner_operator_strings(self) -> None:
        backend = PyperplanBackend(
            ArmPlanningConfig(),
            runner=_steps_runner("(move disk_1 peg_A peg_C)", "(move disk_2 peg_A peg_B)"),
        )
        steps = backend.search("(domain)", "(problem)")
        assert steps is not None
        assert [s.args for s in steps] == [
            ["disk_1", "peg_A", "peg_C"],
            ["disk_2", "peg_A", "peg_B"],
        ]

    def test_returns_none_when_runner_returns_none(self) -> None:
        backend = PyperplanBackend(ArmPlanningConfig(), runner=lambda *_: None)
        assert backend.search("(domain)", "(problem)") is None

    def test_returns_none_when_runner_raises(self) -> None:
        def _boom(*_args: object) -> list[str]:
            raise RuntimeError("runner exploded")

        backend = PyperplanBackend(ArmPlanningConfig(), runner=_boom)
        assert backend.search("(domain)", "(problem)") is None

    def test_passes_configured_timeout_to_runner(self) -> None:
        seen: dict[str, float] = {}

        def _runner(_domain: str, _problem: str, timeout_s: float) -> None:
            seen["timeout"] = timeout_s
            return None

        backend = PyperplanBackend(ArmPlanningConfig(planning_timeout_s=1.25), runner=_runner)
        backend.search("(domain)", "(problem)")
        assert seen["timeout"] == 1.25

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(PyperplanBackend(ArmPlanningConfig()), SymbolicPlannerBackend)


class TestSymbolicPlannerOrchestration:
    """SymbolicPlanner tries the primary backend, then the guaranteed fallback."""

    def test_uses_primary_when_it_returns_a_plan(self) -> None:
        primary = PyperplanBackend(
            ArmPlanningConfig(), runner=_steps_runner("(move d1 peg_A peg_B)")
        )
        fallback_calls: list[int] = []

        class _CountingFallback:
            def search(self, _d: str, _p: str) -> list[PlanStep] | None:
                fallback_calls.append(1)
                return solve_hanoi(2, 3)

        planner = SymbolicPlanner(
            ArmPlanningConfig(),
            ArmTaskConfig(num_disks=2, num_pegs=3),
            primary_backend=primary,
            fallback_backend=_CountingFallback(),
        )
        steps = planner.plan(_make_initial_state(), _make_goal_state())
        assert len(steps) == 1
        assert fallback_calls == []  # fallback untouched when primary succeeds

    def test_falls_back_when_primary_returns_none(self) -> None:
        primary = PyperplanBackend(ArmPlanningConfig(), runner=lambda *_: None)
        planner = SymbolicPlanner(
            ArmPlanningConfig(),
            ArmTaskConfig(num_disks=2, num_pegs=3),
            primary_backend=primary,
        )
        steps = planner.plan(_make_initial_state(), _make_goal_state())
        assert len(steps) == optimal_move_count(2)  # recursive fallback plan

    def test_default_backends_produce_a_plan(self) -> None:
        """A default planner always returns a plan (pyperplan absent → fallback)."""
        planner = _make_planner(num_disks=3)
        steps = planner.plan(_make_initial_state(), _make_goal_state())
        assert len(steps) > 0

    def test_wraps_backend_exception_in_planning_error(self) -> None:
        """A non-fallback backend raising is wrapped as PlanningError by plan()."""

        class _RaisingBackend:
            def search(self, _d: str, _p: str) -> list[PlanStep] | None:
                raise RuntimeError("catastrophic")

        planner = SymbolicPlanner(
            ArmPlanningConfig(),
            ArmTaskConfig(num_disks=2, num_pegs=3),
            primary_backend=_RaisingBackend(),
        )
        with pytest.raises(PlanningError, match="Planning failed"):
            planner.plan(_make_initial_state(), _make_goal_state())

    def test_planning_error_when_fallback_returns_none(self) -> None:
        """If even the fallback yields no plan, plan() raises PlanningError."""

        class _NoneBackend:
            def search(self, _d: str, _p: str) -> list[PlanStep] | None:
                return None

        planner = SymbolicPlanner(
            ArmPlanningConfig(),
            ArmTaskConfig(num_disks=2, num_pegs=3),
            primary_backend=_NoneBackend(),
            fallback_backend=_NoneBackend(),
        )
        with pytest.raises(PlanningError, match="fallback backend returned no plan"):
            planner.plan(_make_initial_state(), _make_goal_state())


class TestMakePrimaryBackend:
    """Backend selection from planner_backend (single source of truth)."""

    def test_pyperplan_selected(self) -> None:
        backend = make_primary_backend(
            ArmPlanningConfig(planner_backend="pyperplan"), ArmTaskConfig()
        )
        assert isinstance(backend, PyperplanBackend)

    def test_recursive_selected(self) -> None:
        backend = make_primary_backend(
            ArmPlanningConfig(planner_backend="recursive"), ArmTaskConfig()
        )
        assert isinstance(backend, RecursiveBackend)

    def test_fast_downward_falls_through_to_pyperplan(self) -> None:
        backend = make_primary_backend(
            ArmPlanningConfig(planner_backend="fast_downward"), ArmTaskConfig()
        )
        assert isinstance(backend, PyperplanBackend)


class TestRunPyperplanSubprocess:
    """The default subprocess runner — availability probe + hard interrupt."""

    _SEAM = "mousedroid.arm.planning.symbolic_planner._import_search_plan"

    def test_unavailable_returns_none_without_spawning(self) -> None:
        with patch(self._SEAM, side_effect=ImportError("No module named 'pyperplan'")):
            assert run_pyperplan_subprocess("(domain)", "(problem)", 1.0) is None

    def test_api_drift_returns_none(self) -> None:
        with patch(self._SEAM, side_effect=AttributeError("search_plan gone")):
            assert run_pyperplan_subprocess("(domain)", "(problem)", 1.0) is None

    def test_collect_result_empty_queue_returns_none(self) -> None:
        """A worker that died without putting a result degrades to the fallback."""
        import queue as _q

        from mousedroid.arm.planning.symbolic_planner import _collect_pyperplan_result

        class _EmptyQueue:
            def get(self, timeout: float) -> object:
                raise _q.Empty

        assert _collect_pyperplan_result(_EmptyQueue(), time.monotonic()) is None

    def test_hard_terminates_on_timeout(self) -> None:
        """A worker that outlives the budget is terminate()-d, not orphaned.

        Patches the seam with a sleeping search_plan; under the ``fork`` start
        method the child inherits the patch, so this exercises the real
        terminate path WITHOUT needing pyperplan installed.
        """
        if multiprocessing.get_start_method() != "fork":
            pytest.skip("hard-interrupt test relies on fork inheriting the patched seam")

        def _sleepy_search(*_args: object, **_kwargs: object) -> None:
            time.sleep(30.0)  # far past the timeout; killed before it returns

        with patch(self._SEAM, return_value=(_sleepy_search, object(), object)):
            t0 = time.monotonic()
            result = run_pyperplan_subprocess("(domain)", "(problem)", 0.2)
            elapsed = time.monotonic() - t0

        assert result is None
        assert elapsed < 10.0, "worker was not hard-terminated near the timeout budget"


# Valid minimal STRIPS PDDL pyperplan can solve in one step — used to exercise
# the REAL subprocess end-to-end. Zero-parameter actions still need ``:parameters ()``.
_TOGGLE_DOMAIN = (
    "(define (domain toggle) (:requirements :strips) (:predicates (off) (on)) "
    "(:action flip :parameters () :precondition (off) :effect (and (on) (not (off)))))"
)
_TOGGLE_PROBLEM = "(define (problem p) (:domain toggle) (:init (off)) (:goal (on)))"


class TestRealPyperplanSubprocess:
    """End-to-end subprocess exercise against the real solver (skips w/o pyperplan)."""

    def test_solvable_problem_returns_operator_strings(self) -> None:
        pytest.importorskip("pyperplan")
        result = run_pyperplan_subprocess(_TOGGLE_DOMAIN, _TOGGLE_PROBLEM, 5.0)
        assert result is not None
        assert len(result) >= 1
        assert all(isinstance(line, str) for line in result)

    def test_parse_error_returns_none_gracefully(self) -> None:
        pytest.importorskip("pyperplan")
        assert run_pyperplan_subprocess("(not valid pddl", "(nope", 5.0) is None


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
