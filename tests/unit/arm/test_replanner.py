"""Tests for adaptive replanner."""

from __future__ import annotations

import pytest

from mousedroid.arm.planning.replanner import Replanner, ReplanningExhaustedError
from mousedroid.arm.planning.symbolic_planner import SymbolicPlanner
from mousedroid.arm.protocols import PlanStep, SymbolicState
from mousedroid.config.schema import ArmPlanningConfig, ArmTaskConfig


def _make_state(predicates: set[str] | None = None) -> SymbolicState:
    return SymbolicState(
        predicates=frozenset(predicates or {"on(disk_1, peg_A)"}),
        objects={"disk_1": "disk", "peg_A": "peg"},
    )


def _make_replanner(max_attempts: int = 3, llm_enabled: bool = False) -> Replanner:
    planning_cfg = ArmPlanningConfig(
        max_replan_attempts=max_attempts,
        llm_replanner_enabled=llm_enabled,
    )
    task_cfg = ArmTaskConfig(num_disks=2, num_pegs=3)
    planner = SymbolicPlanner(planning_cfg, task_cfg)
    return Replanner(planning_cfg, planner)


class TestReplanner:
    """Tests for Replanner."""

    def test_replan_returns_steps(self) -> None:
        replanner = _make_replanner()
        initial = _make_state()
        goal = _make_state({"on(disk_1, peg_C)"})
        steps = replanner.replan(initial, goal, "grasp failed")
        assert len(steps) > 0
        assert all(isinstance(s, PlanStep) for s in steps)

    def test_replan_exhausted_raises(self) -> None:
        replanner = _make_replanner(max_attempts=1)
        state = _make_state()
        goal = _make_state({"on(disk_1, peg_C)"})
        replanner.replan(state, goal, "error 1")  # attempt 1 OK
        with pytest.raises(ReplanningExhaustedError):
            replanner.replan(state, goal, "error 2")  # attempt 2 > max

    def test_reset_clears_attempts(self) -> None:
        replanner = _make_replanner(max_attempts=1)
        state = _make_state()
        goal = _make_state({"on(disk_1, peg_C)"})
        replanner.replan(state, goal, "error 1")
        replanner.reset()
        # After reset, should work again
        steps = replanner.replan(state, goal, "error 2")
        assert len(steps) > 0

    def test_llm_path_falls_back_to_symbolic(self) -> None:
        replanner = _make_replanner(llm_enabled=True)
        state = _make_state()
        goal = _make_state({"on(disk_1, peg_C)"})
        steps = replanner.replan(state, goal, "error")
        assert len(steps) > 0

    def test_attempt_counter_increments(self) -> None:
        replanner = _make_replanner(max_attempts=5)
        state = _make_state()
        goal = _make_state({"on(disk_1, peg_C)"})
        for i in range(3):
            replanner.replan(state, goal, f"error {i}")
        assert replanner._attempt_count == 3
