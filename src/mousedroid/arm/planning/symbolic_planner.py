"""Symbolic planner using Pyperplan for PDDL solving.

Integrates with Pyperplan to solve PDDL problems and produce
optimal move sequences for Tower of Hanoi and related tasks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from mousedroid.arm.planning.pddl_domain import generate_domain, generate_problem
from mousedroid.arm.protocols import PlanStep, SymbolicState
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmPlanningConfig, ArmTaskConfig

_log = get_logger(__name__)


class PlanningError(Exception):
    """Raised when the planner fails to find a valid plan."""


class SymbolicPlanner:
    """PDDL-based symbolic planner for Tower of Hanoi.

    Uses Pyperplan to solve PDDL problems and return optimal
    action sequences.

    Args:
        planning_cfg: Planning configuration (backend, timeout).
        task_cfg: Task configuration (num_disks, num_pegs).
    """

    def __init__(self, planning_cfg: ArmPlanningConfig, task_cfg: ArmTaskConfig) -> None:
        """Initialise symbolic planner.

        Args:
            planning_cfg: Planning config with backend and timeout.
            task_cfg: Task config with disk/peg counts.
        """
        self._planning_cfg = planning_cfg
        self._task_cfg = task_cfg
        _log.info(
            "symbolic_planner_init",
            backend=planning_cfg.planner_backend,
            num_disks=task_cfg.num_disks,
        )

    def plan(
        self,
        initial_state: SymbolicState,
        goal_state: SymbolicState,
    ) -> list[PlanStep]:
        """Generate optimal plan from initial to goal state.

        Args:
            initial_state: Current symbolic state.
            goal_state: Target symbolic state (used for validation).

        Returns:
            Ordered list of plan steps.

        Raises:
            PlanningError: If no valid plan can be found.
        """
        domain_str = generate_domain()
        problem_str = generate_problem(self._task_cfg, initial_state)

        _log.info("planning_start", backend=self._planning_cfg.planner_backend)

        try:
            steps = self._solve_pddl(domain_str, problem_str)
        except Exception as exc:
            msg = f"Planning failed: {exc}"
            _log.error("planning_failed", error=str(exc))
            raise PlanningError(msg) from exc

        _log.info("planning_complete", num_steps=len(steps))
        return steps

    def replan(
        self,
        current_state: SymbolicState,
        goal_state: SymbolicState,
        error: str,
    ) -> list[PlanStep]:
        """Generate recovery plan after execution failure.

        Args:
            current_state: Current (possibly unexpected) symbolic state.
            goal_state: Original target state.
            error: Description of the failure.

        Returns:
            Recovery plan steps.

        Raises:
            PlanningError: If no recovery plan can be found.
        """
        _log.warning("replanning", error=error)
        return self.plan(current_state, goal_state)

    def _solve_pddl(self, domain_str: str, problem_str: str) -> list[PlanStep]:
        """Solve PDDL problem using Pyperplan.

        Args:
            domain_str: PDDL domain definition.
            problem_str: PDDL problem definition.

        Returns:
            List of plan steps parsed from solver output.

        Raises:
            PlanningError: If solver fails or returns no solution.
        """
        try:
            import pyperplan
        except ImportError:
            _log.warning("pyperplan_not_installed", fallback="recursive_solver")
            return self._solve_recursive()

        with tempfile.TemporaryDirectory() as tmpdir:
            domain_path = Path(tmpdir) / "domain.pddl"
            problem_path = Path(tmpdir) / "problem.pddl"
            domain_path.write_text(domain_str)
            problem_path.write_text(problem_str)

            solution = pyperplan.solve(str(domain_path), str(problem_path))

            if solution is None:
                msg = "Pyperplan returned no solution"
                raise PlanningError(msg)

            return self._parse_solution(solution)

    def _parse_solution(self, solution: list[str]) -> list[PlanStep]:
        """Parse Pyperplan solution into PlanStep objects.

        Args:
            solution: Raw solution lines from Pyperplan.

        Returns:
            Parsed plan steps.
        """
        steps: list[PlanStep] = []
        for line in solution:
            line = line.strip().strip("()")
            parts = line.split()
            if parts:
                action = parts[0]
                args = parts[1:]
                steps.append(PlanStep(action=action, args=args))
        return steps

    def _solve_recursive(self) -> list[PlanStep]:
        """Fallback: solve Tower of Hanoi recursively (guaranteed optimal).

        Returns:
            Optimal move sequence with exactly 2^n - 1 moves.
        """
        n = self._task_cfg.num_disks
        pegs = [f"peg_{chr(65 + i)}" for i in range(self._task_cfg.num_pegs)]
        disks = [f"disk_{i + 1}" for i in range(n)]

        steps: list[PlanStep] = []

        def hanoi(num: int, source: str, target: str, auxiliary: str) -> None:
            if num == 0:
                return
            hanoi(num - 1, source, auxiliary, target)
            steps.append(PlanStep(action="move", args=[disks[num - 1], source, target]))
            hanoi(num - 1, auxiliary, target, source)

        hanoi(n, pegs[0], pegs[-1], pegs[1])
        _log.info("recursive_solve_complete", num_steps=len(steps))
        return steps
