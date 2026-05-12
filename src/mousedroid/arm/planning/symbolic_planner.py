"""Symbolic planner with Pyperplan + recursive fallback.

Primary path: invoke Pyperplan via ``pyperplan.planner.search_plan`` to solve
the generated PDDL problem. When Pyperplan is unavailable, its API has shifted,
its PDDL parser rejects the generated problem, or it returns no solution, the
planner falls back to a deterministic recursive Tower-of-Hanoi solver that is
guaranteed to produce the optimal ``2^n - 1`` move plan. Upstream callers
(replanner, BDI loop) see a successful plan in every reachable case.

TODO(F-003-FOLLOWUP): replace the ``_import_search_plan`` seam with a
``@runtime_checkable Protocol`` (e.g. ``SymbolicPlannerBackend``) plus concrete
``PyperplanBackend`` / ``RecursiveBackend`` classes, selected by
``cfg.arm.planning.planner_backend`` (the existing
``Literal["pyperplan", "fast_downward"]`` field in
``ArmPlanningConfig``) via the project's standard factory. The same PR
should also replace the per-call ``ThreadPoolExecutor`` timeout below with
a ``multiprocessing.Process``-based hard interrupt so a pathological
pyperplan run can be ``terminate()``'d rather than orphaned. See
``smoke-reports/smoke_report.json`` entry ``F-003-FOLLOWUP`` for context.
"""

from __future__ import annotations

import concurrent.futures
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mousedroid.arm.planning.pddl_domain import generate_domain, generate_problem
from mousedroid.arm.protocols import PlanStep, SymbolicState
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import ArmPlanningConfig, ArmTaskConfig

_log = get_logger(__name__)


def _import_search_plan() -> tuple[Callable[..., Any], Callable[..., Any], type[Any]]:
    """Resolve pyperplan's ``search_plan`` + a default search + heuristic.

    Extracted as a module-level helper so tests can patch this single seam
    (``mousedroid.arm.planning.symbolic_planner._import_search_plan``) instead
    of trying to monkey-patch the pyperplan package via ``sys.modules``. The
    earlier code called ``pyperplan.solve(...)`` which never existed on the
    installed pyperplan (>=2.0) — the real entry point is
    ``pyperplan.planner.search_plan``.

    TODO(F-003-FOLLOWUP): once the Protocol-based backend lands this helper
    becomes the body of ``PyperplanBackend.import_engine()``.

    Returns:
        Tuple of (search_plan callable, default search callable, default heuristic class).

    Raises:
        ImportError: pyperplan or one of its submodules is not installed.
            (Note: ``ModuleNotFoundError`` is a subclass of ``ImportError`` and
            is caught by the same handler.)
        AttributeError: pyperplan is installed but the public API has shifted
            (e.g. older or vendor-patched build without ``search_plan``).

    Callers should treat either exception as a signal to fall back to the
    recursive Hanoi solver — no semantic difference between them here.
    """
    from pyperplan.heuristics.blind import BlindHeuristic
    from pyperplan.planner import search_plan
    from pyperplan.search import astar_search

    return search_plan, astar_search, BlindHeuristic


class PlanningError(Exception):
    """Raised when neither pyperplan nor the recursive fallback can plan."""


class SymbolicPlanner:
    """PDDL-based symbolic planner for Tower of Hanoi.

    Attempts to solve PDDL problems via Pyperplan (``search_plan`` + astar +
    blind heuristic). Falls back to a guaranteed-optimal recursive solver
    whenever Pyperplan is missing, its API has drifted, its parser rejects the
    generated problem, or it cannot find a plan. The fallback path is the
    common production case until the PDDL generator emits whitespace-correct
    output (tracked as F-005 in ``smoke-reports/smoke_report.json``).

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

        Falls back to the recursive Hanoi solver when pyperplan is missing,
        its public API has shifted, or it returns no solution (e.g. empty
        ``:init`` from test fixtures). The fallback guarantees an optimal
        ``2^n - 1`` plan for the configured disk count.

        Args:
            domain_str: PDDL domain definition.
            problem_str: PDDL problem definition.

        Returns:
            List of plan steps — either parsed from pyperplan output or
            produced by the recursive solver.
        """
        # ImportError covers ModuleNotFoundError (subclass) — single handler
        # routes both to the recursive fallback. AttributeError fires when the
        # package is present but missing the expected submodule attributes.
        try:
            search_plan, astar_search, heuristic_class = _import_search_plan()
        except (ImportError, AttributeError) as exc:
            _log.warning(
                "pyperplan_unavailable",
                reason=str(exc),
                exception_class=type(exc).__name__,
                fallback="recursive_solver",
            )
            return self._solve_recursive()

        search_name = getattr(astar_search, "__name__", "unknown_search")
        heuristic_name = getattr(heuristic_class, "__name__", "unknown_heuristic")
        # Best-effort timeout from config — pyperplan's search_plan is a
        # synchronous Python call with NO native cancellation hook, so we
        # wrap it in a per-call single-worker thread executor and bail out
        # when the configured budget elapses. On timeout the caller is
        # released to the recursive fallback IMMEDIATELY, but the orphan
        # worker thread continues to natural completion (or process exit).
        #
        # Per-call vs shared pool tradeoff (Copilot review #2 on PR #68):
        # a shared module-level pool with a bounded `max_workers` would
        # cap orphan-thread count, BUT once those workers are all stuck
        # on prior unbounded searches, NEW submissions queue behind them
        # and `future.result(timeout=...)` fires after the configured
        # window with the task still un-started — strictly worse than the
        # current per-call behaviour, where every call gets its own
        # worker. The honest fix for repeated-timeout pathologies is a
        # subprocess-based hard interrupt (signal-driven kill); that is
        # tracked as F-003-FOLLOWUP alongside the Protocol-based backend
        # refactor. For the normal case (transient timeout from one
        # malformed PDDL, then traffic continues), the per-call pool
        # accumulates one orphan that drains on its own — bounded and
        # acceptable.
        timeout_s = float(getattr(self._planning_cfg, "planning_timeout_s", 5.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            domain_path = Path(tmpdir) / "domain.pddl"
            problem_path = Path(tmpdir) / "problem.pddl"
            domain_path.write_text(domain_str)
            problem_path.write_text(problem_str)

            _log.debug(
                "pyperplan_search_start",
                search=search_name,
                heuristic=heuristic_name,
                domain_bytes=len(domain_str),
                problem_bytes=len(problem_str),
                timeout_s=timeout_s,
            )

            # Any pyperplan internal failure (ParseError, search failure,
            # heuristic init error, ...) AND a timeout both route to the
            # recursive solver — the upstream callers (replanner, BDI
            # loop) only need *a* valid plan, and the recursive solver is
            # guaranteed optimal for Hanoi. The narrow path lives entirely
            # inside this method so the outer `plan()` catch still wraps
            # truly catastrophic failures (e.g. domain generator bugs)
            # into PlanningError.
            t_start = time.monotonic()
            # NOTE: we deliberately do NOT use ThreadPoolExecutor as a context
            # manager — its __exit__ calls shutdown(wait=True) by default,
            # which would block until the orphan search_plan thread finishes,
            # defeating the whole purpose of the timeout. Instead we
            # explicitly shutdown(wait=False, cancel_futures=True) on every
            # exit path so a runaway pyperplan thread is left to terminate
            # on its own (or on process teardown).
            pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="pyperplan",
            )
            future = pool.submit(
                search_plan,
                str(domain_path),
                str(problem_path),
                astar_search,
                heuristic_class,
            )
            try:
                solution = future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                elapsed = time.monotonic() - t_start
                _log.warning(
                    "pyperplan_search_timeout",
                    timeout_s=timeout_s,
                    elapsed_s=round(elapsed, 4),
                    search=search_name,
                    heuristic=heuristic_name,
                    fallback="recursive_solver",
                    note="orphan worker continues until natural completion or process exit",
                )
                return self._solve_recursive()
            except Exception as exc:
                elapsed = time.monotonic() - t_start
                _log.warning(
                    "pyperplan_search_error",
                    error=str(exc),
                    exception_class=type(exc).__name__,
                    elapsed_s=round(elapsed, 4),
                    search=search_name,
                    heuristic=heuristic_name,
                    fallback="recursive_solver",
                )
                return self._solve_recursive()
            finally:
                # Non-blocking shutdown — happy path is a no-op (worker
                # has already returned); timeout/error paths leave the
                # orphan thread to terminate naturally. Keeping a single
                # finally-clause means every exit path is symmetric and
                # auditable.
                pool.shutdown(wait=False, cancel_futures=True)
            elapsed = time.monotonic() - t_start

            if solution is None:
                _log.info(
                    "pyperplan_no_solution",
                    elapsed_s=round(elapsed, 4),
                    search=search_name,
                    heuristic=heuristic_name,
                    fallback="recursive_solver",
                )
                return self._solve_recursive()

            steps = self._parse_solution([str(op) for op in solution])
            _log.info(
                "pyperplan_search_done",
                num_actions=len(steps),
                elapsed_s=round(elapsed, 4),
                search=search_name,
                heuristic=heuristic_name,
            )
            return steps

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
