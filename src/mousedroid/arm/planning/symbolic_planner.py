"""Symbolic planner with pluggable Protocol backends + recursive fallback.

Layer-1 planning is decomposed into ``@runtime_checkable`` backends
(:class:`~mousedroid.arm.protocols.SymbolicPlannerBackend`):

* :class:`PyperplanBackend` — solves the generated PDDL via
  ``pyperplan.planner.search_plan`` **in a hard-interruptible subprocess** so a
  pathological astar search on malformed PDDL can be ``terminate()``-d rather
  than orphaned. Returns ``None`` (→ fallback) when pyperplan is unavailable,
  its API has drifted, the search times out, it raises, or it finds no plan.
* :class:`RecursiveBackend` — a deterministic Tower-of-Hanoi solver guaranteed
  to emit the optimal ``2^n - 1`` move plan. Total: never returns ``None``.

:class:`SymbolicPlanner` selects a *primary* backend from
``cfg.arm.planning.planner_backend`` (via :func:`make_primary_backend`, mirrored
by ``factory.build_symbolic_planner_backend``) and always keeps a
:class:`RecursiveBackend` as the guaranteed *fallback*, so upstream callers
(replanner, BDI loop) see a valid plan in every reachable case.
"""

from __future__ import annotations

import multiprocessing
import queue as _queue
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mousedroid.arm.planning.pddl_domain import generate_domain, generate_problem
from mousedroid.arm.protocols import PlanStep, SymbolicState
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.arm.protocols import SymbolicPlannerBackend
    from mousedroid.config.schema import ArmPlanningConfig, ArmTaskConfig

_log = get_logger(__name__)

# A search runner turns a (domain, problem, timeout_s) triple into a list of
# pyperplan operator string reprs, or ``None`` when it could not produce a plan
# (unavailable / timeout / error / no solution). Injected into
# :class:`PyperplanBackend` so tests exercise the parse + fallback logic
# in-process without spawning a real subprocess.
SearchRunner = Callable[[str, str, float], "list[str] | None"]


def _import_search_plan() -> tuple[Callable[..., Any], Callable[..., Any], type[Any]]:
    """Resolve pyperplan's ``search_plan`` + a default search + heuristic.

    Extracted as a module-level helper so it can be patched as a single seam
    (``mousedroid.arm.planning.symbolic_planner._import_search_plan``) instead
    of monkey-patching the pyperplan package via ``sys.modules``. Doubles as the
    cheap in-process *availability probe*: :func:`run_pyperplan_subprocess`
    calls it before spawning a worker so a host without pyperplan never pays the
    process-spawn cost.

    Returns:
        Tuple of (search_plan callable, default search callable, default heuristic class).

    Raises:
        ImportError: pyperplan or one of its submodules is not installed.
            (``ModuleNotFoundError`` is a subclass and is caught by the same
            handler.)
        AttributeError: pyperplan is installed but the public API has shifted
            (e.g. older or vendor-patched build without ``search_plan``).
    """
    from pyperplan.heuristics.blind import BlindHeuristic
    from pyperplan.planner import search_plan
    from pyperplan.search import astar_search

    return search_plan, astar_search, BlindHeuristic


class PlanningError(Exception):
    """Raised when neither the primary backend nor the fallback can plan."""


# --------------------------------------------------------------------------- #
# Pure helpers (no I/O, no config) — shared by the backends + planner.
# --------------------------------------------------------------------------- #
def parse_solution(solution: list[str]) -> list[PlanStep]:
    """Parse pyperplan operator string reprs into :class:`PlanStep` objects.

    Args:
        solution: Operator reprs of the form ``"(move disk_1 peg_A peg_C)"``.

    Returns:
        Parsed plan steps (empty/whitespace lines skipped).
    """
    steps: list[PlanStep] = []
    for raw in solution:
        line = raw.strip().strip("()")
        parts = line.split()
        if parts:
            steps.append(PlanStep(action=parts[0], args=parts[1:]))
    return steps


def solve_hanoi(num_disks: int, num_pegs: int) -> list[PlanStep]:
    """Deterministic Tower-of-Hanoi solver (classic single-auxiliary recursion).

    Uses the source, the target, and one auxiliary peg. The move sequence is
    **optimal** (exactly ``2^n - 1`` moves) for the standard 3-peg tower; with
    more than 3 pegs it still returns a *valid* plan but ignores the extra pegs
    (it is not Frame-Stewart optimal). The tower is undefined for fewer than 3
    pegs — a 2-peg tower cannot move a stack of 2+ disks without violating the
    size rule — so that case fails loud rather than emitting invalid moves.

    Args:
        num_disks: Number of disks ``n``; produces ``2^n - 1`` moves (3-peg).
        num_pegs: Number of pegs (must be ``>= 3``); first is source, last is
            target.

    Returns:
        Ordered move sequence.

    Raises:
        ValueError: If ``num_pegs < 3``.
    """
    if num_pegs < 3:
        msg = f"Tower of Hanoi requires at least 3 pegs, got {num_pegs}"
        raise ValueError(msg)
    pegs = [f"peg_{chr(65 + i)}" for i in range(num_pegs)]
    disks = [f"disk_{i + 1}" for i in range(num_disks)]
    steps: list[PlanStep] = []

    def hanoi(num: int, source: str, target: str, auxiliary: str) -> None:
        if num == 0:
            return
        hanoi(num - 1, source, auxiliary, target)
        steps.append(PlanStep(action="move", args=[disks[num - 1], source, target]))
        hanoi(num - 1, auxiliary, target, source)

    hanoi(num_disks, pegs[0], pegs[-1], pegs[1])
    return steps


# --------------------------------------------------------------------------- #
# Pyperplan subprocess execution (hard-interruptible).
# --------------------------------------------------------------------------- #
def _pyperplan_worker(  # pragma: no cover - runs in child process, untraceable by coverage
    domain_path: str, problem_path: str, result_queue: Any
) -> None:
    """Subprocess entry point: run pyperplan and push a picklable result.

    Runs in a separate process so a runaway astar search can be
    ``terminate()``-d by the parent. Only picklable data crosses the boundary:
    the result is ``("ok", list[str] | None)`` or ``("error", message)`` —
    never a live pyperplan object. Coverage.py cannot trace across the
    fork/spawn boundary; exercised by ``TestRealPyperplanSubprocess``.

    Args:
        domain_path: Path to the written PDDL domain file.
        problem_path: Path to the written PDDL problem file.
        result_queue: Multiprocessing queue the parent reads once.
    """
    try:
        search_plan, astar_search, heuristic_class = _import_search_plan()
        solution = search_plan(domain_path, problem_path, astar_search, heuristic_class)
        if solution is None:
            result_queue.put(("ok", None))
        else:
            result_queue.put(("ok", [str(op) for op in solution]))
    except Exception as exc:  # NOT BaseException — let KeyboardInterrupt/SystemExit kill the child
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def run_pyperplan_subprocess(
    domain_pddl: str, problem_pddl: str, timeout_s: float
) -> list[str] | None:
    """Default :data:`SearchRunner`: solve PDDL in a hard-interruptible subprocess.

    Probes pyperplan availability in-process first (cheap, no spawn), then runs
    ``search_plan`` in a worker process joined with ``timeout_s``. On timeout the
    worker is ``terminate()``-d so it cannot orphan; the caller is released to
    the fallback backend.

    Args:
        domain_pddl: Generated PDDL domain definition.
        problem_pddl: Generated PDDL problem definition.
        timeout_s: Hard wall-clock budget for the search.

    Returns:
        Operator string reprs, or ``None`` on unavailable / timeout / error /
        no-solution (every one routes the caller to the fallback).
    """
    try:
        _import_search_plan()
    except (ImportError, AttributeError) as exc:
        _log.warning(
            "pyperplan_unavailable",
            reason=str(exc),
            exception_class=type(exc).__name__,
            fallback="recursive_solver",
        )
        return None

    ctx = multiprocessing.get_context()
    result_queue: Any = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmpdir:
        domain_path = Path(tmpdir) / "domain.pddl"
        problem_path = Path(tmpdir) / "problem.pddl"
        domain_path.write_text(domain_pddl)
        problem_path.write_text(problem_pddl)

        _log.debug(
            "pyperplan_search_start",
            domain_bytes=len(domain_pddl),
            problem_bytes=len(problem_pddl),
            timeout_s=timeout_s,
        )
        t_start = time.monotonic()
        proc = ctx.Process(
            target=_pyperplan_worker,
            args=(str(domain_path), str(problem_path), result_queue),
            name="pyperplan-search",
        )
        proc.start()
        try:
            # Drain the queue BEFORE joining, using the timeout as the budget. A
            # child putting a result larger than the OS pipe buffer blocks on
            # exit until the parent reads it (the ``multiprocessing.Queue``
            # feeder-thread contract); a ``join(timeout)``-before-``get`` would
            # then deadlock on a large plan and time out even though a solution
            # was found. Draining first makes the child exit cleanly, and a
            # genuine runaway search still trips the ``get`` timeout below
            # (empty → the worker is hard-terminated in the reap below).
            result = _collect_pyperplan_result(result_queue, t_start, timeout_s)
            if result is None and proc.is_alive():
                _log.warning(
                    "pyperplan_search_timeout",
                    timeout_s=timeout_s,
                    elapsed_s=round(time.monotonic() - t_start, 4),
                    fallback="recursive_solver",
                    note="worker hard-terminated",
                )
            return result
        finally:
            # Reap the worker (bounded, with SIGKILL escalation) and release the
            # process + queue handles so the many plan()/replan() calls over a
            # mission never leak file descriptors or a lingering feeder thread.
            _reap_process(proc)
            result_queue.close()
            result_queue.join_thread()


def _reap_process(proc: Any, grace_s: float = 5.0) -> None:
    """Reap a worker process (SIGTERM → bounded wait → SIGKILL) + release its handle.

    A plain ``join()`` can block forever if the worker ignores SIGTERM — a
    dependency that installs a signal handler, or a thread stuck in a C
    extension. Terminate, wait a bounded grace period, then hard-``kill()`` so
    the parent never hangs the planner. ``close()`` releases the process handle
    (an open FD) once the worker is dead.

    Args:
        proc: The worker process.
        grace_s: Seconds to wait after SIGTERM before escalating to SIGKILL.
    """
    if proc.is_alive():
        proc.terminate()
    proc.join(grace_s)
    if proc.is_alive():  # pragma: no cover - a worker ignoring SIGTERM is not reproducible in-test
        proc.kill()
        proc.join()
    proc.close()


def _collect_pyperplan_result(
    result_queue: Any, t_start: float, timeout: float
) -> list[str] | None:
    """Drain the worker's single result, mapping every non-plan outcome to ``None``.

    Args:
        result_queue: The queue the worker put its result on.
        t_start: ``time.monotonic()`` start stamp for elapsed logging.
        timeout: Seconds to wait for the worker's result (the planning budget) —
            an expiry means the search is still running (caller hard-terminates)
            or the worker died without a result.

    Returns:
        Operator string reprs, or ``None`` (error / no-solution / empty queue).
    """
    elapsed = round(time.monotonic() - t_start, 4)
    try:
        status, payload = result_queue.get(timeout=timeout)
    except _queue.Empty:
        # No result within the budget — the worker is still running (timeout) or
        # died without putting (crash). The caller inspects ``proc.is_alive()``
        # to emit the actionable ``pyperplan_search_timeout`` warning + kill.
        _log.debug("pyperplan_result_not_ready", elapsed_s=elapsed)
        return None

    if status == "error":
        _log.warning(
            "pyperplan_search_error",
            error=payload,
            elapsed_s=elapsed,
            fallback="recursive_solver",
        )
        return None
    if payload is None:
        _log.info("pyperplan_no_solution", elapsed_s=elapsed, fallback="recursive_solver")
        return None

    _log.info("pyperplan_search_done", num_actions=len(payload), elapsed_s=elapsed)
    return list(payload)


# --------------------------------------------------------------------------- #
# Backends.
# --------------------------------------------------------------------------- #
class PyperplanBackend:
    """PDDL backend that delegates to pyperplan via an injectable runner.

    Args:
        planning_cfg: Planning config (supplies ``planning_timeout_s``).
        runner: Search runner; defaults to :func:`run_pyperplan_subprocess`.
            Tests inject an in-process fake to exercise parse + fallback without
            spawning a subprocess.
    """

    def __init__(
        self, planning_cfg: ArmPlanningConfig, *, runner: SearchRunner = run_pyperplan_subprocess
    ) -> None:
        self._cfg = planning_cfg
        self._runner = runner

    def search(self, domain_pddl: str, problem_pddl: str) -> list[PlanStep] | None:
        """Solve via the runner; ``None`` on any failure (caller falls back)."""
        timeout_s = float(getattr(self._cfg, "planning_timeout_s", 5.0))
        try:
            raw = self._runner(domain_pddl, problem_pddl, timeout_s)
        except Exception as exc:
            _log.warning(
                "pyperplan_runner_error",
                error=str(exc),
                exception_class=type(exc).__name__,
                fallback="recursive_solver",
            )
            return None
        if raw is None:
            return None
        return parse_solution(raw)


class RecursiveBackend:
    """Deterministic Tower-of-Hanoi backend.

    Total for valid inputs — never returns ``None`` — but raises ``ValueError``
    for an unsolvable ``num_pegs < 3`` config (surfaced as ``PlanningError`` by
    :class:`SymbolicPlanner`).

    Args:
        task_cfg: Task config supplying ``num_disks`` / ``num_pegs``.
    """

    def __init__(self, task_cfg: ArmTaskConfig) -> None:
        self._task_cfg = task_cfg

    def search(self, domain_pddl: str, problem_pddl: str) -> list[PlanStep] | None:
        """Return the optimal recursive plan (ignores the PDDL text by design)."""
        steps = solve_hanoi(self._task_cfg.num_disks, self._task_cfg.num_pegs)
        _log.info("recursive_solve_complete", num_steps=len(steps))
        return steps


def make_primary_backend(
    planning_cfg: ArmPlanningConfig, task_cfg: ArmTaskConfig
) -> SymbolicPlannerBackend:
    """Select the primary backend from ``planning_cfg.planner_backend``.

    ``fast_downward`` is not yet wired and transparently uses the Pyperplan
    backend (preserving pre-refactor behaviour, where the field only selected
    pyperplan-then-recursive). Single source of truth for both the
    :class:`SymbolicPlanner` default and ``factory.build_symbolic_planner_backend``.

    Args:
        planning_cfg: Planning config with the backend selector.
        task_cfg: Task config (needed by the recursive backend).

    Returns:
        A backend conforming to :class:`SymbolicPlannerBackend`.
    """
    backend = planning_cfg.planner_backend
    if backend == "recursive":
        return RecursiveBackend(task_cfg)
    if backend == "fast_downward":
        _log.warning(
            "planner_backend_not_implemented",
            requested=backend,
            using="pyperplan",
        )
    return PyperplanBackend(planning_cfg)


# --------------------------------------------------------------------------- #
# Planner (orchestrates primary + guaranteed fallback).
# --------------------------------------------------------------------------- #
class SymbolicPlanner:
    """PDDL-based symbolic planner: primary backend with a guaranteed fallback.

    Attempts the primary backend selected by ``planning_cfg.planner_backend``;
    when it returns ``None`` (unavailable / timeout / error / no solution) the
    :class:`RecursiveBackend` fallback produces a guaranteed-optimal plan, so
    upstream callers always receive a valid plan.

    Args:
        planning_cfg: Planning configuration (backend, timeout).
        task_cfg: Task configuration (num_disks, num_pegs).
        primary_backend: Optional injected primary backend (defaults to
            :func:`make_primary_backend`).
        fallback_backend: Optional injected fallback (defaults to
            :class:`RecursiveBackend`).
    """

    def __init__(
        self,
        planning_cfg: ArmPlanningConfig,
        task_cfg: ArmTaskConfig,
        *,
        primary_backend: SymbolicPlannerBackend | None = None,
        fallback_backend: SymbolicPlannerBackend | None = None,
    ) -> None:
        self._planning_cfg = planning_cfg
        self._task_cfg = task_cfg
        self._primary: SymbolicPlannerBackend = (
            primary_backend
            if primary_backend is not None
            else make_primary_backend(planning_cfg, task_cfg)
        )
        self._fallback: SymbolicPlannerBackend = (
            fallback_backend if fallback_backend is not None else RecursiveBackend(task_cfg)
        )
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
        """Generate a plan from initial to goal state.

        Args:
            initial_state: Current symbolic state.
            goal_state: Target symbolic state (used for validation).

        Returns:
            Ordered list of plan steps.

        Raises:
            PlanningError: If planning fails catastrophically (e.g. the PDDL
                generator raises) or no backend produced a plan.
        """
        domain_str = generate_domain()
        problem_str = generate_problem(self._task_cfg, initial_state)
        _log.info("planning_start", backend=self._planning_cfg.planner_backend)
        try:
            steps = self._solve(domain_str, problem_str)
        except PlanningError:
            raise
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
        """Generate a recovery plan after execution failure.

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

    def _solve(self, domain_str: str, problem_str: str) -> list[PlanStep]:
        """Try the primary backend, then the guaranteed fallback.

        Args:
            domain_str: PDDL domain definition.
            problem_str: PDDL problem definition.

        Returns:
            Plan steps from the primary backend, or the fallback when the
            primary returns ``None``.

        Raises:
            PlanningError: If even the fallback returns ``None`` (should not
                happen with the recursive fallback).
        """
        steps = self._primary.search(domain_str, problem_str)
        if steps is not None:
            return steps
        _log.info("planner_primary_no_plan", fallback="recursive_solver")
        steps = self._fallback.search(domain_str, problem_str)
        if steps is None:
            msg = "fallback backend returned no plan"
            raise PlanningError(msg)
        return steps

    # -- Backwards-compatible thin delegators (kept for existing call sites) --
    def _parse_solution(self, solution: list[str]) -> list[PlanStep]:
        """Delegate to the module-level :func:`parse_solution`."""
        return parse_solution(solution)

    def _solve_recursive(self) -> list[PlanStep]:
        """Delegate to the module-level :func:`solve_hanoi` for this task."""
        return solve_hanoi(self._task_cfg.num_disks, self._task_cfg.num_pegs)
