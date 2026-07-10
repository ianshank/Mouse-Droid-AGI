"""AQA: schema + protocol-conformance contracts for the F-003 planner refactor.

Locks the invariants a future refactor could silently break:

* ``planner_backend`` keeps its backwards-compatible default and gains the new
  ``recursive`` member without dropping the legacy values.
* the concrete backends conform to ``SymbolicPlannerBackend`` (Protocol DI).
* the injectable seams stay keyword-only with ``None`` / default sentinels so
  existing direct constructions keep working.
"""

from __future__ import annotations

import inspect
import typing

from mousedroid.arm.planning.symbolic_planner import (
    PyperplanBackend,
    RecursiveBackend,
    SymbolicPlanner,
    make_primary_backend,
    run_pyperplan_subprocess,
)
from mousedroid.arm.protocols import SymbolicPlannerBackend
from mousedroid.config.schema import ArmPlanningConfig, ArmTaskConfig


def test_planner_backend_literal_membership() -> None:
    """Legacy values preserved + ``recursive`` added; default unchanged."""
    field = ArmPlanningConfig.model_fields["planner_backend"]
    members = set(typing.get_args(field.annotation))
    assert {"pyperplan", "fast_downward", "recursive"} <= members
    assert field.default == "pyperplan"  # backwards-compatible default
    assert field.description


def test_backends_conform_to_protocol() -> None:
    assert isinstance(PyperplanBackend(ArmPlanningConfig()), SymbolicPlannerBackend)
    assert isinstance(RecursiveBackend(ArmTaskConfig()), SymbolicPlannerBackend)


def test_make_primary_backend_returns_protocol() -> None:
    backend = make_primary_backend(ArmPlanningConfig(), ArmTaskConfig())
    assert isinstance(backend, SymbolicPlannerBackend)


def test_pyperplan_backend_runner_keyword_only_with_default() -> None:
    param = inspect.signature(PyperplanBackend.__init__).parameters["runner"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is run_pyperplan_subprocess


def test_symbolic_planner_backend_params_keyword_only_none_default() -> None:
    params = inspect.signature(SymbolicPlanner.__init__).parameters
    for name in ("primary_backend", "fallback_backend"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is None


def test_symbolic_planner_direct_construction_backwards_compatible() -> None:
    """The pre-refactor two-arg constructor still works (no injected backends)."""
    planner = SymbolicPlanner(ArmPlanningConfig(), ArmTaskConfig(num_disks=3, num_pegs=3))
    # Delegating helpers retained for existing call sites.
    assert len(planner._solve_recursive()) == 7  # 2**3 - 1
    assert planner._parse_solution(["(move d1 peg_A peg_B)"])[0].action == "move"
