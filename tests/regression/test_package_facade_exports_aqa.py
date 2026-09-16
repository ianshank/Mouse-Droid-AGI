"""Automated Quality Assurance (AQA) — every ``__all__`` entry is actually bound.

``src/mousedroid/harness/approval/__init__.py`` declared
``__all__ = ["OpenClawSafetyGate", "SandboxPolicyGate"]`` and imported nothing,
so ``from mousedroid.harness.approval import *`` raised
``AttributeError: module 'mousedroid.harness.approval' has no attribute
'OpenClawSafetyGate'``. The package advertised a facade that did not exist.

That is incident 6 in ``.claude/skills/module-split-consistency-sweep/SKILL.md``
and a known by-product of splitting a flat module into a package: the
``__all__`` list is written from the plan, the imports are written from the code,
and nothing cross-checks them. ``tests/unit/factory/test_facade_completeness.py``
pins the inverse direction for one package (every public definition is
*re-exported*); this pins the forward direction for **every** package (every
*declared* export is bound).

Deliberately static (``ast``) rather than ``importlib.import_module``: several
packages under ``src/mousedroid`` reach optional extras (``torch``, ``mujoco``,
``onnxruntime``, ``mlflow``), so an import-based sweep would either pull the
whole dependency tree or skip exactly the packages most likely to drift. A name
declared in ``__all__`` but never bound is a purely syntactic property, so the
AST sees it without importing anything — and cannot be made flaky by which
extras happen to be installed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "mousedroid"


def _declared_exports(tree: ast.Module) -> list[str] | None:
    """Return the string entries of a module-level ``__all__``, or ``None``.

    ``None`` means the module declares no ``__all__`` at all, which is fine —
    only a *declared* export can be an unbound one.
    """
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            continue
        if isinstance(node.value, ast.List | ast.Tuple):
            return [
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
    return None


def _bound_names(tree: ast.Module) -> set[str]:
    """Return every name bound at module level by import, assignment or def."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                # ``import a.b`` binds ``a``; ``import a.b as c`` binds ``c``.
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
    return bound


def _init_files() -> list[Path]:
    return sorted(_SRC.rglob("__init__.py"))


def test_init_files_were_discovered() -> None:
    """Guard the guard: an empty sweep would pass every assertion vacuously."""
    assert len(_init_files()) > 20, "package sweep found suspiciously few __init__.py files"


@pytest.mark.parametrize("init_path", _init_files(), ids=lambda p: str(p))
def test_every_declared_export_is_bound(init_path: Path) -> None:
    """Each ``__all__`` entry resolves to a name the module actually binds.

    Parametrised per file so a failure names the offending package directly
    rather than reporting one aggregated list.
    """
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    declared = _declared_exports(tree)
    if declared is None:
        pytest.skip("module declares no __all__")

    unbound = sorted(set(declared) - _bound_names(tree))
    rel = init_path.relative_to(_SRC.parents[1])
    assert not unbound, (
        f"{rel} declares {unbound} in __all__ but never binds them - "
        "`from <pkg> import *` and `<pkg>.<name>` both raise AttributeError. "
        "Either import the names or drop them from __all__."
    )


def test_approval_facade_specifically_binds_both_gates() -> None:
    """Explicit pin on the package that regressed, independent of the sweep.

    The parametrised sweep above would stop covering this the moment someone
    removed ``__all__`` from the file (skip, not fail). This asserts the facade
    keeps its two names, so deleting the declaration is a visible decision.
    """
    init_path = _SRC / "harness" / "approval" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    assert _declared_exports(tree) == ["OpenClawSafetyGate", "SandboxPolicyGate"]
    assert {"OpenClawSafetyGate", "SandboxPolicyGate"} <= _bound_names(tree)
