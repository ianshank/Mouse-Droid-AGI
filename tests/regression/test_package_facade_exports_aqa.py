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


def _all_value_nodes(tree: ast.Module) -> list[ast.expr] | None:
    """Return the value expression(s) assigned to a module-level ``__all__``.

    Handles ``__all__ = [...]`` (:class:`ast.Assign`), the annotated
    ``__all__: list[str] = [...]`` (:class:`ast.AnnAssign`), and ``__all__ +=
    [...]`` / ``__all__ = a + b`` accumulation. ``None`` means no ``__all__`` is
    declared, which is fine — only a *declared* export can be an unbound one.

    Earlier revisions handled only the plain ``Assign`` list literal, so an
    annotated or concatenated ``__all__`` silently returned ``None`` and the
    parametrised case skipped instead of checking anything.
    """
    values = [value for node in tree.body if (value := _all_assignment_value(node)) is not None]
    return values or None


def _all_assignment_value(node: ast.stmt) -> ast.expr | None:
    """The value a statement assigns to ``__all__``, or ``None`` if it does not.

    Split out per node type rather than combined into one condition: the three
    forms carry the target on different attributes (``Assign.targets`` is a list,
    ``AnnAssign.target``/``AugAssign.target`` are single nodes) and only
    ``AnnAssign`` has an optional value, so a single ``or`` would be both
    unreadable and harder for mypy to narrow.
    """
    if isinstance(node, ast.Assign):
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            return node.value
        return None
    if isinstance(node, ast.AnnAssign | ast.AugAssign):
        if isinstance(node.target, ast.Name) and node.target.id == "__all__":
            return node.value
        return None
    return None


def _declared_exports(tree: ast.Module) -> list[str] | None:
    """Return the literal string entries of ``__all__``, or ``None`` if absent.

    Only literal strings are returned. A dynamically-built entry (a name, a
    comprehension, another module's ``__all__``) is not statically knowable, so it
    is skipped rather than guessed at — see
    :func:`_has_unresolvable_all_entries`, which makes that skipping explicit
    instead of silent.
    """
    value_nodes = _all_value_nodes(tree)
    if value_nodes is None:
        return None
    names: list[str] = []
    for value in value_nodes:
        for element in ast.walk(value):
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                names.append(element.value)
    return names


def _has_unresolvable_all_entries(tree: ast.Module) -> bool:
    """Whether ``__all__`` mixes in entries this static check cannot resolve."""
    value_nodes = _all_value_nodes(tree)
    if value_nodes is None:
        return False
    return any(
        not isinstance(value, ast.List | ast.Tuple)
        # A non-string constant must count as unresolvable, not as a benign
        # entry: ``__all__ = [1]`` makes Python raise TypeError on a star-import,
        # but ``isinstance(el, ast.Constant)`` alone is satisfied by it, so the
        # element contributed no name AND was not flagged — the malformed facade
        # passed both halves of this check.
        or any(
            not isinstance(el, ast.Constant) or not isinstance(el.value, str) for el in value.elts
        )
        for value in value_nodes
    )


def _module_level_bound_names(tree: ast.Module) -> set[str]:
    """Names bound at module level and available at RUNTIME.

    Two exclusions, both load-bearing, and both absent from the first revision of
    this sweep — which used :func:`ast.walk` over the whole tree and therefore
    admitted the exact defect it exists to catch:

    * **``if TYPE_CHECKING:`` blocks are skipped.** Those imports do not execute
      at runtime, so a name bound only there is *not* reachable via
      ``from pkg import *`` — which is precisely the
      ``harness/approval`` failure mode. Counting them made the sweep
      satisfiable by a binding that does not exist when it matters.
    * **Function and class bodies are skipped.** A local inside a function is not
      a module attribute. ``ast.walk`` descended into both.

    Deliberately NOT excluded: ``try: import x / except ImportError:`` bodies. An
    optional-dependency import that fails still leaves the ``except`` branch to
    bind a fallback, and both branches run at module level, so either is a real
    runtime binding.
    """
    bound: set[str] = set()

    def visit(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                for alias in node.names:
                    # ``import a.b`` binds ``a``; ``import a.b as c`` binds ``c``.
                    bound.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                bound.add(node.name)  # the def itself binds; its body does not
            elif isinstance(node, ast.Assign):
                bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
            elif isinstance(node, ast.If):
                if _is_type_checking_test(node.test):
                    continue  # never executes at runtime
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Try):
                visit(node.body)
                for handler in node.handlers:
                    visit(handler.body)
                visit(node.orelse)
                visit(node.finalbody)
            elif isinstance(node, ast.With | ast.AsyncWith | ast.For | ast.While):
                visit(node.body)

    visit(tree.body)
    return bound


def _is_type_checking_test(test: ast.expr) -> bool:
    """Whether an ``if`` test is the ``TYPE_CHECKING`` guard, in any spelling."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _defines_module_getattr(tree: ast.Module) -> bool:
    """Whether the module implements :pep:`562` lazy attribute resolution.

    ``src/mousedroid/validation/__init__.py`` binds all 16 of its ``__all__``
    entries only under ``if TYPE_CHECKING:`` and resolves them at runtime through
    a module-level ``__getattr__`` that forwards to
    ``mousedroid.validation.runtime``. That facade is correct — it exists so that
    importing a dependency-free sibling does not drag in numpy/cv2/pyaudio — so a
    sweep that ignored PEP 562 would fail it for being right, and would also fail
    if the ``TYPE_CHECKING`` block (a mypy/IDE affordance) were ever deleted.
    """
    return any(
        isinstance(node, ast.FunctionDef) and node.name == "__getattr__" for node in tree.body
    )


def _modules_with_all() -> list[Path]:
    """Every module under ``src/mousedroid`` that declares ``__all__``.

    Not just ``__init__.py``. The first revision globbed packages only, covering
    40 files and leaving 67 non-package modules that also declare ``__all__``
    unchecked — including ``harness/approval/auto.py``, ``callback.py`` and
    ``cli.py``, the direct siblings of the package that regressed.
    """
    paths: list[Path] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _all_value_nodes(tree) is not None:
            paths.append(path)
    return paths


def test_modules_with_all_were_discovered() -> None:
    """Guard the guard: an empty sweep would pass every assertion vacuously."""
    modules = _modules_with_all()
    assert len(modules) > 50, f"sweep found suspiciously few modules with __all__: {len(modules)}"
    # The regressed package must be in scope, or the sweep proves nothing here.
    assert any(p.parts[-3:] == ("harness", "approval", "__init__.py") for p in modules)


@pytest.mark.parametrize("module_path", _modules_with_all(), ids=lambda p: str(p))
def test_every_declared_export_is_bound(module_path: Path) -> None:
    """Each ``__all__`` entry resolves to a name the module binds at RUNTIME.

    Parametrised per file so a failure names the offending module directly rather
    than reporting one aggregated list.
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    declared = _declared_exports(tree)
    assert declared is not None, "parametrisation guarantees an __all__ here"

    if _defines_module_getattr(tree):
        pytest.skip("module resolves exports lazily via a PEP 562 __getattr__")
    if _has_unresolvable_all_entries(tree):
        pytest.skip("__all__ contains entries that are not static string literals")

    unbound = sorted(set(declared) - _module_level_bound_names(tree))
    rel = module_path.relative_to(_SRC.parents[1])
    assert not unbound, (
        f"{rel} declares {unbound} in __all__ but never binds them at runtime - "
        "`from <mod> import *` and `<mod>.<name>` both raise AttributeError. "
        "Import the names at module level, drop them from __all__, or add a "
        "PEP 562 __getattr__ if the binding is deliberately lazy."
    )


def test_type_checking_only_binding_is_not_accepted() -> None:
    """The sweep must reject what it was written to catch. Meta-test.

    ``if TYPE_CHECKING:`` imports do not execute at runtime, so a name bound only
    there is exactly the ``harness/approval`` defect wearing a disguise: static
    analysers and IDEs resolve it, ``from pkg import *`` raises. The first
    revision of :func:`_module_level_bound_names` used ``ast.walk`` over the whole
    tree and therefore PASSED this input — so this test exists to pin the fix, not
    just the finding.
    """
    source = (
        "from __future__ import annotations\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from somewhere import BrokenExport\n"
        '__all__ = ["BrokenExport"]\n'
    )
    tree = ast.parse(source)
    assert _declared_exports(tree) == ["BrokenExport"]
    assert "BrokenExport" not in _module_level_bound_names(tree)
    assert not _defines_module_getattr(tree), "no lazy resolver, so this is a real break"


def test_function_local_binding_is_not_accepted() -> None:
    """A name bound inside a function body is not a module attribute."""
    source = "def f():\n    Local = 1\n    return Local\n__all__ = ['Local']\n"
    tree = ast.parse(source)
    assert "Local" not in _module_level_bound_names(tree)
    assert "f" in _module_level_bound_names(tree), "the def itself still binds"


def test_optional_dependency_try_except_binding_is_accepted() -> None:
    """Both branches of a ``try/except ImportError`` run at module level."""
    source = (
        "try:\n"
        "    from fast import Thing\n"
        "except ImportError:\n"
        "    from slow import Thing\n"
        "__all__ = ['Thing']\n"
    )
    tree = ast.parse(source)
    assert "Thing" in _module_level_bound_names(tree)


def test_annotated_and_augmented_all_are_parsed() -> None:
    """``__all__: list[str] = [...]`` and ``__all__ += [...]`` are both read."""
    annotated = ast.parse('__all__: list[str] = ["A"]\n')
    assert _declared_exports(annotated) == ["A"]
    augmented = ast.parse('__all__ = ["A"]\n__all__ += ["B"]\n')
    assert sorted(_declared_exports(augmented) or []) == ["A", "B"]


def test_lazy_pep562_facade_is_recognised() -> None:
    """``validation/__init__.py`` is correct and must not be failed for it.

    It binds all 16 ``__all__`` entries only under ``if TYPE_CHECKING:`` and
    resolves them at runtime through a module-level ``__getattr__`` forwarding to
    ``validation.runtime`` — deliberately, so importing a dependency-free sibling
    does not drag in numpy/cv2/pyaudio. A sweep ignorant of PEP 562 would fail it
    for being right, and would also fail if the ``TYPE_CHECKING`` block (a
    mypy/IDE affordance with no runtime effect) were ever deleted.
    """
    init_path = _SRC / "validation" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    assert _defines_module_getattr(tree), "validation/ lost its PEP 562 resolver"
    declared = _declared_exports(tree) or []
    assert declared, "validation/ declares no __all__ any more"
    # The point: these are NOT module-level bound, and that is fine here.
    assert set(declared) - _module_level_bound_names(tree)


def test_approval_facade_specifically_binds_both_gates() -> None:
    """Explicit pin on the package that regressed, independent of the sweep.

    The parametrised sweep would stop covering this the moment someone removed
    ``__all__`` from the file (it would no longer be discovered at all). This
    asserts the facade keeps its two names, so deleting the declaration is a
    visible decision rather than a silent loss of coverage.
    """
    init_path = _SRC / "harness" / "approval" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    assert _declared_exports(tree) == ["OpenClawSafetyGate", "SandboxPolicyGate"]
    assert {"OpenClawSafetyGate", "SandboxPolicyGate"} <= _module_level_bound_names(tree)
    assert not _defines_module_getattr(tree), "these are eager imports, not lazy"


def test_a_non_string_all_entry_counts_as_unresolvable() -> None:
    """``__all__ = [1]`` must not sail through as a benign entry.

    Python raises ``TypeError: attribute name must be string`` when expanding
    such an ``__all__``, so a facade declaring one is broken. The earlier check
    accepted any ``ast.Constant``: the element yielded no name for the export
    sweep and was not flagged as unresolvable either, so the malformed facade
    satisfied both halves. Found by review, not by the suite.
    """
    assert _has_unresolvable_all_entries(ast.parse("__all__ = [1]\n"))
    assert _has_unresolvable_all_entries(ast.parse("__all__ = ['ok', 2]\n"))
    # The well-formed case must stay unflagged, or the guard is just always-true.
    assert not _has_unresolvable_all_entries(ast.parse("__all__ = ['ok', 'fine']\n"))
