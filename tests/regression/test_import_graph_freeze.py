"""Import-graph freeze for parked/deferred subsystems (F-020, WS-8.2, F-031).

The arm platform is deferred, the HC-SR04 ultrasonic driver is parked
(NEXT_STEPS.md "Deferred / Out Of Scope"), and `AutonomousOrchestrator` is kept off
the production path per ADR-016 (F-031). This test pins that no *active* production
module grows a **module-top-level** dependency on any of them.

Granularity matters: ``factory.py`` legitimately imports arm modules lazily
inside config-gated functions and under ``if TYPE_CHECKING:`` — both are the
documented DI pattern and MUST pass. Only unconditional module-scope imports
(which would execute on every ``import mousedroid.<x>``) are frozen out, so
the check walks module and class bodies while skipping function bodies and
``TYPE_CHECKING`` blocks. Relative imports (``from .x import Y``) ARE
resolved, not skipped — a same-directory module (like ``autonomous.py``
alongside ``orchestrator.py``) is exactly the case a blanket "relative
imports are always safe" assumption misses.

For AutonomousOrchestrator specifically, import-scope freezing alone is not
enough to enforce ADR-016's actual claim: a production entrypoint could call
``factory.py::build_autonomous_orchestrator`` (a public function) directly
without ever importing ``mousedroid.orchestrator.autonomous`` at module
scope itself. ``test_no_production_entrypoint_calls_the_autonomous_orchestrator_builder``
pins that call-site claim separately.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mousedroid"

# Frozen prefixes and the packages allowed to reference them at module scope.
_ARM_PREFIX = "mousedroid.arm"
_ULTRASONIC_PREFIXES = (
    "mousedroid.hardware.sensors.ultrasonic",
    "mousedroid.hardware.sensors.mock_ultrasonic",
)
# The ultrasonic *drivers* may be referenced by the factory (lazy, config-
# gated) and by their own package; the arm package references itself.
_ULTRASONIC_ALLOWED_DIRS = ("hardware/sensors",)

# ADR-016 / F-031: AutonomousOrchestrator stays off the production path. No
# directory-level exemption here (unlike arm/ and hardware/sensors/ above):
# autonomous.py shares a directory with the PRODUCTION orchestrator.py, so a
# directory allowlist would also license orchestrator.py to import it. Only
# the file itself is exempt; factory.py::build_autonomous_orchestrator's sole
# reference is function-scoped and so is never visited by this walker at all.
_AUTONOMOUS_ORCHESTRATOR_MODULE = "mousedroid.orchestrator.autonomous"
_AUTONOMOUS_ORCHESTRATOR_SELF_FILE = Path("orchestrator/autonomous.py")
_AUTONOMOUS_ORCHESTRATOR_BUILDER = "build_autonomous_orchestrator"
# main.py is where the production entrypoints (cli_entry/_run/_health_check)
# live; it is outside src/mousedroid/ (repo-root src layout), so it is read
# directly rather than discovered via _iter_module_files().
_MAIN_PY = Path(__file__).resolve().parents[2] / "src" / "mousedroid" / "main.py"


def _iter_module_files() -> list[Path]:
    return sorted(_SRC_ROOT.rglob("*.py"))


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _resolve_relative_base(own_package: str, level: int) -> str:
    # level=1 means "this package" (own_package itself); level=N means go up
    # (N-1) further levels from own_package.
    parts = own_package.split(".")
    base_parts = parts[: len(parts) - (level - 1)] if level > 1 else parts
    return ".".join(base_parts)


def _import_from_targets(node: ast.ImportFrom, own_package: str) -> list[str]:
    """Every dotted name this ``ImportFrom`` could plausibly refer to.

    Absolute imports (``level == 0``) always carry a ``module``, per
    Python's grammar. Relative imports resolve ``.``/``..`` against
    ``own_package`` (the dotted package name of the file being parsed --
    e.g. ``"mousedroid.orchestrator"``; regular modules and ``__init__.py``
    files share the same ``__package__`` resolution rule, so no
    special-casing is needed for either) first. Earlier versions of this
    walker skipped relative imports outright on the assumption "a relative
    import can only reach the importer's own package, which is always
    allowed" -- accurate for `arm/`/`hardware/sensors/`, but false the
    moment a frozen module shares a directory with an active one, as
    `autonomous.py` does with `orchestrator.py` (F-031/ADR-016): ``from
    .autonomous import X`` written inside `orchestrator.py` is a relative
    import reaching exactly the forbidden module.

    Either way, ``from pkg import submodule`` names the submodule as an
    alias, not inside ``node.module`` -- so both the bare module and each
    ``module.alias`` combination are recorded, e.g. `from
    mousedroid.orchestrator import autonomous`.
    """
    if node.level == 0:
        module = node.module or ""
    else:
        module = _resolve_relative_base(own_package, node.level)
        if node.module:
            module = f"{module}.{node.module}"
    if not node.module and node.level > 0:
        # `from . import submodule` -- the alias itself is the (sub)module,
        # not a name imported *from* one.
        return [f"{module}.{alias.name}" for alias in node.names]
    return [module, *(f"{module}.{alias.name}" for alias in node.names)]


def _walk_module_scope(body: list[ast.stmt], own_package: str, imports: list[str]) -> None:
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue  # lazy imports inside functions are the DI pattern
        if isinstance(node, ast.If):
            if _is_type_checking_guard(node):
                continue  # typing-only imports never execute at runtime
            _walk_module_scope(node.body, own_package, imports)
            _walk_module_scope(node.orelse, own_package, imports)
            continue
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.extend(_import_from_targets(node, own_package))
        elif isinstance(node, ast.ClassDef):
            _walk_module_scope(node.body, own_package, imports)
        elif isinstance(node, ast.Try):
            # Every branch can carry a module-scope import, not just body.
            _walk_module_scope(node.body, own_package, imports)
            for handler in node.handlers:
                _walk_module_scope(handler.body, own_package, imports)
            _walk_module_scope(node.orelse, own_package, imports)
            _walk_module_scope(node.finalbody, own_package, imports)
        elif isinstance(node, ast.With):
            _walk_module_scope(node.body, own_package, imports)


def _module_scope_imports(tree: ast.Module, own_package: str) -> list[str]:
    """Collect imported module names at module/class scope only.

    Function bodies (lazy DI imports) and ``if TYPE_CHECKING:`` blocks
    (typing-only imports, erased at runtime) are deliberately not visited.
    See ``_import_from_targets`` for how relative imports and submodule
    aliases are resolved.
    """
    imports: list[str] = []
    _walk_module_scope(tree.body, own_package, imports)
    return imports


def _own_package(rel: Path) -> str:
    """Dotted package name of ``rel`` (a path relative to ``_SRC_ROOT``)."""
    if not rel.parent.parts:
        return "mousedroid"
    return "mousedroid." + ".".join(rel.parent.parts)


@pytest.fixture(scope="module")
def module_imports() -> dict[Path, list[str]]:
    collected: dict[Path, list[str]] = {}
    for path in _iter_module_files():
        rel = path.relative_to(_SRC_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        collected[path] = _module_scope_imports(tree, _own_package(rel))
    return collected


def test_source_tree_is_non_trivial(module_imports: dict[Path, list[str]]) -> None:
    assert len(module_imports) > 100, "src tree unexpectedly small - wrong root?"


def test_no_active_module_imports_arm_at_module_scope(
    module_imports: dict[Path, list[str]],
) -> None:
    violations: list[str] = []
    for path, imports in module_imports.items():
        rel = path.relative_to(_SRC_ROOT)
        if rel.parts and rel.parts[0] == "arm":
            continue  # the arm package may reference itself
        for name in imports:
            if name == _ARM_PREFIX or name.startswith(_ARM_PREFIX + "."):
                violations.append(f"{rel}: module-scope import of {name}")
    assert not violations, (
        "deferred arm platform imported at module scope by active code "
        f"(use the lazy config-gated factory pattern instead): {violations}"
    )


def test_no_active_module_imports_parked_ultrasonic_driver(
    module_imports: dict[Path, list[str]],
) -> None:
    violations: list[str] = []
    for path, imports in module_imports.items():
        rel = path.relative_to(_SRC_ROOT)
        if any(
            # Segment-anchored match: "hardware/sensors" must not also admit a
            # future sibling like "hardware/sensors_experimental".
            rel.parent.parts[: len(Path(allowed).parts)] == Path(allowed).parts
            for allowed in _ULTRASONIC_ALLOWED_DIRS
        ):
            continue
        for name in imports:
            if any(name.startswith(p) for p in _ULTRASONIC_PREFIXES):
                violations.append(f"{rel}: module-scope import of {name}")
    assert not violations, (
        "parked HC-SR04 driver imported at module scope outside its package "
        f"(factory access must stay lazy/config-gated): {violations}"
    )


def test_no_active_module_imports_autonomous_orchestrator_at_module_scope(
    module_imports: dict[Path, list[str]],
) -> None:
    """F-031 / ADR-016: AutonomousOrchestrator is parked, off the production path.

    factory.py::build_autonomous_orchestrator is the sole, function-scoped
    (lazy) importer, already excluded by the walker's function-body skip.
    """
    violations: list[str] = []
    for path, imports in module_imports.items():
        rel = path.relative_to(_SRC_ROOT)
        if rel == _AUTONOMOUS_ORCHESTRATOR_SELF_FILE:
            continue  # the module may reference itself
        for name in imports:
            if name == _AUTONOMOUS_ORCHESTRATOR_MODULE or name.startswith(
                _AUTONOMOUS_ORCHESTRATOR_MODULE + "."
            ):
                violations.append(f"{rel}: module-scope import of {name}")
    assert not violations, (
        "AutonomousOrchestrator is parked off the production path (ADR-016) -- "
        f"module-scope import found where only a lazy, function-scoped import "
        f"is allowed: {violations}"
    )


def test_no_production_entrypoint_calls_the_autonomous_orchestrator_builder() -> None:
    """F-031 / ADR-016, narrower than the import-scope pin above.

    ``factory.py::build_autonomous_orchestrator`` is a public function --
    nothing stops a production entrypoint from calling it directly
    (``from mousedroid.factory import build_autonomous_orchestrator``)
    without ever importing ``mousedroid.orchestrator.autonomous`` itself at
    module scope anywhere outside ``autonomous.py``. That would leave the
    sibling import-graph test green while violating ADR-016's actual
    Decision: "main.py's _run and _health_check continue to route
    exclusively through factory.py::build_orchestrator". This test pins
    that claim directly rather than relying on the import check to imply it.
    """
    main_source = _MAIN_PY.read_text(encoding="utf-8")
    assert _AUTONOMOUS_ORCHESTRATOR_BUILDER not in main_source, (
        f"main.py references {_AUTONOMOUS_ORCHESTRATOR_BUILDER!r} -- ADR-016 "
        "requires production entrypoints to route exclusively through "
        "build_orchestrator, not just avoid importing autonomous.py directly"
    )
