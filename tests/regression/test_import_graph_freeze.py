"""Import-graph freeze for parked/deferred subsystems (F-020, WS-8.2).

The arm platform is deferred and the HC-SR04 ultrasonic driver is parked
(NEXT_STEPS.md "Deferred / Out Of Scope"). This test pins that no *active*
production module grows a **module-top-level** dependency on them.

Granularity matters: ``factory.py`` legitimately imports arm modules lazily
inside config-gated functions and under ``if TYPE_CHECKING:`` — both are the
documented DI pattern and MUST pass. Only unconditional module-scope imports
(which would execute on every ``import mousedroid.<x>``) are frozen out, so
the check walks module and class bodies while skipping function bodies and
``TYPE_CHECKING`` blocks.
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


def _iter_module_files() -> list[Path]:
    return sorted(_SRC_ROOT.rglob("*.py"))


def _module_scope_imports(tree: ast.Module) -> list[str]:
    """Collect imported module names at module/class scope only.

    Function bodies (lazy DI imports) and ``if TYPE_CHECKING:`` blocks
    (typing-only imports, erased at runtime) are deliberately not visited.
    """
    imports: list[str] = []

    def _is_type_checking_guard(node: ast.If) -> bool:
        test = node.test
        return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )

    def _walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue  # lazy imports inside functions are the DI pattern
            if isinstance(node, ast.If):
                if _is_type_checking_guard(node):
                    continue  # typing-only imports never execute at runtime
                _walk(node.body)
                _walk(node.orelse)
                continue
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imports.append(node.module)
            elif isinstance(node, ast.ClassDef):
                _walk(node.body)
            elif isinstance(node, ast.Try | ast.With):
                _walk(getattr(node, "body", []))

    # NOTE: relative imports (level > 0) can only reference the importer's own
    # package, which is always allowed - they are intentionally not resolved.

    _walk(tree.body)
    return imports


@pytest.fixture(scope="module")
def module_imports() -> dict[Path, list[str]]:
    collected: dict[Path, list[str]] = {}
    for path in _iter_module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        collected[path] = _module_scope_imports(tree)
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
        if any(str(rel.parent).startswith(allowed) for allowed in _ULTRASONIC_ALLOWED_DIRS):
            continue
        for name in imports:
            if any(name.startswith(p) for p in _ULTRASONIC_PREFIXES):
                violations.append(f"{rel}: module-scope import of {name}")
    assert not violations, (
        "parked HC-SR04 driver imported at module scope outside its package "
        f"(factory access must stay lazy/config-gated): {violations}"
    )
