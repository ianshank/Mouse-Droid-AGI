"""AQA pins for F-038 CI/docs honesty.

The bug was the label, not a missing production twin: ``tests/functional/``
and ``tests/user_journey/`` uniquely prove parked ``AutonomousOrchestrator``
APIs. Silently retargeting them at ``build_orchestrator`` under
``mock_hardware`` reconstructs the F-025 vacuity. Relabel + pin instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CI_YML = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_CI_SH = _REPO_ROOT / "scripts" / "ci.sh"
_FUNCTIONAL = _REPO_ROOT / "tests" / "functional"
_USER_JOURNEY = _REPO_ROOT / "tests" / "user_journey"


def _factory_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "mousedroid.factory":
            continue
        for alias in node.names:
            names.add(alias.name)
    return names


def test_parked_tiers_import_autonomous_builder_only() -> None:
    """A future 'fix' that points these at production must be a reviewed choice."""
    py_files = [
        p
        for directory in (_FUNCTIONAL, _USER_JOURNEY)
        for p in directory.glob("*.py")
        if p.name != "__init__.py"
    ]
    assert py_files, "parked tiers vanished"
    for path in py_files:
        imported = _factory_imports(path)
        assert "build_autonomous_orchestrator" in imported, path.name
        assert "build_orchestrator" not in imported, (
            f"{path.name} imported production build_orchestrator — that silent "
            "retarget is the F-038 anti-pattern"
        )


def test_ci_step_is_labelled_parked_autonomous() -> None:
    text = _CI_YML.read_text(encoding="utf-8")
    assert "Run parked-autonomous functional/user-journey + security tiers" in text
    assert "Run functional + user-journey + security tiers\n" not in text


def test_ci_and_local_gates_invoke_doc_hygiene_strict() -> None:
    """F-038: the size budget is a real gate, not advisory check_doc."""
    ci_yml = _CI_YML.read_text(encoding="utf-8")
    ci_sh = _CI_SH.read_text(encoding="utf-8")
    assert "tools/doc_hygiene.py NEXT_STEPS.md --strict" in ci_yml
    assert "tools/doc_hygiene.py NEXT_STEPS.md --strict" in ci_sh
    assert "Doc hygiene (advisory" not in ci_yml
