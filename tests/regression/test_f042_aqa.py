"""AQA pins for F-042 enumerated branch-coverage allowlist."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_branch_coverage.py"
_FACTORY_DIR = _REPO_ROOT / "src" / "mousedroid" / "factory"
_ORCH_DIR = _REPO_ROOT / "src" / "mousedroid" / "orchestrator"

_GATED_FACTORY_FILES = frozenset(
    {
        "src/mousedroid/factory/on_device_learning.py",
        "src/mousedroid/factory/mcp_harness.py",
        "src/mousedroid/factory/_replay_batch_helpers.py",
    }
)


def _load_coverage_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_branch_coverage_f042",
        _SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load check_branch_coverage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_factory_and_orchestrator_are_not_directory_prefixes() -> None:
    mod = _load_coverage_script_module()
    prefixes: tuple[str, ...] = mod._ALLOWED_DIR_PREFIXES
    assert "src/mousedroid/factory/" not in prefixes
    assert "src/mousedroid/orchestrator/_" not in prefixes
    assert not any(p.startswith("src/mousedroid/factory") for p in prefixes)
    assert not any(p.startswith("src/mousedroid/orchestrator") for p in prefixes)


def test_every_factory_module_is_enumerated_or_gated() -> None:
    """A new factory/*.py must be classified; it must not inherit a prefix."""
    mod = _load_coverage_script_module()
    on_disk = {f"src/mousedroid/factory/{path.name}" for path in _FACTORY_DIR.glob("*.py")}
    exempt = {path for path in mod._ALLOWED_FILES if path.startswith("src/mousedroid/factory/")}
    assert on_disk == exempt | _GATED_FACTORY_FILES
    assert exempt.isdisjoint(_GATED_FACTORY_FILES)
    for path in _GATED_FACTORY_FILES:
        assert not mod._is_exempted_from_branch_gate(path)
    for path in exempt:
        assert mod._is_exempted_from_branch_gate(path)


def test_every_orchestrator_underscore_module_is_enumerated() -> None:
    """Mixin/_state files stay exempt; __init__.py must not ride a prefix."""
    mod = _load_coverage_script_module()
    on_disk = {
        f"src/mousedroid/orchestrator/{path.name}"
        for path in _ORCH_DIR.iterdir()
        if path.suffix == ".py" and path.name.startswith("_") and not path.name.startswith("__")
    }
    exempt = {
        path for path in mod._ALLOWED_FILES if path.startswith("src/mousedroid/orchestrator/")
    }
    assert on_disk == exempt
    assert not mod._is_exempted_from_branch_gate("src/mousedroid/orchestrator/__init__.py")
    assert not mod._is_exempted_from_branch_gate("src/mousedroid/orchestrator/autonomous.py")


def test_hypothetical_new_modules_are_not_exempt() -> None:
    mod = _load_coverage_script_module()
    assert not mod._is_exempted_from_branch_gate("src/mousedroid/factory/new_builder.py")
    assert not mod._is_exempted_from_branch_gate("src/mousedroid/orchestrator/_new_mixin.py")
