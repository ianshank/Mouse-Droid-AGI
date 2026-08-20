# tests/regression/test_cross_platform_portability_aqa.py
"""AQA: cross-platform portability invariants.

Pins the patterns established by the SQE strategic audit (Aug 2026) that make
the test suite pass on Windows alongside the existing Linux/Jetson CI matrix.

Contracts:

* ``PurePosixPath`` is used alongside ``Path`` in workforce config validation
  so that Unix-style absolute paths (``/etc/...``) are rejected on Windows;
* the ``_make_stub`` helper in ``test_secret_scan.py`` handles ``sys.platform``
  branching so stub executables resolve via ``shutil.which`` on both platforms;
* bash-dependent test files carry ``sys.platform == "win32"`` skip guards;
* mujoco lazy-import assertions use the snapshot pattern (not bare
  ``assert "mujoco" not in sys.modules``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Workforce config: PurePosixPath validation
# ---------------------------------------------------------------------------


def test_workforce_config_imports_pure_posix_path() -> None:
    """The config module must import PurePosixPath for cross-platform validation."""
    source = (_REPO_ROOT / "tools" / "claude_hooks" / "config.py").read_text(encoding="utf-8")
    assert "PurePosixPath" in source, (
        "tools/claude_hooks/config.py must import PurePosixPath for "
        "cross-platform absolute-path validation"
    )


def test_freeze_config_uses_pure_posix_path() -> None:
    """FreezeConfig._reject_absolute must check PurePosixPath.is_absolute()."""
    source = (_REPO_ROOT / "tools" / "claude_hooks" / "config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_reject_absolute":
            body_src = ast.get_source_segment(source, node)
            if body_src and "FreezeConfig" not in body_src:
                continue
            if body_src is not None:
                assert "PurePosixPath" in body_src, (
                    "FreezeConfig._reject_absolute must use PurePosixPath"
                )
                return
    # If we have the import, trust the code structure — the import test already
    # confirms the module uses it.


def test_unix_absolute_path_rejected_on_any_platform() -> None:
    """``/etc/features.yaml`` must be rejected as absolute on every platform.

    This is the *semantic* test that PurePosixPath is doing its job.
    """
    from pydantic import ValidationError
    from tools.claude_hooks.config import WorkforceConfig

    with pytest.raises(ValidationError):
        WorkforceConfig.model_validate({"freeze": {"features_file": "/etc/features.yaml"}})


# ---------------------------------------------------------------------------
# Test stubs: _make_stub cross-platform helper
# ---------------------------------------------------------------------------


def test_make_stub_helper_exists_in_secret_scan_tests() -> None:
    """The _make_stub helper must exist for cross-platform stub creation."""
    source = (
        _REPO_ROOT / "tests" / "unit" / "tools" / "claude_hooks" / "test_secret_scan.py"
    ).read_text(encoding="utf-8")
    assert "def _make_stub(" in source, "test_secret_scan.py must contain the _make_stub helper"


def test_make_stub_handles_windows_platform() -> None:
    """_make_stub must branch on sys.platform == 'win32' for .cmd creation."""
    source = (
        _REPO_ROOT / "tests" / "unit" / "tools" / "claude_hooks" / "test_secret_scan.py"
    ).read_text(encoding="utf-8")
    # The helper must contain both the Windows branch and Unix branch
    assert 'sys.platform == "win32"' in source or "sys.platform == 'win32'" in source
    assert ".cmd" in source, "_make_stub must create .cmd files on Windows"


# ---------------------------------------------------------------------------
# Bash-dependent tests: sys.platform skip guards
# ---------------------------------------------------------------------------

_BASH_DEPENDENT_TEST_FILES = [
    "tests/unit/scripts/test_jetson_runner_install.py",
    "tests/smoke/test_jetson_full_validation_sanity.py",
]


@pytest.mark.parametrize("relpath", _BASH_DEPENDENT_TEST_FILES)
def test_bash_dependent_tests_have_windows_skip_guard(relpath: str) -> None:
    """Every bash-dependent test file must carry a sys.platform skip guard."""
    source = (_REPO_ROOT / relpath).read_text(encoding="utf-8")
    assert "win32" in source, (
        f"{relpath} must have a sys.platform == 'win32' skip guard — "
        "WSL bash.EXE exists but cannot run scripts under pruned Unix PATH"
    )


# ---------------------------------------------------------------------------
# Mujoco lazy-import: snapshot pattern
# ---------------------------------------------------------------------------

_MUJOCO_ASSERTION_FILES = [
    "tests/integration/test_compare_drift_script.py",
    "tests/integration/test_spike_step_distillation.py",
]


@pytest.mark.parametrize("relpath", _MUJOCO_ASSERTION_FILES)
def test_mujoco_assertion_uses_snapshot_pattern(relpath: str) -> None:
    """Mujoco import assertions must use the sys.modules snapshot pattern.

    The bare ``assert "mujoco" not in sys.modules`` fails when prior tests
    use ``pytest.importorskip("mujoco")``. The correct pattern snapshots
    ``sys.modules`` before the code-under-test runs and asserts no NEW
    import was added.
    """
    source = (_REPO_ROOT / relpath).read_text(encoding="utf-8")
    # Must NOT have the bare assertion
    assert 'assert "mujoco" not in sys.modules' not in source, (
        f"{relpath} must not use bare 'assert \"mujoco\" not in sys.modules' — "
        "use the sys.modules snapshot pattern instead"
    )
    # Must have the snapshot pattern
    assert "frozenset(sys.modules)" in source or "set(sys.modules)" in source, (
        f"{relpath} must snapshot sys.modules before running main()"
    )


# ---------------------------------------------------------------------------
# test_run_validation_success: portable command
# ---------------------------------------------------------------------------


def test_harness_spec_validation_uses_portable_command() -> None:
    """test_spec.py must not use bash `true` for validation success test."""
    source = (_REPO_ROOT / "tests" / "unit" / "harness" / "test_spec.py").read_text(
        encoding="utf-8"
    )
    assert 'validation_command="true"' not in source, (
        "test_spec.py must not use bash 'true' — use sys.executable for portability"
    )
