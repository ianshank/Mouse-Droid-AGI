"""PR #105b B.5 — regression guard: ``mypy --strict`` stays clean on touched files.

This test subprocess-runs ``mypy --strict --no-incremental`` against the
exact two source files PR #105b closed errors on
(``src/mousedroid/factory.py`` + ``src/mousedroid/reward/vlm_progress.py``)
and asserts the report is ``Success: no issues found``. Without this guard
a future PR could re-introduce either the ``PolicyApprovalGate`` type hole
or the ``cachetools`` missing-stub error and CI's broader ``mypy --strict``
job wouldn't flag it specifically — operators would have to bisect.

The test is :pyattr:`pytest.mark.slow` because a cold ``mypy --strict``
run typically takes 30-60 s on this codebase. The default test sweep
(``pytest -m "not slow"``) skips it; CI's ``typecheck`` job runs it via
``pytest -m slow tests/regression/test_pr105b_mypy_clean.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# PR-105b harden gap-fix #3: pull the subprocess timeout out of the test
# body into a module-level constant with an env-var override. Slow CI
# runners (or the Jetson Orin Nano if anyone ever runs this on-device)
# can extend via ``MYPY_TIMEOUT_S=600 pytest tests/regression/test_pr105b_mypy_clean.py``.
# Default 300 s preserves the workstation-measured 282 s + ~30-40 %
# headroom that the earlier comment documented.
_MYPY_TIMEOUT_S: Final[int] = int(os.environ.get("MYPY_TIMEOUT_S", "300"))

# The two files PR #105b explicitly closed mypy errors on. Pinning them
# individually (rather than running mypy on the whole tree) keeps the
# test fast — single-file mypy is ~5 s vs whole-tree ~60 s — AND scopes
# the assertion to the exact files the guard targets. If a future PR
# adds NEW mypy errors elsewhere in the tree, CI's broader typecheck
# job catches those; this guard catches re-regression of the PR-98 +
# PR-105b debt specifically.
_TARGET_FILES: Final[tuple[str, ...]] = (
    "src/mousedroid/factory.py",
    "src/mousedroid/reward/vlm_progress.py",
)

# Allow the operator to override the python interpreter (matches the
# pattern at ``scripts/ci.sh`` which honours ``PYTHON``). Default to the
# interpreter running the test, which is the same one mypy would use.
_PYTHON_EXE: Final[str] = sys.executable


@pytest.mark.slow
def test_targeted_files_mypy_strict_clean() -> None:
    """``mypy --strict`` on the PR-105b-touched files returns 0 errors.

    This is a guard against silent re-introduction of:

    * ``factory.py:2216-2261`` — the ``PolicyApprovalGate(inner, ...)``
      argument-type hole (originally PR-98, closed in PR-105b via
      ``inner: ApprovalGateProtocol`` annotation tightening).
    * ``reward/vlm_progress.py:31`` — the ``cachetools`` missing-stub
      ``[import-untyped]`` error (closed in PR-105b by adding
      ``types-cachetools`` to the ``[dev]`` extras).

    If a future PR weakens either fix, this test fails with the mypy
    output so the regression's file:line is obvious.
    """
    target_paths = [str(_REPO_ROOT / rel) for rel in _TARGET_FILES]

    # Timeout sourced from ``_MYPY_TIMEOUT_S`` (env-overridable; see
    # module docstring). Sized for a cold ``mypy --strict`` run on this
    # codebase's import graph — factory.py transitively pulls in the
    # world_model + arm + harness trees, which dominates wall time.
    # S603 noqa not required: tests/** has a broad subprocess-call waiver
    # in pyproject.toml's per-file-ignores.
    result = subprocess.run(
        [_PYTHON_EXE, "-m", "mypy", "--strict", "--no-incremental", *target_paths],
        capture_output=True,
        text=True,
        timeout=_MYPY_TIMEOUT_S,
        cwd=str(_REPO_ROOT),
        check=False,
    )

    # Surface the full mypy output on failure so the operator can read
    # the regression directly without re-running the subprocess. ``stderr``
    # carries the unused-section warnings; the actual error reporting
    # goes to ``stdout``.
    assert result.returncode == 0, (
        f"mypy --strict found errors on PR-105b-touched files. "
        f"Exit code {result.returncode}.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "Success: no issues found" in result.stdout, (
        f"mypy returned 0 but the success marker is missing.\n" f"--- stdout ---\n{result.stdout}"
    )
