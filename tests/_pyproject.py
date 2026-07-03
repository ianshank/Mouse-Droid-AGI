"""Shared helpers for asserting on ``pyproject.toml`` contents in tests.

Two suites guard invariants about the numpy requirement —
``tests/unit/test_numpy_pin.py`` (defensive upper bound) and
``tests/regression/test_numpy_mypy_target_compat.py`` (PEP 695 stub grammar vs
the mypy target). They previously used two divergent extraction strategies
(a raw-text regex and an inline tomllib walk), so a relocation or reshaping of
the requirement could break one while the other silently kept passing. This
module is the single parser both consume.

Follows the repo's ``tests/_<name>.py`` shared-helper convention (cf.
``tests/_script_loader.py``, ``tests/_jetson_hardware.py``); import as
``from tests._pyproject import numpy_specifier``.
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: stdlib tomllib lands in 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def load_pyproject() -> dict[str, object]:
    """Parse and return the repo's ``pyproject.toml``."""
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def numpy_requirement(data: dict[str, object] | None = None) -> Requirement:
    """Return the single numpy requirement from ``[project] dependencies``.

    Matches by parsed requirement name (``numpy-quaternion`` etc. never
    match), and asserts exactly one entry exists — zero or duplicates both
    mean the invariant tests are no longer looking at the real constraint.
    """
    project = (data or load_pyproject())["project"]
    assert isinstance(project, dict)
    deps = project["dependencies"]
    assert isinstance(deps, list)
    reqs = [Requirement(d) for d in deps if isinstance(d, str)]
    numpy_reqs = [r for r in reqs if r.name == "numpy"]
    assert len(numpy_reqs) == 1, f"expected exactly one numpy requirement, got {numpy_reqs}"
    return numpy_reqs[0]


def numpy_specifier(data: dict[str, object] | None = None) -> SpecifierSet:
    """Return the numpy requirement's version specifier set."""
    return numpy_requirement(data).specifier
