# tests/regression/test_numpy_mypy_target_compat.py
"""Regression: numpy stub grammar must stay parseable under the mypy target.

numpy 2.5.0 (requires-python >=3.12) ships type stubs using PEP 695 ``type``
statements. mypy parses dependency stubs with the grammar of
``[tool.mypy].python_version`` — NOT the running interpreter — so with the
repo-wide 3.10 target, any environment that resolves numpy >=2.5 aborts type
checking inside ``numpy/__init__.pyi`` before a single repo file is checked
(the exact CI breakage fixed on PR #150).

This test encodes the *invariant*, not the literal pin: while the configured
mypy target is below the first Python version whose grammar has ``type``
statements, the numpy requirement must exclude every release that ships them.
If the mypy target moves to >=3.12 (or to per-interpreter targeting), the test
passes with or without the cap — the bound can then be lifted without touching
this file.
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: stdlib tomllib lands in 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from packaging.specifiers import SpecifierSet
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
# First Python whose grammar accepts PEP 695 `type` statements.
_PEP695_MIN_TARGET = Version("3.12")
# First numpy release whose stubs require that grammar.
_FIRST_PEP695_NUMPY = Version("2.5.0")


def _load_pyproject() -> dict[str, object]:
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _numpy_specifier(data: dict[str, object]) -> SpecifierSet:
    project = data["project"]
    assert isinstance(project, dict)
    deps = project["dependencies"]
    assert isinstance(deps, list)
    numpy_reqs = [d for d in deps if isinstance(d, str) and d.startswith("numpy")]
    assert len(numpy_reqs) == 1, f"expected exactly one numpy requirement, got {numpy_reqs}"
    return SpecifierSet(numpy_reqs[0].removeprefix("numpy"))


def test_numpy_constraint_compatible_with_mypy_target() -> None:
    data = _load_pyproject()
    tool = data.get("tool")
    assert isinstance(tool, dict)
    mypy_cfg = tool.get("mypy")
    assert isinstance(mypy_cfg, dict)
    target = mypy_cfg.get("python_version")
    if target is None or Version(str(target)) >= _PEP695_MIN_TARGET:
        # Per-interpreter or >=3.12 target parses PEP 695 stubs fine — the
        # invariant imposes no numpy bound.
        return
    spec = _numpy_specifier(data)
    assert not spec.contains(_FIRST_PEP695_NUMPY), (
        f"[tool.mypy] python_version = {target} cannot parse numpy "
        f">={_FIRST_PEP695_NUMPY} PEP 695 stubs, but the numpy requirement "
        f"({spec}) admits it — mypy would abort inside numpy/__init__.pyi on "
        f"any Python >=3.12 environment. Re-add an upper bound or move the "
        f"mypy target to >=3.12."
    )


def test_numpy_specifier_still_admits_supported_range() -> None:
    # The cap must not over-tighten: the floor of the supported range and the
    # last pre-PEP695 minor must both stay installable.
    spec = _numpy_specifier(_load_pyproject())
    for version in ("1.26.4", "2.4.0"):
        assert spec.contains(Version(version)), f"numpy {version} unexpectedly excluded by {spec}"
