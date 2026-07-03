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

from packaging.version import Version

from tests._pyproject import load_pyproject, numpy_specifier

# First Python whose grammar accepts PEP 695 `type` statements.
_PEP695_MIN_TARGET = Version("3.12")
# Representative releases whose stubs require that grammar: the first one,
# a patch, and a later minor — so a point exclusion (!=2.5.0) that would
# still admit 2.5.1/2.6 cannot satisfy the invariant.
_PEP695_NUMPY_VERSIONS = ("2.5.0", "2.5.1", "2.6.0")


def test_numpy_constraint_compatible_with_mypy_target() -> None:
    data = load_pyproject()
    tool = data.get("tool")
    assert isinstance(tool, dict)
    mypy_cfg = tool.get("mypy")
    assert isinstance(mypy_cfg, dict)
    target = mypy_cfg.get("python_version")
    if target is None or Version(str(target)) >= _PEP695_MIN_TARGET:
        # Per-interpreter or >=3.12 target parses PEP 695 stubs fine — the
        # invariant imposes no numpy bound.
        return
    spec = numpy_specifier(data)
    admitted = [v for v in _PEP695_NUMPY_VERSIONS if spec.contains(Version(v))]
    assert not admitted, (
        f"[tool.mypy] python_version = {target} cannot parse the PEP 695 "
        f"stubs numpy ships from 2.5.0 on, but the numpy requirement ({spec}) "
        f"admits {admitted} — mypy would abort inside numpy/__init__.pyi on "
        f"any Python >=3.12 environment. Re-add an upper bound (a point "
        f"exclusion is not enough) or move the mypy target to >=3.12."
    )


def test_numpy_specifier_still_admits_supported_range() -> None:
    # The cap must not over-tighten: the floor of the supported range and the
    # last pre-PEP695 minor must both stay installable.
    spec = numpy_specifier()
    for version in ("1.26.4", "2.4.0"):
        assert spec.contains(Version(version)), f"numpy {version} unexpectedly excluded by {spec}"
