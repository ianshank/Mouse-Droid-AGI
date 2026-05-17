"""Verify numpy has a defensive upper bound (NumPy 2.x removed np.float_, np.NAN, ...)."""

from __future__ import annotations

import re
from pathlib import Path


def test_numpy_has_defensive_upper_bound() -> None:
    """numpy spec must lock out NumPy 2.x or have explicit exclusions."""
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    match = re.search(r'"numpy[^"]*"', pyproject)
    assert match is not None, "numpy dep line not found in pyproject.toml"
    spec = match.group(0)
    has_upper_bound = "<2" in spec
    has_exclusion = "!=2." in spec
    assert has_upper_bound or has_exclusion, (
        f"numpy is missing an upper bound or version exclusion, got {spec!r}. "
        "NumPy 2.x removed np.float_/np.NAN/np.in1d — either pin <2.0 "
        "or exclude specific 2.x versions to lock the transitive surface."
    )
