"""Verify numpy always carries a real (semantic) upper bound.

History: NumPy 2.0 removed ``np.float_``/``np.NAN``/``np.in1d`` (hence the
``!=2.0.0,!=2.0.1`` exclusions), and numpy 2.5.0 shipped PEP 695 ``type``-
statement stubs the repo's mypy 3.10 target cannot parse (hence the current
``<2.5`` cap — see ``tests/regression/test_numpy_mypy_target_compat.py``).

Policy (deliberately stricter than the original either/or wording): the
specifier must BOTH exclude the known-breaking 2.0.x releases AND be bounded
above, asserted semantically via ``SpecifierSet.contains`` — never by operator
substring sniffing (the old ``"<2" in spec`` check was satisfied by accident
via the ``<2.5`` cap). When the mypy 3.10 target is retired and the ``<2.5``
cap is lifted, move the bound UP (e.g. ``<3``) — don't delete it; an unbounded
numpy is how the next silent stub/API break reaches CI.
"""

from __future__ import annotations

from packaging.version import Version

from tests._pyproject import numpy_specifier


def test_numpy_excludes_known_breaking_releases() -> None:
    """The 2.0.x releases that removed np.float_/np.NAN must never resolve."""
    spec = numpy_specifier()
    admitted = [v for v in ("2.0.0", "2.0.1") if spec.contains(Version(v))]
    assert not admitted, (
        f"numpy requirement ({spec}) admits known-breaking releases {admitted} "
        "(NumPy 2.0 removed np.float_/np.NAN/np.in1d)."
    )


def test_numpy_has_defensive_upper_bound() -> None:
    """The specifier must be semantically bounded above.

    ``not contains("999.0.0")`` is the form-agnostic check: it holds for
    ``<2.5``, ``<3``, ``<=2.4.6`` alike, and fails for any specifier that lets
    arbitrary future majors resolve. If this fires after lifting the ``<2.5``
    mypy-era cap, re-add a higher bound (e.g. ``<3``) instead of deleting it.
    """
    spec = numpy_specifier()
    assert not spec.contains(Version("999.0.0")), (
        f"numpy requirement ({spec}) has no effective upper bound — arbitrary "
        "future majors would resolve straight into CI. Bound it (e.g. <3)."
    )
