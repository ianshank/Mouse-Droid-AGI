"""Regression: the pinned ruff version has one source of truth.

``ruff`` is pinned in ``[project.optional-dependencies] dev`` (pyproject.toml)
but re-stated as a standalone literal in two workflows, because their lint
jobs ``pip install`` ruff directly rather than installing the project's dev
extra:

- ``.github/workflows/ci.yml`` — ``pip install "ruff==<V>"`` (lint job)
- ``.github/workflows/release.yml`` — ``pip install "ruff==<V>"`` (lint job)

Nothing asserted they agree. A bump in one spot silently forks the linter:
local ``make install`` resolves the pyproject pin while CI keeps its own, and
the two disagree about formatting and rule behaviour in both directions. This
is not hypothetical — the project has already lost time to exactly this skew
(``S603`` in ``tools/_jetson_helpers.py`` going red repo-wide), and a
dependency-bot PR that touches only ``pyproject.toml`` reintroduces it by
construction.

Mirrors ``test_coverage_gate_single_source.py``, which guards the coverage
threshold the same way.

``mypy`` is deliberately NOT asserted here: it is pinned only in
``pyproject.toml`` and every workflow installs it transitively via the dev
extra, so it already has a single source of truth and no literal to drift.
"""

from __future__ import annotations

import re
from pathlib import Path

from packaging.requirements import Requirement

from tests._pyproject import load_pyproject

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Workflows whose lint job installs ruff standalone instead of via the dev extra.
_WORKFLOWS_PINNING_RUFF = ("ci.yml", "release.yml")

_RUFF_PIN_RE = re.compile(r'ruff==([0-9][^"\'\s]*)')


def _pyproject_ruff_version() -> str:
    """Return the exact ruff version pinned in the ``dev`` extra."""
    project = load_pyproject()["project"]
    assert isinstance(project, dict)
    extras = project["optional-dependencies"]
    assert isinstance(extras, dict)
    dev = extras["dev"]
    assert isinstance(dev, list)

    ruff_reqs = [
        Requirement(d) for d in dev if isinstance(d, str) and Requirement(d).name == "ruff"
    ]
    assert len(ruff_reqs) == 1, f"expected exactly one ruff requirement in [dev], got {ruff_reqs}"

    specifier = str(ruff_reqs[0].specifier)
    assert specifier.startswith("=="), (
        f"ruff must be pinned with '==' so CI and local agree exactly, got {specifier!r}"
    )
    return specifier.removeprefix("==")


def test_workflow_ruff_pins_match_pyproject() -> None:
    pinned = _pyproject_ruff_version()

    for workflow in _WORKFLOWS_PINNING_RUFF:
        text = (_REPO_ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        found = _RUFF_PIN_RE.findall(text)
        assert found, (
            f"{workflow} no longer pins a ruff version — if its lint job now "
            f"installs the dev extra instead, drop it from _WORKFLOWS_PINNING_RUFF"
        )
        assert all(v == pinned for v in found), (
            f"{workflow} ruff pin {found} != pyproject dev-extra pin {pinned!r}; "
            f"bump every literal in the same change"
        )
