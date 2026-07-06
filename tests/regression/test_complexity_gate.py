"""Regression guard: the C901 cyclomatic-complexity gate stays enforced.

ADR-014 introduced a McCabe complexity ceiling (``C901`` at
``max-complexity = 15``) and decomposed every ``src/`` offender below it, so
the ``src/`` C901 baseline is **empty**. This guard pins three things a future
change could silently weaken:

1. ``C901`` remains in the ruff ``select`` set and ``max-complexity`` stays 15.
2. No ``src/mousedroid/**`` path carries a file-level ``C901`` ``per-file-ignore``
   (the ratchet must not be re-opened to dodge a decomposition).
3. No ``src/`` function actually exceeds the ceiling — proven behaviourally by
   running ``ruff check src/ --select C901`` under the repo config.

Text assertions (not ``tomllib``) keep this green on the 3.10 CI leg without a
parser shim; the behavioural check is a fast (~1-2 s) ruff subprocess.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Final

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_PYPROJECT: Final[str] = (_REPO_ROOT / "pyproject.toml").read_text()


def test_c901_is_selected() -> None:
    """``C901`` is present in the ruff lint select set."""
    # Grab the select = [ ... ] block (closing ] is on its own line; an inline
    # comment inside the block may itself contain a ']', so anchor on "\n]").
    match = re.search(r"select\s*=\s*\[(.*?)\n\]", _PYPROJECT, re.DOTALL)
    assert match is not None, "could not locate [tool.ruff.lint] select list"
    assert '"C901"' in match.group(1), "C901 must stay in the ruff select set"


def test_mccabe_max_complexity_is_15() -> None:
    """The McCabe ceiling is pinned at 15 (ADR-014's deliberate threshold)."""
    match = re.search(
        r"\[tool\.ruff\.lint\.mccabe\].*?max-complexity\s*=\s*(\d+)",
        _PYPROJECT,
        re.DOTALL,
    )
    assert match is not None, "missing [tool.ruff.lint.mccabe] max-complexity"
    assert match.group(1) == "15", "max-complexity must stay 15 (see ADR-014)"


def test_no_src_c901_baseline() -> None:
    """No ``src/mousedroid`` file carries a C901 per-file-ignore.

    The ``src/`` backlog was fully cleared; re-adding a file-level C901 ignore
    to work around a complex function (instead of decomposing it) re-opens the
    ratchet and must fail here. The ``scripts/`` glob C901 ignore is allowed.
    """
    offenders = [
        line.strip()
        for line in _PYPROJECT.splitlines()
        if line.strip().startswith('"src/mousedroid/') and "C901" in line
    ]
    assert not offenders, f"src/ must carry no C901 baseline; found: {offenders}"


def test_src_tree_has_no_c901_violations() -> None:
    """No ``src/`` function exceeds the complexity ceiling under the repo config."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/", "--select", "C901"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        "ruff C901 gate found a src/ function over the complexity ceiling — "
        "decompose it rather than baselining.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
