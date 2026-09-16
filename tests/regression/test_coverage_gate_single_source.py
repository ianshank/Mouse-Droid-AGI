"""Regression: the 90% coverage threshold has one source of truth.

The repo-wide line-coverage gate is declared in ``[tool.coverage.report]
fail_under`` (pyproject.toml) but re-stated as a literal in five other
places. Nothing previously asserted they agree, so a bump in one spot would
silently fork the gate:

- ``scripts/ci.sh`` — ``--cov-fail-under=<N>`` (unit+property+integration
  stage) and ``check_branch_coverage.py --min <N>``
- ``.github/workflows/ci.yml`` — ``MINIMUM_COVERAGE: "<N>"`` env
- ``.github/workflows/release.yml`` — ``MINIMUM_COVERAGE: "<N>"`` env
- ``Makefile`` — ``COV_MIN := <N>``, which feeds both ``make test-cov`` and
  ``make branch-coverage``. Added after review noted that the Makefile literal
  was outside this check, so local enforcement could sit at a different
  threshold than CI while this test stayed green.

``.claude/workforce.yaml``'s ``coverage.tools_line_min`` is a DIFFERENT gate
(tools/claude_hooks, its own invocation) and is deliberately NOT asserted
equal here — it tracks its own threshold and may lag this one.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests._pyproject import load_pyproject

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _fail_under() -> int:
    report_cfg = load_pyproject()["tool"]["coverage"]["report"]  # type: ignore[index]
    fail_under = report_cfg["fail_under"]
    assert isinstance(fail_under, int)
    return fail_under


def test_ci_sh_matches_pyproject_fail_under() -> None:
    text = (_REPO_ROOT / "scripts" / "ci.sh").read_text(encoding="utf-8")
    gate = _fail_under()

    cov_flags = re.findall(r"--cov-fail-under=(\d+)", text)
    assert cov_flags, "ci.sh no longer passes --cov-fail-under — update this test"
    assert all(int(v) == gate for v in cov_flags), (
        f"ci.sh --cov-fail-under {cov_flags} != pyproject fail_under {gate}"
    )

    min_flags = re.findall(r"check_branch_coverage\.py --min (\d+)", text)
    assert min_flags, "ci.sh no longer runs check_branch_coverage.py --min — update this test"
    assert all(int(v) == gate for v in min_flags), (
        f"ci.sh check_branch_coverage --min {min_flags} != pyproject fail_under {gate}"
    )


def test_workflow_envs_match_pyproject_fail_under() -> None:
    gate = _fail_under()
    for workflow in ("ci.yml", "release.yml"):
        text = (_REPO_ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        envs = re.findall(r"MINIMUM_COVERAGE:\s*\"(\d+)\"", text)
        assert envs, f"{workflow} no longer declares MINIMUM_COVERAGE — update this test"
        assert all(int(v) == gate for v in envs), (
            f"{workflow} MINIMUM_COVERAGE {envs} != pyproject fail_under {gate}"
        )


def test_makefile_cov_min_matches_pyproject_fail_under() -> None:
    """``COV_MIN`` drives the local gate, so it must not fork from pyproject.

    It is consumed twice (``make test-cov``'s ``--cov-fail-under`` and
    ``make branch-coverage``'s ``--min``), so a drifted value makes the whole
    local ladder enforce a different bar than CI — the precise failure this file
    exists to prevent, on the one surface it did not yet cover.
    """
    text = (_REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    gate = _fail_under()

    declared = re.findall(r"^COV_MIN\s*:?=\s*(\d+)", text, flags=re.MULTILINE)
    assert declared, "Makefile no longer declares COV_MIN — update this test"
    assert all(int(v) == gate for v in declared), (
        f"Makefile COV_MIN {declared} != pyproject fail_under {gate}"
    )

    # And it must actually be *used* via the variable, not re-hardcoded beside it.
    assert "--cov-fail-under=$(COV_MIN)" in text, (
        "make test-cov no longer reads COV_MIN — a literal there would re-fork the gate"
    )
    assert "--min $(COV_MIN)" in text, (
        "make branch-coverage no longer reads COV_MIN — same fork risk"
    )
