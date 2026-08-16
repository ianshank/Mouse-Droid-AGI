"""Regression: cap ``# hardcoded-ok`` waiver markers in src/.

``scripts/check_no_hardcoded_values.py`` (invariant 3's enforcement
mechanism) only scans *changed* lines and accepts an unconditional
``# hardcoded-ok`` / ``# noqa: PLR2004`` suppression on any line — unlike
the ruff ``noqa``/``type: ignore`` budget in ``test_suppression_budget.py``,
nothing capped how many such waivers could accumulate. This ratchets the
``# hardcoded-ok`` marker the same way: update the budget DOWN as waivers
are resolved; never raise it without a documented reason.

``# noqa: PLR2004`` is excluded — ``PLR2004`` isn't in ruff's enabled rule
set (see pyproject.toml ``[tool.ruff.lint] select``), so that marker is
currently inert and has zero live occurrences to budget.
"""

from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "mousedroid"
# Budget = the measured count at the time this ratchet was added: 23
# occurrences across config/migration.py, training/domain_randomization.py,
# training/replay/mixer.py, validation/{latency_stats,report_store,summary}.py,
# reward/vlm_progress.py, comms/command_set.py. May only ratchet DOWN.
_MAX_HARDCODED_OK = 23


def _count_hardcoded_ok() -> int:
    return sum(
        line.count("# hardcoded-ok")
        for p in _SRC.rglob("*.py")
        for line in p.read_text(encoding="utf-8").splitlines()
    )


def test_hardcoded_ok_marker_within_budget() -> None:
    assert _count_hardcoded_ok() <= _MAX_HARDCODED_OK
