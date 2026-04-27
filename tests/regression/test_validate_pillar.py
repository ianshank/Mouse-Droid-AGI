"""Regression tests for scripts/validate_pillar.sh.

Verifies structural invariants so that pillar names, blocking defaults, and
the summary-writing contract are not accidentally broken by future edits.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SCRIPT = Path("scripts/validate_pillar.sh")

EXPECTED_PILLARS = [
    "safety",
    "world_model",
    "memory",
    "cognitive",
    "reward",
    "curiosity",
    "continual",
    "meta",
    "scaling",
    "growth",
]

# Pillars that must default to blocking=yes (core runtime path).
BLOCKING_PILLARS = {"safety", "world_model", "memory", "cognitive", "reward"}

# Pillars that must default to blocking=no (training/exploration path).
NON_BLOCKING_PILLARS = {"curiosity", "continual", "meta", "scaling", "growth"}


@pytest.fixture(scope="module")
def script_text() -> str:
    if not _SCRIPT.exists():
        pytest.skip(f"{_SCRIPT} not present in this checkout")
    return _SCRIPT.read_text()


# ---------------------------------------------------------------------------
# All 10 pillars are present
# ---------------------------------------------------------------------------


def test_all_ten_pillars_have_case_entry(script_text: str) -> None:
    """Every pillar in EXPECTED_PILLARS must appear as a case branch."""
    for pillar in EXPECTED_PILLARS:
        pattern = rf"^\s+{re.escape(pillar)}\)"
        assert re.search(
            pattern, script_text, re.MULTILINE
        ), f"Pillar '{pillar}' has no case branch in {_SCRIPT}"


def test_all_pillars_listed_in_usage_help(script_text: str) -> None:
    """The usage/help text must mention every pillar so operators know valid names."""
    for pillar in EXPECTED_PILLARS:
        assert pillar in script_text, (
            f"Pillar '{pillar}' not mentioned anywhere in {_SCRIPT} "
            "(at minimum it must appear in the usage hint)"
        )


# ---------------------------------------------------------------------------
# Blocking defaults
# ---------------------------------------------------------------------------


def _blocking_flag_for_pillar(text: str, pillar: str) -> str | None:
    """Return the first yes/no argument on the run_pillar_check call for a pillar."""
    # Pattern: run_pillar_check <pillar> <yes|no> …
    # We look for the line immediately following the case label.
    pattern = rf"run_pillar_check\s+{re.escape(pillar)}\s+(yes|no)\b"
    m = re.search(pattern, text)
    return m.group(1) if m else None


def test_blocking_pillars_default_yes(script_text: str) -> None:
    for pillar in BLOCKING_PILLARS:
        flag = _blocking_flag_for_pillar(script_text, pillar)
        assert flag == "yes", f"Pillar '{pillar}' expected blocking default 'yes', got {flag!r}"


def test_non_blocking_pillars_default_no(script_text: str) -> None:
    for pillar in NON_BLOCKING_PILLARS:
        flag = _blocking_flag_for_pillar(script_text, pillar)
        assert flag == "no", f"Pillar '{pillar}' expected blocking default 'no', got {flag!r}"


# ---------------------------------------------------------------------------
# Summary / report contract
# ---------------------------------------------------------------------------


def test_summary_writes_ten_pillars_log(script_text: str) -> None:
    """The script must write a ten_pillars.log (Markdown table) to REPORT_DIR."""
    assert (
        "ten_pillars.log" in script_text
    ), f"{_SCRIPT} does not write ten_pillars.log; SUMMARY.md integration will break"


def test_summary_table_has_three_columns(script_text: str) -> None:
    """The Markdown table header must define Pillar, Status, and Note columns."""
    assert (
        "| Pillar | Status | Note |" in script_text
    ), f"{_SCRIPT} summary table header does not match expected format"


def test_overall_pass_fail_line_present(script_text: str) -> None:
    """The summary must include an 'Overall: PASS' / 'Overall: FAIL' line."""
    assert (
        "Overall: PASS" in script_text
    ), f"{_SCRIPT} does not emit 'Overall: PASS' line in summary"
    assert (
        "Overall: FAIL" in script_text
    ), f"{_SCRIPT} does not emit 'Overall: FAIL' line in summary"


# ---------------------------------------------------------------------------
# python3-in-container shim fallback
# ---------------------------------------------------------------------------


def test_fallback_shim_created_when_env_var_absent(script_text: str) -> None:
    """When MOUSEDROID_SMOKE_PYTHON is unset the script must create its own shim."""
    assert (
        "python3-in-container" in script_text
    ), f"{_SCRIPT} does not create a python3-in-container shim fallback"


# ---------------------------------------------------------------------------
# SUMMARY.md integration in jetson_full_smoke_run.sh
# ---------------------------------------------------------------------------


def test_full_smoke_run_appends_ten_pillars_table() -> None:
    """jetson_full_smoke_run.sh must append ten_pillars.log to SUMMARY.md when present."""
    full_run = Path("scripts/jetson_full_smoke_run.sh")
    if not full_run.exists():
        pytest.skip("jetson_full_smoke_run.sh not present")
    text = full_run.read_text()
    assert "ten_pillars.log" in text, (
        "jetson_full_smoke_run.sh does not reference ten_pillars.log; "
        "Ten Pillars results will not appear in SUMMARY.md"
    )
    assert (
        "Ten Pillars Validation" in text
    ), "jetson_full_smoke_run.sh SUMMARY section missing 'Ten Pillars Validation' heading"
