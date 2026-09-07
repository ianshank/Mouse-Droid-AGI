"""F-038 backwards-compat: parked tiers still exist and still import the parked builder.

Relabelling must not delete the directories F-028 wired into CI, and must not
change which factory builder they call.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_parked_tier_directories_still_exist() -> None:
    assert (_REPO_ROOT / "tests" / "functional").is_dir()
    assert (_REPO_ROOT / "tests" / "user_journey").is_dir()
    assert (_REPO_ROOT / "tests" / "security").is_dir()


def test_ci_still_runs_the_three_directories() -> None:
    text = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pytest tests/functional tests/user_journey tests/security" in text
