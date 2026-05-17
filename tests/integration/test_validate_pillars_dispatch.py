"""Smoke-pass: end-to-end validate_all_pillars dispatch against mock_hardware."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mousedroid.config.schema import Settings
from mousedroid.validation.pillars import (
    PillarStatus,
    validate_all_pillars,
)

# Editable installs of the package may resolve to a sibling worktree
# (see user MEMORY.md: "pip -e can point to worktree, causing stale
# imports"). For the subprocess tests we MUST inject the current
# worktree's ``src/`` into PYTHONPATH so ``mousedroid.cli`` resolves
# to the new module instead of the install layer's older copy.
_WORKTREE_ROOT = Path(__file__).resolve().parents[2]
_WORKTREE_SRC = _WORKTREE_ROOT / "src"


def _subprocess_env() -> dict[str, str]:
    """Build a ``subprocess.run`` env that puts this worktree's ``src/`` first."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    sep = os.pathsep
    env["PYTHONPATH"] = f"{_WORKTREE_SRC}{sep}{existing}" if existing else str(_WORKTREE_SRC)
    return env


@pytest.mark.asyncio
async def test_pattern_a_pillars_pass_on_mock_hardware() -> None:
    """Pattern-A (factory-builder) pillars all complete successfully on mock_hardware."""
    cfg = Settings(mock_hardware=True)
    report = await validate_all_pillars(
        cfg,
        pillar_names={"safety", "world_model", "memory", "cognitive", "reward", "curiosity"},
    )
    # All six Pattern-A pillars must complete with OK or SKIPPED (no FAIL).
    fail = [r for r in report.results if r.status == PillarStatus.FAIL]
    assert fail == [], f"Pattern-A pillars failed: {fail}"


def test_cli_dry_run_via_subprocess_exits_zero() -> None:
    """``python -m mousedroid.cli.validate_pillars --dry-run`` exits 0 + prints all 10 pillars."""
    result = subprocess.run(
        [sys.executable, "-m", "mousedroid.cli.validate_pillars", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    for pillar in (
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
    ):
        assert pillar in result.stdout, f"missing pillar {pillar} in dry-run output"


def test_cli_filter_subset_via_subprocess_exits_zero() -> None:
    """``--pillars safety,memory --json`` returns JSON with only the two pillars."""
    import json
    import re

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mousedroid.cli.validate_pillars",
            "--dry-run",
            "--json",
            "--pillars",
            "safety,memory",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_subprocess_env(),
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    # Extract the trailing JSON block — structlog events also write to
    # stdout, so the JSON document isn't necessarily the only content.
    match = re.search(r"(\{[\s\S]*\})\s*$", result.stdout.strip())
    assert match is not None, f"no JSON object in stdout: {result.stdout!r}"
    payload = json.loads(match.group(1))
    assert {r["name"] for r in payload["results"]} == {"safety", "memory"}
