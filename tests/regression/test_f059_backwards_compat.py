# tests/regression/test_f059_backwards_compat.py
"""Backwards-compat pins for F-059: the catalog edits change selection, not any other feature."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Trunk tip this change branched from (F-052 Slices D + E, #247). The full 40-char SHA, not
# the 7-char prefix: an ambiguous prefix would make the missing-history skip below fire for
# the wrong reason.
_TRUNK = "f57d99b09ce219e8947dffa3d7db6a90b47d7b58"


def test_f050_f051_are_deferred_with_a_reason() -> None:
    """EX-3: deferred (reversible), not deleted; the reason is in notes."""
    raw = yaml.safe_load((_REPO_ROOT / "features.yaml").read_text(encoding="utf-8"))
    by_id = {f["id"]: f for f in raw["features"]}
    for fid in ("F-050", "F-051"):
        assert by_id[fid]["status"] == "deferred", fid
        assert "resume after F-008" in by_id[fid]["notes"], fid


def test_no_other_feature_changed_status() -> None:
    """Every feature except F-008 (edge only), F-050, F-051 and the new F-059 equals trunk."""
    import subprocess

    # The CI `test` job checks out at depth 1, so trunk is not in its history. Skipping is
    # deliberate: failing there would flag missing history, not a catalog regression.
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{_TRUNK}^{{commit}}"], capture_output=True, cwd=_REPO_ROOT
    )
    if probe.returncode != 0:
        pytest.skip(f"trunk commit {_TRUNK[:7]} is not in this clone's history (shallow checkout)")
    # encoding="utf-8", not text=True: text mode decodes with the locale codepage (cp1252 on
    # Windows), which mangles the em dashes and section signs in the catalog.
    trunk = subprocess.run(
        ["git", "show", f"{_TRUNK}:features.yaml"],
        capture_output=True,
        encoding="utf-8",
        check=True,
        cwd=_REPO_ROOT,
    ).stdout
    current = (_REPO_ROOT / "features.yaml").read_text(encoding="utf-8")
    before = {f["id"]: f for f in yaml.safe_load(trunk)["features"]}
    after = {f["id"]: f for f in yaml.safe_load(current)["features"]}
    for fid, feat in before.items():
        if fid in {"F-008", "F-050", "F-051"}:
            continue
        assert after[fid] == feat, fid
