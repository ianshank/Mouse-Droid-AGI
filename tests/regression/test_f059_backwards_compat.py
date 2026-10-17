# tests/regression/test_f059_backwards_compat.py
"""Backwards-compat pins for F-059: the catalog edits change selection, not any other feature."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_f050_f051_are_deferred_with_a_reason() -> None:
    """EX-3: deferred (reversible), not deleted; the reason is in notes."""
    raw = yaml.safe_load((_REPO_ROOT / "features.yaml").read_text(encoding="utf-8"))
    by_id = {f["id"]: f for f in raw["features"]}
    for fid in ("F-050", "F-051"):
        assert by_id[fid]["status"] == "deferred", fid
        assert "resume after F-008" in by_id[fid]["notes"], fid
