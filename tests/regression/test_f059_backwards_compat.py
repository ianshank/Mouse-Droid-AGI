# tests/regression/test_f059_backwards_compat.py
"""Backwards-compat pins for F-059: the F-050/F-051 disposition.

Each is either deferred with the reason recorded in its notes, or legitimately resumed after
F-008. Neither may sit silently at todo/blocked. No other feature's entry is frozen here: the
catalog is living, so such a snapshot would go red on the next legitimate edit.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_f050_f051_are_deferred_with_a_reason_or_resumed() -> None:
    """EX-3: deferred with the reason recorded, or legitimately resumed after F-008 —
    never silently todo/blocked.
    """
    raw = yaml.safe_load((_REPO_ROOT / "features.yaml").read_text(encoding="utf-8"))
    by_id = {f["id"]: f for f in raw["features"]}
    for fid in ("F-050", "F-051"):
        status = by_id[fid]["status"]
        if status == "deferred":
            assert "resume after F-008" in by_id[fid]["notes"], fid
        else:
            assert status in {"in_progress", "done"}, (fid, status)
