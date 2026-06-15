#!/usr/bin/env python3
"""Print the next feature an agent should work on, honoring depends_on.

Thin CLI shim over :mod:`mousedroid.harness.spec` (ADR-012); the DAG-aware
selection logic lives in the importable, unit-tested package module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mousedroid.harness import spec  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("features", nargs="?", default=str(_REPO_ROOT / "features.yaml"))
    args = ap.parse_args(argv)

    feats = spec.load_features(args.features)
    sel = spec.select_next(feats)

    if sel.kind in ("resume", "ready") and sel.feature is not None:
        f = sel.feature
        if sel.kind == "resume":
            print(f"{f['id']}  {f['name']}  (in_progress — resume)")
        else:
            tier = f.get("tier", spec.DEFAULT_TIER)
            print(f"{f['id']}  {f['name']}  (priority={f['priority']}, tier={tier})")
        return 0
    if sel.kind == "blocked":
        print("No feature is ready. Unmet dependencies:")
        for fid, missing in sel.blocked:
            print(f"  {fid} waits on {missing}")
        return 2
    print("No todo features. Run validate.py to confirm completion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
