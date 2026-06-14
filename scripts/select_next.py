#!/usr/bin/env python3
"""Print the next feature an agent should work on, honoring depends_on."""

from __future__ import annotations

import sys
from typing import Any

import yaml

PRIORITY = {"critical": 0, "high": 1, "medium": 2, "low": 3}

Feature = dict[str, Any]


def main(path: str = "features.yaml") -> int:
    with open(path) as fh:
        feats: list[Feature] = yaml.safe_load(fh)["features"]
    by_id = {f["id"]: f for f in feats}

    def deps_done(f: Feature) -> bool:
        return all(by_id.get(d, {}).get("status") == "done" for d in f.get("depends_on", []))

    inprog = [f for f in feats if f["status"] == "in_progress"]
    if inprog:
        f = sorted(inprog, key=lambda x: (PRIORITY[x["priority"]], x["id"]))[0]
        print(f"{f['id']}  {f['name']}  (in_progress — resume)")
        return 0

    ready = [f for f in feats if f["status"] == "todo" and deps_done(f)]
    if not ready:
        blocked = [f for f in feats if f["status"] == "todo" and not deps_done(f)]
        if blocked:
            print("No feature is ready. Unmet dependencies:")
            for f in blocked:
                missing = [d for d in f["depends_on"] if by_id.get(d, {}).get("status") != "done"]
                print(f"  {f['id']} waits on {missing}")
            return 2
        print("No todo features. Run validate.py to confirm completion.")
        return 0

    f = sorted(ready, key=lambda x: (PRIORITY[x["priority"]], x["id"]))[0]
    print(f"{f['id']}  {f['name']}  (priority={f['priority']}, tier={f.get('tier', 'fast')})")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
