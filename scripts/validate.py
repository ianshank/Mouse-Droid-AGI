#!/usr/bin/env python3
"""Validate features.yaml structure, DAG integrity, git provenance, and that
every `done` feature's validation_command passes for the selected tier(s).

    python scripts/validate.py --tier fast
    python scripts/validate.py --tier fast,slow
    python scripts/validate.py --tier fast,slow,hardware --strict-git
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

import yaml

DEFAULT_TIER = "fast"

Feature = dict[str, Any]


def load(path: str) -> list[Feature]:
    with open(path) as fh:
        return yaml.safe_load(fh)["features"]


def check_schema(feats: list[Feature], schema_path: str) -> list[str]:
    try:
        import jsonschema
    except ImportError:
        print("[warn] jsonschema not installed; skipping structural check")
        return []
    with open(schema_path) as fh:
        schema = json.load(fh)
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"schema: {list(e.path)}: {e.message}"
        for e in validator.iter_errors({"features": feats})
    ]


def check_dag(feats: list[Feature]) -> list[str]:
    by_id = {f["id"]: f for f in feats}
    errs: list[str] = []
    for f in feats:
        for d in f.get("depends_on", []):
            if d not in by_id:
                errs.append(f"dag: {f['id']} depends_on unknown id {d}")

    white, grey, black = 0, 1, 2
    color = {f["id"]: white for f in feats}

    def visit(node: str, stack: list[str]) -> None:
        color[node] = grey
        for d in by_id[node].get("depends_on", []):
            if d not in by_id:
                continue
            if color[d] == grey:
                # e.g. F-001 -> F-002 -> F-001
                errs.append("dag: cycle " + " -> ".join([*stack, d]))
            elif color[d] == white:
                visit(d, [*stack, d])
        color[node] = black

    for f in feats:
        if color[f["id"]] == white:
            visit(f["id"], [f["id"]])
    return errs


def git_rev_ok(ref: str | None) -> bool:
    if not ref:
        return False
    r = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],  # noqa: S607
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def run_validation(f: Feature) -> str | None:
    cmd = f.get("validation_command")
    if not cmd:
        return f"{f['id']}: status=done but no validation_command"
    # shell=True is intentional: validation_command is an operator-authored shell
    # string (HARNESS_SPEC.md §5), not untrusted input.
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)  # noqa: S602
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        return f"{f['id']}: validation_command failed ({r.returncode})\n      " + "\n      ".join(
            tail
        )
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.yaml")
    ap.add_argument("--schema", default="features.schema.json")
    ap.add_argument("--tier", default=DEFAULT_TIER, help="comma-separated: fast,slow,hardware")
    ap.add_argument(
        "--strict-git",
        action="store_true",
        help="treat unresolved implemented_in refs as errors (needs full history)",
    )
    ap.add_argument("--check", help="run a single feature id regardless of tier, then exit")
    args = ap.parse_args()
    feats = load(args.features)
    tiers = {t.strip() for t in args.tier.split(",") if t.strip()}

    if args.check:
        f = next((x for x in feats if x["id"] == args.check), None)
        if not f:
            print(f"unknown feature {args.check}")
            return 1
        err = run_validation(f)
        print(err or f"{args.check}: OK")
        return 1 if err else 0

    errs = check_schema(feats, args.schema) + check_dag(feats)
    ran = skipped = 0
    for f in feats:
        if f["status"] != "done":
            continue
        if not git_rev_ok(f.get("implemented_in")):
            msg = f"{f['id']}: implemented_in '{f.get('implemented_in')}' is not a resolvable git ref"
            if args.strict_git:
                errs.append(msg)
            else:
                print(f"[warn] {msg}")
        if f.get("tier", DEFAULT_TIER) in tiers:
            err = run_validation(f)
            ran += 1
            if err:
                errs.append(err)
        else:
            skipped += 1

    if errs:
        print("VALIDATION FAILED:\n  - " + "\n  - ".join(errs))
        return 1
    done = sum(1 for f in feats if f["status"] == "done")
    print(f"OK: {done} done; ran {ran} for tier(s) {sorted(tiers)}, skipped {skipped} (other tiers).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
