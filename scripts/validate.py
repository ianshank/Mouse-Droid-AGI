#!/usr/bin/env python3
"""Validate features.yaml structure, DAG integrity, git provenance, and that
every `done` feature's validation_command passes for the selected tier(s).

    python scripts/validate.py --tier fast
    python scripts/validate.py --tier fast,slow
    python scripts/validate.py --tier fast,slow,hardware --strict-git

Thin CLI shim over :mod:`mousedroid.harness.spec` (ADR-012). All logic lives in
the importable, unit-tested package module; this file only parses args and
renders output, mirroring scripts/validate_configs.py.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

# Resolve the repo root so the script is invocable from any CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mousedroid.harness import spec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(_REPO_ROOT / "features.yaml"))
    ap.add_argument("--schema", default=str(_REPO_ROOT / "features.schema.json"))
    ap.add_argument("--tier", default=spec.DEFAULT_TIER, help="comma-separated: fast,slow,hardware")
    ap.add_argument(
        "--strict-git",
        action="store_true",
        help="treat unresolved implemented_in refs as errors (needs full history)",
    )
    ap.add_argument("--check", help="run a single feature id regardless of tier, then exit")
    args = ap.parse_args()

    feats = spec.load_features(args.features)

    # Commands and git provenance resolve repo-relative paths regardless of CWD.
    run = functools.partial(spec.run_validation, cwd=_REPO_ROOT)
    rev = functools.partial(spec.git_rev_ok, cwd=_REPO_ROOT)

    if args.check:
        f = next((x for x in feats if x["id"] == args.check), None)
        if not f:
            print(f"unknown feature {args.check}")
            return 1
        err = run(f)
        print(err or f"{args.check}: OK")
        return 1 if err else 0

    tiers = {t.strip() for t in args.tier.split(",") if t.strip()}
    invalid = sorted(tiers - spec.VALID_TIERS)
    if invalid:
        # Fail loudly: a typo like `--tier fasst` must not silently match no
        # features and exit 0 (which would let a broken build pass in CI).
        print(f"invalid tier(s) {invalid}; valid tiers are {sorted(spec.VALID_TIERS)}")
        return 1

    result = spec.run_features(
        feats, args.schema, tiers, strict_git=args.strict_git, runner=run, rev_checker=rev
    )
    for w in result.warnings:
        print(f"[warn] {w}")

    if result.errors:
        print("VALIDATION FAILED:\n  - " + "\n  - ".join(result.errors))
        return 1
    print(
        f"OK: {result.done} done; ran {result.ran} for tier(s) {sorted(tiers)}, "
        f"skipped {result.skipped} (other tiers)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
