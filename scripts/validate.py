#!/usr/bin/env python3
"""Validate features.yaml structure, DAG integrity, git provenance, and that
every `done` feature's validation_command passes for the selected tier(s).

    python scripts/validate.py --tier fast
    python scripts/validate.py --tier fast,slow
    python scripts/validate.py --tier fast,slow,hardware --strict-git
"""
import argparse, json, subprocess, sys, yaml

DEFAULT_TIER = "fast"

def load(path): return yaml.safe_load(open(path))["features"]

def check_schema(feats, schema_path):
    try:
        import jsonschema
    except ImportError:
        print("[warn] jsonschema not installed; skipping structural check")
        return []
    schema = json.load(open(schema_path))
    return [f"schema: {list(e.path)}: {e.message}"
            for e in jsonschema.Draft202012Validator(schema).iter_errors({"features": feats})]

def check_dag(feats):
    by_id = {f["id"]: f for f in feats}
    errs = []
    for f in feats:
        for d in f.get("depends_on", []):
            if d not in by_id:
                errs.append(f"dag: {f['id']} depends_on unknown id {d}")
    WHITE, GREY, BLACK = 0, 1, 2
    color = {f["id"]: WHITE for f in feats}
    def visit(n, stack):
        color[n] = GREY
        for d in by_id[n].get("depends_on", []):
            if d not in by_id: continue
            if color[d] == GREY:
                errs.append("dag: cycle " + " -> ".join(stack + [d]))   # e.g. F-001 -> F-002 -> F-001
            elif color[d] == WHITE:
                visit(d, stack + [d])
        color[n] = BLACK
    for f in feats:
        if color[f["id"]] == WHITE:
            visit(f["id"], [f["id"]])
    return errs

def git_rev_ok(ref):
    if not ref: return False
    r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                       capture_output=True, text=True)
    return r.returncode == 0

def run_validation(f):
    cmd = f.get("validation_command")
    if not cmd:
        return f"{f['id']}: status=done but no validation_command"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        return f"{f['id']}: validation_command failed ({r.returncode})\n      " + "\n      ".join(tail)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.yaml")
    ap.add_argument("--schema", default="features.schema.json")
    ap.add_argument("--tier", default=DEFAULT_TIER, help="comma-separated: fast,slow,hardware")
    ap.add_argument("--strict-git", action="store_true",
                    help="treat unresolved implemented_in refs as errors (needs full history)")
    ap.add_argument("--check", help="run a single feature id regardless of tier, then exit")
    args = ap.parse_args()
    feats = load(args.features)
    tiers = {t.strip() for t in args.tier.split(",") if t.strip()}

    if args.check:
        f = next((x for x in feats if x["id"] == args.check), None)
        if not f:
            print(f"unknown feature {args.check}"); return 1
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
            e = run_validation(f); ran += 1
            if e: errs.append(e)
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
