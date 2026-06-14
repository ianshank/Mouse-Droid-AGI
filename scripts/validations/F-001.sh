#!/usr/bin/env bash
# F-001 — Harness initialized: core files present and parseable.
#
# Routed through a script (not `validate.py --check F-001`) on purpose: a
# validation_command that re-invokes `--check F-001` would recurse forever,
# since `--check` runs the feature's own validation_command. This script is the
# non-recursive ground truth for "the harness exists and parses".
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

# Explicit `if ! ...; then ... exit 1` rather than `assert` — the Jetson Docker
# entrypoint sets PYTHONOPTIMIZE=1, which strips Python asserts (CLAUDE.md).
for f in HARNESS_SPEC.md features.yaml features.schema.json \
         scripts/validate.py scripts/select_next.py; do
  if [ ! -f "$f" ]; then
    echo "F-001 FAIL: missing core harness file: $f" >&2
    exit 1
  fi
done

if ! python -c "import yaml, sys; yaml.safe_load(open('features.yaml'))"; then
  echo "F-001 FAIL: features.yaml does not parse as YAML" >&2
  exit 1
fi

if ! python -c "import json, sys; json.load(open('features.schema.json'))"; then
  echo "F-001 FAIL: features.schema.json does not parse as JSON" >&2
  exit 1
fi

echo "F-001 OK: harness core files present and parseable"
