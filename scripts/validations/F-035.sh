#!/usr/bin/env bash
# F-035 — deploy/feature pin re-tagging (scripts/repin_tags.sh).
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

# The dedicated unit-test file runs in full: it builds a real bare remote per
# test, so it covers the dry-run, --push, idempotence, foreign-tag-coverage,
# SHA-format-rejection and cross-script paths end to end without touching this
# repo's own remote. Scoped to this one file (not the whole
# tests/unit/scripts/ directory) so an unrelated regression in a sibling
# script's tests cannot produce a misleading "F-035 FAIL" diagnosis -- the
# same reason F-031.sh names its node IDs individually.
# Explicit `if ! ...` rather than `assert` — the Jetson Docker entrypoint sets
# PYTHONOPTIMIZE=1, which strips Python asserts (CLAUDE.md).
if ! "$PY_BIN" -m pytest \
      tests/unit/scripts/test_repin_tags.py \
      --import-mode=importlib --no-cov -q; then
  echo "F-035 FAIL: repin_tags.sh does not honour its dry-run/--push/validation contract" >&2
  exit 1
fi

# The script must survive its own --help and reject an unknown flag, since both
# are operator-facing entry points that no pytest case drives through the real
# file on disk.
if ! bash scripts/repin_tags.sh --help >/dev/null 2>&1; then
  echo "F-035 FAIL: scripts/repin_tags.sh --help does not exit cleanly" >&2
  exit 1
fi
if bash scripts/repin_tags.sh --definitely-not-a-flag >/dev/null 2>&1; then
  echo "F-035 FAIL: scripts/repin_tags.sh accepted an unknown argument" >&2
  exit 1
fi

# The deploy pin must be a real 40-char SHA. This is the value repin_tags.sh
# turns into a tag name, and the file's own notes are the reason F-035 exists,
# so a malformed pin here would make the feature's premise unverifiable.
if ! "$PY_BIN" - <<'PYCHECK'
import json
import re
import sys

sha = json.load(open("deployments/jetson-image.json"))["sha"]
if not re.fullmatch(r"[0-9a-f]{40}", sha):
    print(f"deploy pin is not a 40-char lowercase hex SHA: {sha!r}", file=sys.stderr)
    sys.exit(1)
PYCHECK
then
  echo "F-035 FAIL: deployments/jetson-image.json carries a malformed sha" >&2
  exit 1
fi

echo "F-035 OK: repin_tags.sh validates pin format, is idempotent, and frees carrier branches"
