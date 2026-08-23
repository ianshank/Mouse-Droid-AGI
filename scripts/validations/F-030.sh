#!/usr/bin/env bash
# F-030 -- Documentation reconciliation: no live surface states a checkable
# falsehood about the CI job count, coverage floors, orchestrator symbols, the
# SKILLS.md skill/agent index, or the growth-pillar wiring claim.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of caller CWD

PY_BIN="${MOUSEDROID_PYTHON:-python}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN="python3"
fi

# Explicit `if ! ...` rather than `assert` -- the Jetson Docker entrypoint sets
# PYTHONOPTIMIZE=1, which strips Python asserts (CLAUDE.md).
if ! "$PY_BIN" -m pytest \
      tests/regression/test_doc_reconciliation_aqa.py \
      tests/regression/test_claude_workforce_aqa.py::test_every_skill_directory_is_mentioned_in_the_index \
      tests/regression/test_claude_workforce_aqa.py::test_every_agent_is_listed_in_the_subagent_skills_table \
      --import-mode=importlib --no-cov -q; then
  echo "F-030 FAIL: a live doc surface disagrees with the tree it describes" >&2
  exit 1
fi

echo "F-030 OK: CI job count, coverage floors, and the skills/agents index all match the tree"
