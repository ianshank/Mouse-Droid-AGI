#!/usr/bin/env bash
# F-030 -- Documentation reconciliation: no live surface states a checkable
# falsehood about the CI job count, coverage floors, orchestrator symbols, or
# the SKILLS.md skill/agent index. NOT covered here: the growth-pillar wiring
# claim -- a sentence-scoped sweep for that specific claim was attempted and
# abandoned as too fragile (false positives on legitimate text); see this
# feature's openspec bundle tasks.md "Explicitly deferred" section. A doc
# claiming a check this script does not actually run is the exact defect
# class this comment itself once had -- keep this list matching the pytest
# targets below, not the other way around.
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

echo "F-030 OK: CI job count, coverage floors, orchestrator symbols, and the skills/agents index all match the tree"
