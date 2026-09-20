#!/usr/bin/env bash
# F-053 — Per-directory CLAUDE.md surfaces + package import map (WS-8d).
#
# Always-on pytest only. No skill was authored for this change, so SKILLS.md
# is untouched. Does not implement F-052.
set -euo pipefail

cd "$(dirname "$0")/../.."
bash scripts/validations/_run_always_on_pytest.sh F-053 \
  tests/regression/test_f053_aqa.py \
  tests/regression/test_f053_package_map_aqa.py \
  tests/regression/test_f053_backwards_compat.py
