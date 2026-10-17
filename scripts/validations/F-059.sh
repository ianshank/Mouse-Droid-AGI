#!/usr/bin/env bash
# F-059 — rover gate closeout. Fixture-based: every pin runs in hosted CI.
set -euo pipefail
cd "$(dirname "$0")/../.."
bash scripts/validations/_run_always_on_pytest.sh F-059 \
  tests/regression/test_f059_aqa.py \
  tests/regression/test_f059_backwards_compat.py
