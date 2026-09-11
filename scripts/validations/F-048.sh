#!/usr/bin/env bash
# F-048 — docs addenda; no Prometheus Isaac family; harness stays None.
set -euo pipefail

cd "$(dirname "$0")/../.."
bash scripts/validations/_run_always_on_pytest.sh F-048 \
  tests/regression/test_f048_aqa.py \
  tests/regression/test_f048_backwards_compat.py
