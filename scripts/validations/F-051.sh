#!/usr/bin/env bash
# F-051 — PC-to-rover delivery hardening + trust-boundary pin.
#
# Fixture-based: no rover, no SSH, no Docker. The on-rover promotion and
# rollback drill are operator steps recorded as a local-only evidence chain.
set -euo pipefail

cd "$(dirname "$0")/../.."
bash scripts/validations/_run_always_on_pytest.sh F-051 \
  tests/regression/test_f051_aqa.py \
  tests/regression/test_f051_backwards_compat.py
