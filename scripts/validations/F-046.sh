#!/usr/bin/env bash
# F-046 — RSSM gates accept isaac_lab; lidar_dim from observation; mock still skips.
set -euo pipefail

cd "$(dirname "$0")/../.."
bash scripts/validations/_run_always_on_pytest.sh F-046 \
  tests/regression/test_f046_aqa.py \
  tests/regression/test_f046_backwards_compat.py \
  tests/unit/factory/test_factory_rssm_trainable.py \
  tests/unit/training/test_orchestrator_train_rssm.py \
  tests/e2e/test_rssm_pretrain_backend_gate.py
