#!/usr/bin/env bash
# F-043 — nested Isaac schema defaults, factory docstring, always-on AQA.
set -euo pipefail

cd "$(dirname "$0")/../.."
bash scripts/validations/_run_always_on_pytest.sh F-043 \
  tests/regression/test_f043_aqa.py \
  tests/regression/test_f043_backwards_compat.py \
  tests/unit/sim/test_rover_config.py \
  tests/unit/sim/test_isaaclab_scene.py
