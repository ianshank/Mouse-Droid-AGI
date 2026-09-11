#!/usr/bin/env bash
# F-045 — Isaac apply_domain_params (CI fakes; Replicator stays behind the method).
set -euo pipefail

cd "$(dirname "$0")/../.."
bash scripts/validations/_run_always_on_pytest.sh F-045 \
  tests/regression/test_f045_aqa.py \
  tests/regression/test_f045_backwards_compat.py \
  tests/unit/sim/isaaclab/test_randomization.py \
  tests/unit/sim/test_rover_env_isaaclab.py \
  tests/integration/test_sim_episode_generator_mock.py
