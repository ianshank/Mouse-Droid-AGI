#!/usr/bin/env bash
# F-044 — fake-injectable Isaac sensor readers + training seams.
set -euo pipefail

cd "$(dirname "$0")/../.."
bash scripts/validations/_run_always_on_pytest.sh F-044 \
  tests/regression/test_f044_aqa.py \
  tests/regression/test_f044_backwards_compat.py \
  tests/unit/sim/test_rover_env_isaaclab.py \
  tests/unit/sim/test_kinematics.py \
  tests/unit/sim/test_isaaclab_sensors.py \
  tests/property/test_rover_kinematics_property.py
