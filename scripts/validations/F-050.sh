#!/usr/bin/env bash
# F-050 — observe-step instrumentation + ONNX provider proof.
#
# Always-on tests only: the blocking `test` job and harness `validate-fast`
# install .[dev,telemetry,mcp], which carries no onnxruntime, so a script
# pointing at an importorskip-gated module would report passed=0 and fail.
# Real-ORT behaviour lives in the advisory onnx-world-model-extras job.
set -euo pipefail

cd "$(dirname "$0")/../.."
bash scripts/validations/_run_always_on_pytest.sh F-050 \
  tests/regression/test_f050_aqa.py \
  tests/regression/test_f050_backwards_compat.py \
  tests/regression/test_f050_narrative_aqa.py \
  tests/unit/world_model/test_observe_step_timing.py \
  tests/unit/scripts/test_analyze_observe_step_ceiling.py \
  tests/property/test_observe_step_ceiling_properties.py \
  tests/integration/test_world_model_metrics_seam.py
