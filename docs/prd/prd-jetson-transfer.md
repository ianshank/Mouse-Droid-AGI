# Product Requirements Document (PRD)

## Epic 4: E2E Training & Jetson Transfer

**Status**: Draft
**Date**: 2026-03-14

---

### User Story

**As an** AI/Robotics Engineer,
**I want** to execute the complete end-to-end training pipeline locally (or on GPU server) and transfer the validated weights to the Jetson Orin Nano,
**So that** I can verify that all recent bug fixes (BDI accuracy, MCTS latency) produce a stable 30Hz target inference loop on the physical hardware.

### Acceptance Criteria

1. **Pipeline Execution**: 
   - `run_pipeline.py` successfully completes Phase 0 (Data) through Phase 4 (RL) without crashing.
   - The `--validate` flag automatically generates `training_report.json` at the end.

2. **Validation Gates**:
   - BDI accuracy is ≥ 60% (as enforced by `check_report.py`).
   - MCTS p50 latency is ≤ 50 ms.
   - All tests in the test suite pass.

3. **Packaging & Transfer**:
   - Weights in `weights/` directory are packaged correctly (e.g. `upload_weights.py` to HuggingFace or `.sh` script push).
   - Jetson inference service (`app.py` or equivalent) can successfully load the transferred weights without dimension or architecture errors.

4. **Smoke Test**:
   - The Jetson inference loop (using the loaded weights) runs for at least 1 minute without crashing.

### Out of Scope

- Hardware sensor calibration (Microphone, Camera). This epic assumes mock sensors or pre-calibrated sensors exist on the Jetson.
- Modifying the RSSM core architecture.
- Optimizing TensorRT pipelines (saving for Epic 5 if needed).

### Success Metrics

- `training_report.json` shows all checks passed.
- Weights deploy successfully with zero layer-shape mismatch errors on hardware startup.
- Inference loop on hardware achieves ≤ 50ms latency (20Hz+ stable).
