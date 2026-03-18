# MouseDroid Deployment Notes
**Date**: 2026-03-16  
**Repository**: https://huggingface.co/ianshank/mousedroid-weights  
**Status**: ✅ Retrained, uploaded, and deployed on Jetson Orin Nano with Docker-first validation

## Current Validation Snapshot
Latest validated branch state:

| Phase | Status | Details |
|-------|--------|---------|
| Weight Files | ✅ PASS | 8/8 required artefacts present |
| RSSM Shapes | ✅ PASS | 513,185 parameters |
| BDI Accuracy | ✅ PASS | 75.87% holdout accuracy, 8 classes |
| Constitutional RL | ✅ PASS | Policy + value networks verified |
| MCTS Latency | ✅ PASS | p50=16ms, p95=16ms, mean reward=0.4844 |
| Jetson Docker Smoke | ✅ PASS | container, CUDA, weights, LLM, ultrasonic, microphone, and logs validated |

## What Changed Since The Earlier Deployment Draft
1. BDI validation and training alignment bugs were fixed.
2. BDI retraining now clears the 60% threshold with margin.
3. The uploaded Hugging Face weights match the post-refactor retrain branch.
4. Jetson deployment now uses the baked-audio Docker image path and the consolidated smoke test.

## Runtime Verification Workflow

### 1. Deploy Weights + Application
```bash
./scripts/deploy_remote.sh --weights --with-llm
# or on the Jetson host
sudo bash scripts/docker_deploy.sh
```

### 2. Run Post-Deploy Smoke Validation
```bash
./scripts/jetson_smoke_test_deploy.sh
```

Optional flags when a device is intentionally disconnected:
```bash
MOUSEDROID_CHECK_MIC=false ./scripts/jetson_smoke_test_deploy.sh
MOUSEDROID_CHECK_ULTRASONIC=false ./scripts/jetson_smoke_test_deploy.sh
```

### 3. Rebuild The Training Report After A New Retrain
```bash
python -m training.validate_weights \
  --weights-dir weights \
  --annotations training/data/bdi_annotations.npz \
  --report
```

The current validator now records holdout diagnostics beyond aggregate accuracy, including balanced accuracy, majority-class baseline, per-class precision/recall, and prediction distributions.

## Known Operational Constraints

### 1. Partial-Hardware Validation Still Pending
- Camera and ESP32 full-path validation remain the main remaining real-hardware close-out items.
- The orchestrator is already hardened to degrade gracefully when those devices are absent.

### 2. Ultrasonic Pinmux Can Still Require Manual Bring-Up
- Some Jetson boots still need the manual pinmux fix before direct GPIO reads succeed:
```bash
sudo busybox devmem 0x243D020 w 0x5
```

### 3. Microphone Validation Is Log-Assisted
- The standalone mic probe can fail if the main app already owns the device.
- Treat `usb_microphone_started` plus absence of `audio_capture_failed` in recent logs as the authoritative healthy signal.

### 4. LLM Functional Validation Is In Place, Latency Characterisation Is Not Final
- GGUF loading and prompt smoke tests pass in-container.
- Broader mission-to-goal latency benchmarking on Jetson remains an open follow-up item.

## Training Outcome Summary
- Phase 0: synthetic sequence generation complete
- Phase 1: RSSM converged and validated
- Phase 2: MCTS warm-start tuned to `best_ucb_c=1.41`
- Phase 3: BDI weights trained with normalization fixes and passing holdout accuracy
- Phase 4: Constitutional RL policy and value weights trained and validated
- Upload: artefacts pushed to `ianshank/mousedroid-weights`

## Remaining Close-Out Tasks
1. Run camera integration validation on the Jetson production path.
2. Run ESP32 loopback / motion validation on the Jetson production path.
3. Regenerate and commit `training/results/training_report.json` from the latest validator when running in a clean Python environment.

## References
- Training pipeline: [training/run_pipeline.py](training/run_pipeline.py)
- Validation: [training/validate_weights.py](training/validate_weights.py)
- Jetson smoke test: [scripts/jetson_smoke_test_deploy.sh](scripts/jetson_smoke_test_deploy.sh)
- Weights + LLM probe: [scripts/_test_weights_llm.py](scripts/_test_weights_llm.py)
- HuggingFace repo: https://huggingface.co/ianshank/mousedroid-weights
