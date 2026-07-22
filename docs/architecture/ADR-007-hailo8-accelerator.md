# ADR-007: Hailo-8 Neural Accelerator for Perception Offload

## Status

Accepted

## Context

MouseDroid runs a 30 Hz sense-plan-act loop on a Jetson Orin Nano where
a single GPU time-slices 7+ concurrent inference workloads: YOLO detection,
feature extraction, RSSM world modeling, MCTS rollouts, SAC policy inference,
LLM replanning, and arm 6-DoF pose estimation.

The system already employs aggressive compute mitigations — FP16 precision,
TensorRT compilation, adaptive MCTS budgets, MoE sparsity, VRAM-aware batch
tuning, and thermal throttle pauses at 85C. These indicate the GPU is at the
edge of its compute envelope. Adding any new model (e.g. FoundationPose for
6-DoF) or increasing any frequency (perception beyond 30 Hz, MCTS beyond
10 Hz) would blow the remaining budget.

The workloads naturally split along a perception/reasoning boundary:

| Property | Perception | Reasoning |
|---|---|---|
| Computation graph | Static, compiled | Dynamic, gradient-sensitive |
| Precision need | INT8 sufficient | FP16/FP32 needed |
| Latency profile | Deterministic, low | Variable, bursty |
| Models | YOLO, feature extractor | RSSM, SAC, MCTS, LLM |

## Decision

Integrate a Hailo-8 M.2 neural accelerator (26 TOPS INT8, ~$80, ~3W) to
offload perception workloads to dedicated silicon. The Jetson GPU is freed
exclusively for reasoning.

### Architecture

- New `HailoRuntime` wraps `hailo_platform` with async dispatch and PCIe
  serialization (asyncio.Lock prevents bus contention with NVMe SSD)
- `HailoFeatureExtractor` implements existing `FeatureExtractorProtocol`
- `HailoYOLODetector` implements the same detection interface as `ObjectDetector`
- Factory wiring via `build_hailo_runtime()` with singleton shared runtime
- Graceful degradation: every Hailo class falls back to the existing GPU pipeline
- Configuration: `Optional[HailoConfig]` with `None` default on `Settings`

### Key Design Choices

1. **Shared runtime singleton** — both YOLO and feature extraction share one
   `HailoRuntime` instance. The Hailo-8 manages multiple HEF models internally.
2. **asyncio.Lock for PCIe serialization** — prevents bandwidth contention
   between Hailo inference and NVMe I/O on the shared PCIe root complex.
3. **INT8 on Hailo / FP16 on GPU** — quantization happens offline during HEF
   compilation on x86, not at runtime on the Jetson.
4. **Protocol-based DI** — zero regression path. All Hailo types implement
   existing protocols and are wired via `factory.py`.

## Consequences

### Positive

- Perception and reasoning run in true parallel on separate silicon
- Perception frequency can increase to 60+ Hz (Hailo handles YOLOv8n at 60+ FPS)
- MCTS planning can increase from 10 Hz toward 30 Hz
- 6-DoF pose estimation (PVNet/FoundationPose) becomes feasible on the freed GPU
- LLM 500ms latency target is structurally achievable
- GPU thermal load reduced (~30-40% inference offloaded), enabling larger batch sizes
- ~$80 cost vs ~$700+ for Jetson AGX Orin upgrade

### Negative

- PCIe bandwidth is shared with NVMe SSD — monitor with `hailortcli monitor` + `iotop`
- HEF model compilation requires x86 workstation (offline step)
- INT8 quantization may reduce detection accuracy vs FP16 — requires calibration validation
- New optional dependency (`hailort>=4.18`) and hardware requirement

### Risks

- Orin Nano M.2 slot availability (may need USB adapter if both slots occupied)
- Hailo SDK version compatibility with JetPack 6.x
- PCIe Gen3 x1 bandwidth ceiling at sustained high throughput

## References

- Hailo-8 datasheet: 26 TOPS INT8, 2.5W typical power
- Existing DLA placeholder: `JetsonConfig.dla_enabled` (disabled, insufficient throughput)
- Related: ADR-005 GPU pre-training pipeline (thermal/VRAM management)
