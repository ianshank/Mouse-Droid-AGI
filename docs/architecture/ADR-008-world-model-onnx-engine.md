# ADR-008 — World-Model ONNX Engine (Tier B2)

**Status:** Accepted
**Date:** 2026-05-16
**Sprint:** Tier B Track B2 (working sprint plan; not retained in the repo)

> **Note on numbering:** the original Tier B sprint plan referred to this
> as "ADR-007". That number was already assigned to the Hailo-8 accelerator
> ADR. This document is filed as ADR-008; the Isaac Lab Phase B ADR
> (originally planned as ADR-008) will become ADR-009 when B3 lands.

---

## Context

The `DualStreamRSSM` world model (GRU + CfC dual streams) trains in
PyTorch and runs production inference as a `torch.nn.Module`. On the
Orin Nano, the PyTorch interpreter overhead — Python dispatch, autograd
tape allocation, ATen kernel-launch cost — pushes `observe_step` outside
the 30 Hz orchestrator tick budget when the world model is co-resident
with the VLM, the MCTS planner, and the safety monitor on the same
device.

The VLA hot path solved an identical problem with the
`distilled_onnx` backend ([`vla/policy.py`](../../src/mousedroid/vla/policy.py)):
export to ONNX, load via `onnxruntime.InferenceSession` with the
TensorRT execution provider, and benefit from fused kernels +
constant-folded subgraphs that the PyTorch interpreter cannot deliver.
Tier B Track B2 generalises that pattern to the world-model
`observe_step`.

## Decision

Adopt **ONNX-via-ORT** (not direct `torch2trt`) as the world-model
inference engine for the rover hot path.

### Public surface

A single Pydantic field flips the engine:

```yaml
world_model:
  engine: onnx_trt           # default: "torch"
  onnx_path: weights/dual_stream_rssm/observe_step.onnx
  onnx_repo_id: ianshank/mousedroid-dual-stream-rssm
  onnx_filename: observe_step.onnx
  onnx_warmup_iterations: 1
```

`cfg.world_model.engine = "torch"` is the **default** — every existing
YAML loads unchanged and continues to instantiate the PyTorch
`DualStreamRSSM`. Operators opt into the ONNX path explicitly.

### Provider fallback chain

Mirrors `DistilledVLAOnnx`:

```
TensorrtExecutionProvider → CUDAExecutionProvider → CPUExecutionProvider
```

`onnxruntime.get_available_providers()` is intersected with the
requested chain; the first available provider wins. Hosts without
TensorRT (dev workstations, GH Actions runners) silently degrade to CPU.

### What `engine="onnx_trt"` accelerates

`observe_step` only. MCTS planning continues to use the PyTorch
`DualStreamRSSM` because:

1. `imagine_step` requires running the CfC step-by-step with
   per-rollout state, which the export's single-step graph cannot
   reproduce one-step-at-a-time without externalised state buffers.
2. The MCTS critical path is dominated by the policy and value head
   evaluations, not the world model — moving it to ONNX would be a
   smaller win for a much larger surface.

The factory keeps both models around when `engine="onnx_trt"`:
`build_world_model(cfg)` returns the ONNX runtime;
`build_planner(cfg, ...)` (when wired) keeps a separate PyTorch
reference for rollouts.

## Performance contract

- **Production target (Orin Nano, TensorRT EP):** `observe_step`
  mean latency under 10ms.
- **Portable dev gate (CPU EP):** mean under 33ms (30Hz tick).
- **Test:** [`tests/performance/test_observe_step_budget.py`](../../tests/performance/test_observe_step_budget.py),
  budget env-tunable via `MOUSEDROID_OBSERVE_STEP_BUDGET_MS`.

## Cross-engine equivalence guarantee

The torch ↔ ONNX numerical-equivalence test in
[`tests/unit/training/test_export_dual_stream_rssm_onnx.py`](../../tests/unit/training/test_export_dual_stream_rssm_onnx.py)
asserts `np.allclose(torch_out, onnx_out, atol=1e-4)` on the
deterministic outputs (`new_h`, `obs_embed`, `surprise`).

`new_z` (the posterior Gaussian sample) is **intentionally excluded** from
the strict-equality check — its `torch.randn_like` source diverges
between PyTorch and ONNX RNG paths. The underlying distribution
(`post_mean`, `post_logvar`) is identical, which is what justifies
swapping engines at runtime without retraining: downstream consumers
that depend on the distribution (the planner, the critic) see the same
signal; consumers that depend on the specific sample (none, in the
current architecture) see a different sample per engine.

## Migration path

1. Export an `.onnx` from a trained checkpoint:
   ```bash
   python scripts/export_dual_stream_rssm_onnx.py \
       --checkpoint weights/dual_stream_rssm/final.pt \
       --config config/jetson_production.yaml \
       --output weights/dual_stream_rssm/observe_step.onnx \
       --opset 17 \
       --push-to-hf            # optional — uploads to HF Hub
   ```
2. Edit `config/jetson_production.yaml`:
   ```yaml
   world_model:
     engine: onnx_trt
     onnx_path: /opt/mousedroid/weights/dual_stream_rssm/observe_step.onnx
   ```
   Or rely on the HF Hub fallback by leaving `onnx_path: null` — the
   factory will auto-download from `cfg.world_model.onnx_repo_id`.
3. Restart `mousedroid`. The `world_model_engine_selected` structured
   log event confirms the active engine.

## What was de-risked first

Tier B2 Story 0 ran a [throwaway spike](../../tools/spikes/cfc_onnx_spike.py)
that exported the bare `CfCWrapper` via `torch.onnx.export` before
committing to the full sprint. Outcome: ✅ SUCCESS — export 0.45s,
numerical equivalence 4.47e-08 (~5 orders of magnitude under tolerance),
deterministic across 5 ORT runs. Full record in
[`tools/spikes/CFC_ONNX_SPIKE_REPORT.md`](../../tools/spikes/CFC_ONNX_SPIKE_REPORT.md).

The spike found one obstacle that influenced the production design:
`torch.onnx.export` cannot trace `CfCWrapper.forward`'s keyword-only `dt`
parameter directly. Resolved with a thin
[`_ObserveStepExportShim(nn.Module)`](../../scripts/export_dual_stream_rssm_onnx.py)
that narrows the API to positional tensor args for tracing only — full
runtime functionality (including `dt`-driven continuous-time stepping)
remains in `CfCWrapper`.

## Consequences

### Positive

- **`observe_step` becomes one config flip away from <10ms** on Orin Nano.
- **Single source of truth for observation packing**
  ([`observation_packer.pack_observation`](../../src/mousedroid/world_model/observation_packer.py))
  shared by both engines — no drift on dtype, empty-buffer handling, or
  disabled-modality semantics.
- **HF Hub fallback** means deployments without a baked-in `.onnx` can
  auto-download on first boot.
- **Mirrors the proven VLA pattern** — same lazy import, same provider
  fallback, same `torch.no_grad()` boundary.

### Negative / out of scope

- **`imagine_step` not accelerated.** MCTS planning still pays the
  PyTorch interpreter cost. If profiling on Orin Nano shows MCTS hot
  enough to matter, a future ADR can wire a second `.onnx` graph for
  one-step imagined rollouts.
- **Direct `torch2trt` path remains a future option** via the existing
  [`JetsonTensorRTCompiler`](../../src/mousedroid/efficiency/tensorrt.py)
  if ORT overhead proves prohibitive in production profiling.
- **`new_z` divergence between engines.** Documented above; no
  downstream consumer depends on the specific sample today.
- **Operator must remember to re-export after re-training.** The
  factory does NOT auto-detect a mismatched `.onnx` against the
  trained PyTorch weights — wrong weights = wrong inference, silently.
  Operators are expected to re-run the export script after every
  significant retraining and update `onnx_path` (or push to HF).

## References

- Tier B sprint plan — a working planning doc that was not retained in the repo (historical reference).
  (Track B2, Stories 0–7)
- CfC spike report: [`tools/spikes/CFC_ONNX_SPIKE_REPORT.md`](../../tools/spikes/CFC_ONNX_SPIKE_REPORT.md)
- VLA precedent: [`src/mousedroid/vla/policy.py`](../../src/mousedroid/vla/policy.py)
  (DistilledVLAOnnx class)
- Export tooling: [`scripts/export_dual_stream_rssm_onnx.py`](../../scripts/export_dual_stream_rssm_onnx.py)
- Runtime class: [`src/mousedroid/world_model/dual_stream_rssm_onnx.py`](../../src/mousedroid/world_model/dual_stream_rssm_onnx.py)
- Factory dispatch: `build_world_model` in
  [`src/mousedroid/factory/world_model.py`](../../src/mousedroid/factory/world_model.py)
- Performance budget: [`tests/performance/test_observe_step_budget.py`](../../tests/performance/test_observe_step_budget.py)
