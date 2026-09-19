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
YAML loads unchanged. Those overlays instantiate the plain `RSSM`, **not**
`DualStreamRSSM`: `ModelConfig.cfc_hidden_dim` defaults to `0`, so
`build_world_model` falls through past the dual-stream branch
([`src/mousedroid/factory/world_model.py`](../../src/mousedroid/factory/world_model.py)).
Operators opt into the ONNX path explicitly.

`engine: onnx_trt` additionally **requires `model.cfc_hidden_dim > 0`** —
`_build_onnx_world_model` raises `ValueError` at boot otherwise, because the
export is built from `DualStreamRSSM`. Only
[`config/jetson_dual_stream.yaml`](../../config/jetson_dual_stream.yaml) sets
`cfc_hidden_dim: 64`, and it self-gates that behind human review ("Only enable
(64+) after human review of training metrics").
[`config/jetson_production.yaml`](../../config/jetson_production.yaml) carries
no `model:` block at all, so the ONNX engine cannot be enabled from the
production overlay as shipped.

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
`build_world_model(cfg)` returns a
[`CompositeWorldModel`](../../src/mousedroid/world_model/composite.py) that
serves `observe_step` from the ONNX runtime and `imagine_step` from a PyTorch
`DualStreamRSSM`. The agent that consumes it is built by `build_agent` in
[`src/mousedroid/factory/cognitive.py`](../../src/mousedroid/factory/cognitive.py);
there is no `build_planner` function in the tree.

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
(`post_mean`, `post_logvar`) is identical.

The exclusion is correct for a **single-step** parity gate: an unseeded
sampler makes `new_z` non-reproducible even between two runs of the *same*
engine, so asserting equality on it would pin RNG implementation rather than
model behaviour.

It is **not** justified by an absence of consumers. `new_z` has three, all in
production today:

1. **`observe_step` itself, on the next tick.**
   [`src/mousedroid/orchestrator/_world_model_state_mixin.py`](../../src/mousedroid/orchestrator/_world_model_state_mixin.py)
   assigns `self._h, self._z` and feeds `self._z` straight back into the next
   `observe_step` call;
   [`src/mousedroid/world_model/dual_stream_rssm.py`](../../src/mousedroid/world_model/dual_stream_rssm.py)
   builds `recurrent_input = torch.cat([z, prev_action], dim=-1)` for **both**
   the GRU and the CfC stream. The sample is recurrent state, not a leaf.
2. **`MCTSPlanner.plan(h, z)`** — the planner is handed the sample, not the
   distribution ([`src/mousedroid/agents/navigation.py`](../../src/mousedroid/agents/navigation.py)).
3. **The VLA policy** — `VLAObservation(h=..., z=...)` in
   [`src/mousedroid/orchestrator/_action_mixin.py`](../../src/mousedroid/orchestrator/_action_mixin.py).

**Consequence:** because `new_z` is fed back into the recurrence, per-engine
sample divergence compounds across ticks. **Multi-step parity between the two
engines is therefore impossible** until the sampler's noise becomes an injected
graph input (an explicit `eps` tensor) rather than an internal
`torch.randn_like`. Until then, cross-engine agreement is only ever a
single-step, distribution-level claim, and any multi-step A/B must be read as
two different trajectories from the same distribution.

## Migration path

1. Export an `.onnx` from a trained checkpoint. The `--config` overlay must
   set `model.cfc_hidden_dim > 0`, so use `config/jetson_dual_stream.yaml` —
   `config/jetson_production.yaml` has no `model:` block and exports nothing:
   ```bash
   python scripts/export_dual_stream_rssm_onnx.py \
       --checkpoint weights/dual_stream_rssm/final.pt \
       --config config/jetson_dual_stream.yaml \
       --output weights/dual_stream_rssm/observe_step.onnx \
       --opset 17 \
       --push-to-hf            # optional — uploads to HF Hub
   ```
2. Edit `config/jetson_dual_stream.yaml` — **not**
   `config/jetson_production.yaml`, which cannot enable this engine
   (`_build_onnx_world_model` raises `ValueError` when `cfc_hidden_dim <= 0`,
   and the production overlay leaves it at the schema default `0`) — for the
   `model:` half of the switch, and that half is already in it
   (`cfc_hidden_dim: 64`). The `world_model:` half does **not** go into tracked
   YAML at all. Select the engine with environment variables in
   `/etc/mousedroid/docker.env`, which is outside the sync target and is never
   transferred, so it survives a promotion:
   ```bash
   MOUSEDROID_WORLD_MODEL__ENGINE=onnx_trt
   MOUSEDROID_WORLD_MODEL__ONNX_PATH=/opt/mousedroid/weights/dual_stream_rssm/observe_step.onnx
   ```
   A `world_model:` block in a tracked overlay is not merely discouraged, it
   silently does nothing: at the schema SHA that `config-compat` pins
   (`deployments/jetson-image.json`), `WorldModelConfig` is a plain `BaseModel`
   with `extra="ignore"`, so `scripts/check_config_compat.py` passes the key and
   the pinned schema then drops it on load. The gate goes green, nothing logs a
   complaint, and the rover stays on the `torch` engine — worse than a hard
   failure, which is what a new *top-level* block would have given you.
   `docs/runbooks/pc-to-jetson-promotion.md` states the same rule for every new
   runtime field.

   `onnx_path` is **required in practice**, which is why
   `MOUSEDROID_WORLD_MODEL__ONNX_PATH` is in the set above. Leaving it unset
   engages the HF Hub fallback, but the default `onnx_repo_id`
   (`ianshank/mousedroid-dual-stream-rssm`) currently publishes **no `.onnx`
   artifact** — only `.gitattributes` and `README.md` — so the fallback
   resolves nothing and boot fails. Point it at a file you exported
   in step 1, or push that artifact to the Hub first.
3. Restart `mousedroid`. The `world_model_engine_selected` structured
   log event confirms the active engine.

> **Prerequisite, not a footnote:** step 2 also means the CfC stream is on.
> `config/jetson_dual_stream.yaml` self-gates that ("Only enable (64+) after
> human review of training metrics"), so this migration is blocked on that
> review, not just on having an artifact.

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

- **`observe_step` gets an ONNX/TensorRT path** targeting <10ms on Orin Nano.
  Not "one config flip": it needs `model.cfc_hidden_dim > 0` (human-review
  gated), an exported `.onnx`, and an `onnx_path` pointing at it.
- **Single source of truth for observation packing**
  ([`observation_packer.pack_observation`](../../src/mousedroid/world_model/observation_packer.py))
  shared by both engines — no drift on dtype, empty-buffer handling, or
  disabled-modality semantics.
- **HF Hub fallback** is wired in the factory, so deployments without a
  baked-in `.onnx` *could* auto-download on first boot. Not usable today:
  `ianshank/mousedroid-dual-stream-rssm` holds only `.gitattributes` and
  `README.md`. Operators must supply `onnx_path` until an artifact is pushed.
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
- **`new_z` divergence between engines.** Documented above. Three production
  consumers depend on the specific sample (the recurrence itself, the MCTS
  planner, the VLA policy), so the two engines produce diverging trajectories
  and **multi-step parity is impossible** until the sampler's noise is an
  injected input.
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
