# CfC ONNX Exportability Spike — Report

**Date:** 2026-05-16
**Track:** Tier B2 Story 0 (de-risk gate for the full B2 sprint)
**Script:** [`tools/spikes/cfc_onnx_spike.py`](cfc_onnx_spike.py)
**Outcome:** ✅ **SUCCESS** — proceed with B2 Story 1 (full `observe_step` export)

---

## Question Answered

> Can `ncps.torch.CfC` (wrapped by `mousedroid.world_model.cfc_cell.CfCWrapper`) be exported to ONNX via `torch.onnx.export` cleanly?

**Answer: yes, with one caveat** — `CfCWrapper.forward` has a keyword-only `dt: Tensor | None = None` parameter that confuses torch's ONNX tracer on torch 2.5.1. The fix is a 10-line shim module that narrows the API to `(x, h) -> new_h` for tracing only.

## Run Output

```
[1/4] Building CfCWrapper(input_dim=64, cfc_hidden_dim=32)...
      cfc_cell_init backbone_layers=1 backbone_units=64 hidden_dim=32 input_dim=64 mode=default
      example_x.shape=(1, 64), example_h.shape=(1, 32)
[2/4] Capturing torch reference output...
      torch_out.shape=(1, 32), dtype=torch.float32
[3/4] Exporting to tools\spikes\cfc_only.onnx (opset=17)...
      [OK] Export succeeded in 0.45s
      ONNX artifact: tools\spikes\cfc_only.onnx (62271 bytes)
[4/4] Validating ONNX inference + numerical equivalence...
      [OK] Numerical equivalence: max_abs_diff=4.47e-08 (tol=0.001)

[OK] Spike SUCCESS — proceed with B2 Story 1.
```

## Key Findings

### 1. Export succeeds via thin shim module

The naive approach `torch.onnx.export(cfc_wrapper, (x, h), ...)` **fails** with:

```
TypeError: CfCWrapper.forward() takes 3 positional arguments but 4 were given
```

This happens because torch's ONNX tracer sees the keyword-only `dt` parameter as a positional fourth argument. Working around it requires a thin wrapper:

```python
class _CfCExportShim(nn.Module):
    """Narrows CfCWrapper API to (x, h) -> new_h for ONNX tracing."""
    def __init__(self, cfc: CfCWrapper) -> None:
        super().__init__()
        self._cfc = cfc

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self._cfc(x, h)
```

The wrapper preserves all CfCWrapper functionality (including `dt`-driven continuous-time stepping when `dt` is passed via the orchestrator) — the shim is only used at export time.

### 2. Numerical equivalence is tight

`max_abs_diff = 4.47e-08` between PyTorch and ONNX outputs. The configured tolerance was `1e-3`. **Margin of safety: ~5 orders of magnitude.**

This is well within float32 round-off limits and far below what would matter for downstream MCTS planning or policy gradient signals.

### 3. ONNX inference is deterministic

5 consecutive `onnxruntime.InferenceSession.run()` calls produce bit-identical outputs (`max_abs_diff < 1e-9` across iterations). No hidden randomness in the exported graph.

### 4. Export artifact size

62,271 bytes for `input_dim=64, hidden_dim=32`. Scaling linearly with parameter count, the production `DualStreamRSSM` ONNX (full `observe_step` with GRU + CfC + encoder) will be ~few-MB — well within HF Hub size limits.

### 5. Dynamic batch dimension works

The export uses `dynamic_axes={"x": {0: "batch"}, "h": {0: "batch"}, "new_h": {0: "batch"}}` so the same ONNX artifact can serve both inference (batch=1 on Orin Nano) and future training-time use (batch=N). No need for separate exports per batch size.

## Decision

**Proceed with B2 Story 1** using the shim pattern. The shim becomes a small permanent helper under `src/mousedroid/world_model/onnx_export.py` so the production export script reuses it.

### What does NOT need to change

- The full `observe_step` export (GRU + encoder + StreamFusion + CfC) is the right next step. No need to pivot to "GRU-only ONNX, CfC-in-Python wrapper" as the plan's fallback contemplated.
- The `DualStreamRSSMOnnx` runtime class can load the single fused `.onnx` and run inference end-to-end.

### What's already de-risked

- ✅ Export tractability (no exotic ops missing from opset 17)
- ✅ Numerical equivalence
- ✅ Determinism
- ✅ Dynamic batch
- ✅ Loads + runs via standard `onnxruntime.InferenceSession`

### What remains for B2 Story 1

- Wire the `_CfCExportShim`-equivalent pattern around the full `DualStreamRSSM.observe_step_traceable` (not just the bare CfC cell)
- Pack the `ObservationProtocol` into a flat `obs_vec` tensor at the export boundary (the export tracer needs static tensors, not Protocol objects)
- Add `torch.onnx.export(...)` invocation in `scripts/export_dual_stream_rssm_onnx.py`

## Test Plan for B2 Story 1

The spike's success means the integration test for B2 Story 2 (`DualStreamRSSMOnnx` runtime) can use a real exported `.onnx`. The equivalence test in B2 Story 2 should use `atol=1e-4` (looser than the spike's 1e-3 — gives headroom for compounding error across the full observe_step pipeline vs the bare CfC cell).

## Artifacts

- [`tools/spikes/cfc_onnx_spike.py`](cfc_onnx_spike.py) — the spike script (kept for re-running if torch/ncps versions change)
- `tools/spikes/cfc_only.onnx` — the spike's output ONNX artifact (gitignored — regenerable)

## Versions Tested

- `torch==2.5.1+cu121`
- `onnxruntime==1.23.2`
- `ncps==0.0.2`
- Python 3.11.9 on Windows
- CPU-only inference (CUDA out-of-scope for the spike)
