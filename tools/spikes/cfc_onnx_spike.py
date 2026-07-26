"""Tier B2 Story 0 — CfC ONNX exportability spike.

This script answers a single question that gates the rest of B2:

    Can ``ncps.torch.CfC`` (wrapped by ``CfCWrapper``) be exported to ONNX
    via ``torch.onnx.export`` cleanly?

If **yes**: Story B2.1 proceeds with the full ``observe_step`` export
(GRU + CfC fused in one ONNX graph).

If **no**: pivot to "GRU-only ONNX, CfC-in-Python wrapper" — slower but
unblocks B2. The decision is documented in
``tools/spikes/CFC_ONNX_SPIKE_REPORT.md``.

Usage:
    python tools/spikes/cfc_onnx_spike.py --output tools/spikes/cfc_only.onnx

This is a throwaway de-risk script; ``tools/spikes/`` is intentionally
separate from the production ``scripts/`` directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.cfc_cell import CfCWrapper


class _CfCExportShim(nn.Module):
    """Thin wrapper that exposes a single (x, h) -> new_h forward.

    Bypasses the keyword-only ``dt`` parameter on CfCWrapper.forward which
    confuses torch.onnx tracing on some torch versions. The wrapper inside
    still gets its full functionality; the shim just narrows the API for
    tracing.
    """

    def __init__(self, cfc: CfCWrapper) -> None:
        super().__init__()
        self._cfc = cfc

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self._cfc(x, h)


# Spike parameters — tiny dims so the export runs fast.
_INPUT_DIM = 64
_HIDDEN_DIM = 32
_BATCH_SIZE = 1
_OPSET = 17
_NUMERICAL_TOL = 1e-3  # ONNX/torch tend to drift; 1e-3 is realistic
_INFERENCE_ITERATIONS = 5


def _run_torch(cfc: CfCWrapper, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Run the PyTorch CfC forward pass."""
    cfc.train(False)
    with torch.no_grad():
        return cfc(x, h)


def _build_onnx_session(model_path: Path):  # type: ignore[no-untyped-def]
    """Build a fresh ``onnxruntime.InferenceSession`` on the CPU provider.

    Kept separate from :func:`_run_onnx_with_session` so the stability
    check loop below can reuse one session across iterations — building
    a session per call reloads the model from disk and re-allocates ORT
    runtime resources, which is both slow and unrepresentative of how
    production inference behaves.
    """
    import onnxruntime as ort

    return ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],  # CPU is enough for the spike
    )


def _run_onnx_with_session(sess, x: torch.Tensor, h: torch.Tensor):  # type: ignore[no-untyped-def]
    """Run inference on an already-built ORT session."""
    outputs = sess.run(
        None,
        {"x": x.detach().cpu().numpy(), "h": h.detach().cpu().numpy()},
    )
    return outputs[0]


def _run_onnx(model_path: Path, x: torch.Tensor, h: torch.Tensor):  # type: ignore[no-untyped-def]
    """One-shot helper: build a session + run inference + drop the session.

    Used for the equivalence check (single call). The repeated-call
    stability check uses :func:`_build_onnx_session` directly so it can
    reuse the session across iterations.
    """
    sess = _build_onnx_session(model_path)
    return _run_onnx_with_session(sess, x, h)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/spikes/cfc_only.onnx"),
        help="Output .onnx path",
    )
    parser.add_argument("--input-dim", type=int, default=_INPUT_DIM, help="CfC input dim")
    parser.add_argument("--hidden-dim", type=int, default=_HIDDEN_DIM, help="CfC hidden dim")
    parser.add_argument("--opset", type=int, default=_OPSET, help="ONNX opset version")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Build the CfC wrapper in inference mode + export shim.
    print(
        f"[1/4] Building CfCWrapper(input_dim={args.input_dim}, "
        f"cfc_hidden_dim={args.hidden_dim})..."
    )
    cfg = ModelConfig(cfc_hidden_dim=args.hidden_dim)
    cfc_raw = CfCWrapper(input_dim=args.input_dim, cfg=cfg)
    cfc_raw.train(False)
    cfc = _CfCExportShim(cfc_raw)
    cfc.train(False)

    example_x = torch.zeros(_BATCH_SIZE, args.input_dim, dtype=torch.float32)
    example_h = cfc_raw.initial_state(batch_size=_BATCH_SIZE, device=torch.device("cpu"))
    print(
        f"      example_x.shape={tuple(example_x.shape)}, example_h.shape={tuple(example_h.shape)}"
    )

    # Step 2: Capture the torch reference output (before export, since
    # ONNX export can mutate the model in some edge cases).
    print("[2/4] Capturing torch reference output...")
    torch_out = _run_torch(cfc, example_x, example_h)
    print(f"      torch_out.shape={tuple(torch_out.shape)}, dtype={torch_out.dtype}")

    # Step 3: Attempt the ONNX export.
    print(f"[3/4] Exporting to {args.output} (opset={args.opset})...")
    export_start = time.perf_counter()
    try:
        torch.onnx.export(
            cfc,
            (example_x, example_h),
            str(args.output),
            opset_version=args.opset,
            input_names=["x", "h"],
            output_names=["new_h"],
            dynamic_axes={
                "x": {0: "batch"},
                "h": {0: "batch"},
                "new_h": {0: "batch"},
            },
            do_constant_folding=True,
        )
    except Exception as exc:
        export_elapsed = time.perf_counter() - export_start
        print(f"\n      [FAIL] EXPORT FAILED after {export_elapsed:.2f}s")
        print(f"      Exception type: {type(exc).__name__}")
        print(f"      Message: {exc!s}")
        print("\nDecision: pivot to GRU-only ONNX, CfC-in-Python wrapper.")
        print("See tools/spikes/CFC_ONNX_SPIKE_REPORT.md for the full record.")
        return 2  # non-zero so CI can detect

    export_elapsed = time.perf_counter() - export_start
    print(f"      [OK] Export succeeded in {export_elapsed:.2f}s")
    print(f"      ONNX artifact: {args.output} ({args.output.stat().st_size} bytes)")

    # Step 4: Load the .onnx via onnxruntime + numerical-equivalence check.
    print("[4/4] Validating ONNX inference + numerical equivalence...")
    try:
        onnx_out_np = _run_onnx(args.output, example_x, example_h)
    except Exception as exc:
        print(f"      [FAIL] ONNX inference FAILED: {type(exc).__name__}: {exc!s}")
        print("\nDecision: pivot to GRU-only ONNX, CfC-in-Python wrapper.")
        return 2

    torch_out_np = torch_out.detach().cpu().numpy()
    if torch_out_np.shape != onnx_out_np.shape:
        print(f"      [FAIL] Shape mismatch: torch={torch_out_np.shape}, onnx={onnx_out_np.shape}")
        return 2

    max_abs_diff = float(abs(torch_out_np - onnx_out_np).max())
    if max_abs_diff > _NUMERICAL_TOL:
        print(
            f"      [WARN] Numerical drift max_abs_diff={max_abs_diff:.6f} "
            f"exceeds tol={_NUMERICAL_TOL}"
        )
        print(
            "      Spike PARTIAL: export ran but numerical equivalence is "
            "outside the configured tolerance. Investigate the divergence "
            "before committing to full B2.1 export."
        )
        return 1
    print(
        f"      [OK] Numerical equivalence: max_abs_diff={max_abs_diff:.2e} (tol={_NUMERICAL_TOL})"
    )

    # Inference-stability check across multiple calls — catches non-determinism
    # in the export. Reuse one ORT session across iterations: building a new
    # session per call would reload the .onnx from disk and re-allocate ORT
    # resources, which is both slow and not what production inference does.
    stability_session = _build_onnx_session(args.output)
    for i in range(_INFERENCE_ITERATIONS):
        out_i = _run_onnx_with_session(stability_session, example_x, example_h)
        if abs(out_i - onnx_out_np).max() > 1e-9:
            print(f"      [WARN] Iteration {i} non-deterministic — diverges from first run")
            return 1

    print("\n[OK] Spike SUCCESS — proceed with B2 Story 1 (full observe_step export).")
    print("Update tools/spikes/CFC_ONNX_SPIKE_REPORT.md with run details.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
