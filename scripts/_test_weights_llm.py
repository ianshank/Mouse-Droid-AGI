"""Quick test: can we load weights and use the LLM inside the container?"""

from __future__ import annotations

import glob
import os

import torch


def _load_checkpoint(path: str):
    """Load checkpoint with safest available mode for this torch version."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch versions don't support weights_only.
        return torch.load(path, map_location="cpu")


def main() -> int:
    failed = False

    # --- Weights ---
    wfiles = sorted(glob.glob("/opt/mousedroid/weights/**/*.pt", recursive=True))
    print(f"WEIGHTS: {len(wfiles)} .pt files found")
    if wfiles:
        final = [f for f in wfiles if "final" in f]
        pick = final[0] if final else wfiles[-1]
        try:
            weights = _load_checkpoint(pick)
        except Exception as exc:
            print(f"  ERROR loading weights from {pick}: {exc}")
            failed = True
        else:
            if isinstance(weights, dict):
                keys = list(weights.keys())[:6]
                print(f"  Checkpoint keys: {keys}")
                total_params = sum(
                    value.numel() for value in weights.values() if isinstance(value, torch.Tensor)
                )
                print(f"  Total params: {total_params:,}")
            print(f"  Loaded OK: {os.path.basename(pick)}")
    else:
        print("  ERROR no .pt weights found under /opt/mousedroid/weights")
        failed = True

    # --- LLM file ---
    llm_path = "/home/jetson/models/Phi-3-mini-4k-instruct-q4.gguf"
    llm_exists = os.path.exists(llm_path)
    if llm_exists:
        size_mb = os.path.getsize(llm_path) / (1024**2)
        print(f"LLM FILE: {size_mb:.0f} MB at {llm_path}")
    else:
        print("LLM FILE: NOT FOUND")
        failed = True

    # --- LLM inference test ---
    if llm_exists:
        try:
            from llama_cpp import Llama

            llm = Llama(model_path=llm_path, n_ctx=512, n_gpu_layers=0, verbose=False)
            out = llm("Q: What is 2+2? A:", max_tokens=16, stop=["\n"])
            answer = out["choices"][0]["text"].strip()
            print(f"LLM INFERENCE: '{answer}'")
            del llm
        except Exception as exc:
            print(f"LLM INFERENCE ERROR: {exc}")
            failed = True
    else:
        print("LLM INFERENCE: SKIPPED (model file not found)")

    if failed:
        print("TEST FAILED")
        return 1

    print("TEST COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
