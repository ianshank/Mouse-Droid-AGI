"""Quick test: can we load weights and use the LLM inside the container?"""
import torch
import os
import glob

# --- Weights ---
wfiles = sorted(glob.glob("/opt/mousedroid/weights/**/*.pt", recursive=True))
print(f"WEIGHTS: {len(wfiles)} .pt files found")
if wfiles:
    final = [f for f in wfiles if "final" in f]
    pick = final[0] if final else wfiles[-1]
    w = torch.load(pick, map_location="cpu", weights_only=False)
    if isinstance(w, dict):
        keys = list(w.keys())[:6]
        print(f"  Checkpoint keys: {keys}")
        total_params = sum(
            v.numel() for v in w.values() if isinstance(v, torch.Tensor)
        )
        print(f"  Total params: {total_params:,}")
    print(f"  Loaded OK: {os.path.basename(pick)}")

# --- LLM file ---
llm_path = "/home/jetson/models/Phi-3-mini-4k-instruct-q4.gguf"
if os.path.exists(llm_path):
    sz = os.path.getsize(llm_path) / (1024**2)
    print(f"LLM FILE: {sz:.0f} MB at {llm_path}")
else:
    print("LLM FILE: NOT FOUND")

# --- LLM inference test ---
try:
    from llama_cpp import Llama
    llm = Llama(model_path=llm_path, n_ctx=512, n_gpu_layers=0, verbose=False)
    out = llm("Q: What is 2+2? A:", max_tokens=16, stop=["\n"])
    answer = out["choices"][0]["text"].strip()
    print(f"LLM INFERENCE: '{answer}'")
    del llm
except Exception as e:
    print(f"LLM INFERENCE ERROR: {e}")

print("TEST COMPLETE")
