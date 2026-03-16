#!/bin/bash
set -euo pipefail

echo '=== CONTAINER STATUS ==='
sudo docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'

echo '=== IMPORT TEST ==='
sudo docker exec mousedroid python3 -c 'import mousedroid; print("mousedroid OK")'

echo '=== TORCH/CUDA TEST ==='
sudo docker exec mousedroid python3 -c '
import torch
v = torch.__version__
cuda = torch.cuda.is_available()
gpu = torch.cuda.get_device_name(0) if cuda else None
print(f"torch={v}, cuda={cuda}, gpu={gpu}")
'

echo '=== WEIGHTS TEST ==='
sudo docker exec mousedroid python3 -c '
import os
wdir = "/opt/mousedroid/weights"
files = [os.path.join(r, f) for r, d, fs in os.walk(wdir) for f in fs if f.endswith(".pt")]
print(f"{len(files)} .pt weight files found")
for f in sorted(files)[:8]:
    relpath = os.path.relpath(f, wdir)
    size_kb = os.path.getsize(f) / 1024
    print(f"  {relpath}: {size_kb:.0f} KB")
'

echo '=== LLM FILE TEST ==='
sudo docker exec mousedroid python3 -c '
import os
p = "/home/jetson/models/Phi-3-mini-4k-instruct-q4.gguf"
if os.path.exists(p):
    size_mb = os.path.getsize(p) / 1024 / 1024
    print(f"LLM exists, size={size_mb:.0f} MB")
else:
    print("LLM NOT FOUND at " + p)
'

echo '=== LLAMA_CPP TEST ==='
sudo docker exec mousedroid python3 -c 'from llama_cpp import Llama; print("llama_cpp OK")'

echo '=== ALL SMOKE TESTS PASSED ==='
