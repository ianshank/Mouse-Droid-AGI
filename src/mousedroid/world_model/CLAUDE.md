# World Model Subsystem — Surface Contract

> Recurrent State Space Model (RSSM) latent dynamics and Monte Carlo Tree Search (MCTS)
> trajectory planning for autonomous navigation.

## Invariants & Dynamics Rules

1. **Latent Space Planning**: MCTS simulation rollouts execute entirely in latent feature space
   without expensive raw-pixel rendering.
2. **ONNX / TensorRT Acceleration**: Pretrained dynamics models execute via ONNX Runtime
   with TensorRT execution provider on Jetson Orin Nano.
3. **No In-Place Tensor Mutation**: Operations maintain functional tensor flow for clean
   gradient and state transitions.
4. **`torch.no_grad()` on Rollouts**: Search trees and evaluation passes must not retain
   computation graphs.

## Key Files

- `rssm.py` — Recurrent State Space Model transition and observation dynamics.
- `mcts.py` — Latent MCTS planner.
- `onnx_io.py`, `dual_stream_rssm_onnx.py` — ONNX Runtime / TensorRT export and execution wrapper.
- `tests/unit/world_model/` — Subsystem unit tests.
