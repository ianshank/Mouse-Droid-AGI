# Continual Learning Subsystem — Surface Contract

> Elastic Weight Consolidation (EWC), progressive neural networks, and experience replay
> preventing catastrophic forgetting across navigation domains.

## Invariants & Learning Rules

1. **Inference with `torch.no_grad()`**: All evaluation and forward-pass inference paths
   must be wrapped in `torch.no_grad()`.
2. **Deterministic Fisher Estimation**: Fisher information matrix calculations must respect
   sample count limits configured in `LearningConfig`.
3. **Memory Footprint Bounds**: Replay buffers use bounded capacity with FIFO or reservoir
   sampling to avoid memory exhaustion on edge hardware.
4. **Isolated Task Boundaries**: Progressive column addition respects maximum column capacity.

## Key Files

- `ewc.py` — Elastic Weight Consolidation regularization.
- `progressive.py` — Progressive neural network column adapter.
- `replay.py` — Bounded replay buffer.
- `tests/unit/learning/` — Unit tests for continual learning modules.
