# Growth Subsystem — Surface Contract

> Knowledge distillation and model compression (VLA teacher → compact student model).

## Invariants & Growth Rules

1. **Default-OFF Off-Loop Execution**: Knowledge distillation is an off-loop batch process
   and is disabled by default in runtime configuration.
2. **Resource Throttling**: Distillation passes run outside the 30 Hz real-time mission loop
   to prevent frame drops on Jetson Orin Nano.
3. **Loss Divergence Protection**: Teacher-student divergence is clamped and monitored.
   Distillation automatically aborts if NaN loss occurs.

## Key Files

- `distillation.py` — Teacher-student knowledge distillation pipeline.
- `compression.py` — Pruning and quantization adapters.
- `tests/unit/growth/` — Unit tests for distillation logic.
