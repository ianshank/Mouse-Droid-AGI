# Spec Delta: world-model-memory

## ADDED Requirements

### Requirement: Bounded-Context Memory Manager

The world model runtime SHALL maintain a bounded-size context consisting of a
persistent anchor state ("sink") and a compressed rolling history, rather than an
unbounded or naively truncated history buffer.

Implementation note (declared adaptation): MouseDroid's world model consumes a
recurrent latent `(h, z)`, not frames. The memory manager
(`src/mousedroid/world_model/bounded_context.py`) therefore stores latent vectors —
sink = frozen `concat(h, z)` anchor (per-mission, re-armed on OTA weight swap),
compressed history = a recent ring (`deque(maxlen=recent_size)`) plus one EMA
long-summary vector — and incorporates them by blending an attention-retrieved
context into the orchestrator-carried `(h, z)` at the observe seam
(`h' = (1-λ)h + λ·c_h`). The module is ablation-switchable: the config block is
Optional/default-OFF and the disabled path is byte-identical to the pre-feature
tick. Unhealthy (non-finite) ticks never enter or read the memory.

#### Scenario: Long-horizon rollout maintains bounded memory size

- **GIVEN** the droid agent executes a rollout exceeding the configured history window
- **WHEN** the memory manager updates its internal state
- **THEN** total memory size SHALL remain bounded (constant with respect to
  rollout length — `recent_size + 2` vectors) while retaining the persistent sink
  state and a compressed summary of recent history

#### Scenario: Sink state persists across long rollouts

- **GIVEN** a rollout has proceeded well beyond the compressed-history window
- **WHEN** the agent queries the world model for a prediction
- **THEN** the persistent sink state SHALL still be accessible and incorporated
  into the prediction (verified by an automated test asserting the
  sink-present prediction output measurably differs from the sink-ablated
  (λ=0) path)
