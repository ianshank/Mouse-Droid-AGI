# Design — mouse-droid-alayaworld-memory-distill

The full design lives in the repo-native (authoritative) design spec:

**`docs/superpowers/specs/2026-07-23-alayaworld-memory-distill-design.md`**

Summary of the locked decisions (see the design spec for rationale, ASCII
architecture, and the incorporated 3-agent adversarial peer-review findings):

- **D1** — `BoundedContextMemory` operates at the orchestrator observe seam on the
  carried `(h, z)`; no world-model signature change. Sink = frozen per-mission
  anchor; compressed history = recent ring (`deque(maxlen=recent_size)`) + one EMA
  long-summary vector; softmax-attention retrieval; λ-blend. Default-OFF; exact
  identity when disabled/empty/λ=0. NaN-skip contract (unhealthy ticks never touch
  the memory); sink re-arms on OTA weight swap and at mission boundaries.
- **D2** — Drift training on the concrete `RSSM` feasibility vehicle only (the
  sole `train_sequence` engine; the deployed `DualStreamRSSM` port is explicitly
  deferred — ADR-015). `train_sequence_corrupted` = open-loop prior prefix
  (self-generated corruption, no_grad) + posterior recovery suffix via a shared
  per-step helper; forced prefix length 0 is allclose-identical to
  `train_sequence`. Drift metric = deterministic seeded per-modality open-loop
  MSE (`measure_drift`), range headline, zero-filled channels excluded, plus
  latent divergence; no pose channel exists on this robot.
- **D3** — Distillation spike is scripts-only and non-binding: deterministic
  prior-MEAN k-step teacher, MLP jump student, existing
  `KnowledgeDistiller(objective="regression")`; the report separates
  primitive-level speedup from the ~1.25-1.6× MCTS consumer-level ceiling and
  keeps the Jetson criterion pending an operator run.

Companion decisions record: `docs/architecture/ADR-015-bounded-context-latent-memory.md`.
