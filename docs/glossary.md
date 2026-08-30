# Glossary

- **MSE-6** — the Star Wars "mouse droid" this robot replicates.
- **RSSM** — Recurrent State-Space Model; the world model's latent dynamics core.
- **MCTS** — Monte-Carlo Tree Search; planning over RSSM latent rollouts.
- **BDI** — Belief–Desire–Intention cognitive architecture.
- **EWC** — Elastic Weight Consolidation; a continual-learning regulariser.
- **ICM** — Intrinsic Curiosity Module; novelty-driven exploration reward.
- **CfC** — Closed-form Continuous-time network (a liquid neural network); the fast-reflex stream.
- **GoalVector** — the structured target a natural-language mission is translated into.
- **VLA** — Vision-Language-Action policy (the distillation teacher).
- **Three Laws** — Asimov's laws encoded as hard safety constraints (`safety/three_laws.py`).
- **Pillar** — one of the "10 Pillars" cognitive modules; each is *wired*, *not-yet-wired*, or *parked*.
- **Overlay** — a YAML file layered over `config/default.yaml`.
- **Factory** — `src/mousedroid/factory/`, the single DI wiring point that returns protocol types.
- **L4T** — Linux for Tegra; NVIDIA's Jetson base OS / container lineage.
- **ADR / C4** — Architecture Decision Record / the C4 architecture-diagram model.
- **Off-loop** — runs outside the deterministic 30 Hz reactive control loop (LLM translation, learning).
- **F-number** — a feature identifier in the spec harness (`features.yaml`; ADR-012/013).
