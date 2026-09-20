# Cognitive Subsystem — Surface Contract

> Dual-cadence cognitive loop: Belief-Desire-Intention (BDI), metacognition, and
> constitutional RL — numpy-only (``CognitiveCore``, ``NeuralBDI``, ``MetacognitiveModel``).

## Invariants & Cognitive Rules

1. **Dual Cadence**: ``CognitiveCore.tick_fast`` targets the 30 Hz loop (<1 ms); a slow
   loop (~1 Hz) runs BDI inference and metacognitive updates.
2. **Numpy-Only Compute**: Cognitive modules use numpy arrays — no torch dependency in
   this package's hot path.
3. **Exported Surface**: Public imports are ``CognitiveCore``, ``NeuralBDI``, and
   ``MetacognitiveModel`` (see package ``__init__.py``).

## Key Files

- `cognitive_core.py::CognitiveCore` — dual-cadence controller (fast constitutional /
  curiosity tick + slow BDI / metacognitive loop).
- `bdi_model.py::NeuralBDI` — neural Belief-Desire-Intention model.
- `metacognitive.py::MetacognitiveModel` — metacognitive state updates.
- `constitutional_rl.py` — constitutional checker, curiosity aggregator, and policy MLP.
- `tests/unit/cognitive/` — subsystem unit tests.
