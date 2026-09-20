# Memory Subsystem — Surface Contract

> Layered memory — working, episodic, and semantic tiers with offline consolidation
> (``MemoryProtocol``, ``ReplayBufferProtocol``; ``WorkingMemory``, ``EpisodicReplay``,
> ``SemanticIndex``).

## Invariants & Memory Rules

1. **Protocol Surfaces**: Storage implements ``MemoryProtocol`` (``store`` / ``retrieve``);
   replay buffers implement ``ReplayBufferProtocol`` (``push`` / ``sample`` / ``__len__``).
2. **Tier Separation**: ``MemoryTier`` and the working / episodic / semantic modules keep
   short-term, replay, and long-term stores distinct.
3. **Consolidation**: ``MemoryConsolidation`` replays episodic batches into the semantic
   index offline (batch size from ``MemoryConfig``).

## Key Files

- `protocol.py::MemoryProtocol` / `ReplayBufferProtocol` — storage and replay interfaces.
- `working.py::WorkingMemory` — short-term working memory.
- `episodic.py::EpisodicReplay` — episodic replay buffer.
- `semantic.py::SemanticIndex` — semantic long-term index.
- `consolidation.py::MemoryConsolidation` — offline episodic → semantic consolidation.
- `tier.py::MemoryTier` / `exporter.py` — tier enum and export helpers.
- `tests/unit/memory/` — subsystem unit tests.
