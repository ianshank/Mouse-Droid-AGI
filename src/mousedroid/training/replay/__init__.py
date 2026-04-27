"""Real-episode replay loop (Phase 2 — Physical AI roadmap).

Provides chunked, schema-version-guarded LMDB streaming and a deterministic
sim/real episode mixer with optional alpha ramp. All entry points are
opt-in: leaving ``TrainingReplayConfig.enabled=False`` preserves byte-identical
behavior with the pre-Phase-2 dataset loader.
"""

from __future__ import annotations

from mousedroid.training.replay.lmdb_reader import (
    LmdbReplayReader,
    ReplayReaderStats,
    SchemaVersionMismatchError,
)
from mousedroid.training.replay.mixer import EpisodeMixer, MixerStats

__all__ = [
    "EpisodeMixer",
    "LmdbReplayReader",
    "MixerStats",
    "ReplayReaderStats",
    "SchemaVersionMismatchError",
]
