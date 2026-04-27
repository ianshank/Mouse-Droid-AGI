"""Phase 2 — Real-episode replay loop.

Async, chunked LMDB replay reader and a deterministic sim/real episode mixer
used to feed real Jetson rollouts back into RSSM and Constitutional-RL
training.

Both components are opt-in (off by default) so existing training pipelines
remain byte-identical until a flag flips.
"""

from __future__ import annotations

from mousedroid.training.replay.lmdb_reader import (
    LMDBReplayReader,
    ReplayReaderProtocol,
)
from mousedroid.training.replay.mixer import MixerConfig, RealSimMixer

__all__ = [
    "LMDBReplayReader",
    "MixerConfig",
    "RealSimMixer",
    "ReplayReaderProtocol",
]
