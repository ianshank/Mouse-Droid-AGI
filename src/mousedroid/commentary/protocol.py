"""Commentary protocols + the grounded-facts value object.

Defines the structural contracts for the commentary subsystem (kept separate
from concrete implementations per the project's protocol-DI invariant):

* :class:`CommentaryFacts` — an immutable snapshot of the grounded situation a
  comment may be made about. Spatial/acoustic/motion/battery only — there are
  **no semantic object labels** in the mouse-droid loop, so nothing here invents
  object identities.
* :class:`CommentaryComposerProtocol` — turns facts into one plain-English line.
* :class:`CommentaryEngineProtocol` — the out-of-loop engine that samples facts,
  gates on statistical novelty + safety, and speaks.
* :class:`SpeakerBusyProtocol` — OPTIONAL capability seam (mirrors
  ``QueryCapableLLMProtocol``): a voice engine that can report whether it is
  currently producing audio, so commentary can defer. Feature-detected.
* :class:`GroundedReferentStoreProtocol` — **Phase-1 seam, intentionally unused
  in Phase 0**: matches the existing :class:`SemanticIndex` store/retrieve shape
  so a future learning phase can persist and recall novel referents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class CommentaryFacts:
    """Immutable grounded snapshot the engine may narrate about.

    Every numeric field is a plain ``float`` (numpy scalars cast at extraction
    time). ``*_valid`` flags let composers phrase around missing modalities
    instead of asserting facts they don't have. ``novelty`` is ``None`` when no
    curiosity signal is available (distinct from a genuine ``0.0``).
    """

    min_clearance_m: float
    forward_distance_m: float
    audio_rms: float
    speed_mps: float
    turn_rate: float
    battery_v: float
    novelty: float | None
    is_emergency: bool
    lidar_valid: bool
    audio_valid: bool
    timestamp: float
    # Phase-1 seam — the semantic embedding of this moment, carried so a future
    # grounding phase can key a referent on it. Intentionally UNUSED in Phase 0.
    embedding: NDArray[np.float32] | None = None


@runtime_checkable
class CommentaryComposerProtocol(Protocol):
    """Turns grounded facts into one plain-English narration line.

    Implementations return PLAIN English (no Rocky styling — the engine applies
    :func:`rocky_transform` uniformly). An empty string means "nothing worth
    saying", which the engine treats as a suppression (never spoken).
    """

    async def compose(self, facts: CommentaryFacts) -> str:
        """Return a plain-English line, or ``""`` for nothing to say."""
        ...


@runtime_checkable
class CommentaryEngineProtocol(Protocol):
    """Out-of-loop commentary engine.

    ``observe_emergency`` is called every control tick (O(1)); ``observe`` is
    called on a stride (~2 Hz) with the freshly-sampled novelty + facts;
    ``run`` is the background cadence loop spawned by the orchestrator.
    """

    def observe_emergency(self, is_emergency: bool) -> None:
        """Record this tick's emergency state (cheap, every-tick)."""
        ...

    def observe(self, novelty: float | None, facts: CommentaryFacts | None) -> None:
        """Feed a strided novelty sample + facts snapshot (cheap, ~2 Hz)."""
        ...

    async def run(self) -> None:
        """Background loop: gate on novelty + safety, compose, speak."""
        ...

    async def stop(self) -> None:
        """Signal shutdown (the spawning task is cancelled+drained externally)."""
        ...


@runtime_checkable
class SpeakerBusyProtocol(Protocol):
    """Optional capability: a voice engine that reports active playback.

    Separate from :class:`~mousedroid.voice.protocol.VoiceEngineProtocol` so
    existing engines/test-doubles stay valid; the commentary engine
    feature-detects with ``isinstance(voice, SpeakerBusyProtocol)``. NOTE: this
    only lets commentary avoid *starting* over playback — it cannot preempt an
    in-flight phrase. Full serialisation of speaker writes is a follow-up.
    """

    @property
    def is_speaking(self) -> bool:
        """True while the engine is actively writing audio to the speaker."""
        ...


@runtime_checkable
class GroundedReferentStoreProtocol(Protocol):
    """Phase-1 seam (UNUSED in Phase 0): persist/recall grounded referents.

    Matches the existing :class:`mousedroid.memory.semantic.SemanticIndex`
    store/retrieve shape so a future learning phase can map a novel moment's
    embedding to a label and recall it. Defined here to pin the contract; the
    Phase-0 engine never calls it.
    """

    def store(self, key: str, value: NDArray[np.float32]) -> None:
        """Persist a labelled embedding."""
        ...

    def retrieve(self, query: NDArray[np.float32], k: int = 1) -> list[tuple[str, float]]:
        """Return the ``k`` nearest ``(key, distance)`` pairs, nearest first."""
        ...


__all__ = [
    "CommentaryComposerProtocol",
    "CommentaryEngineProtocol",
    "CommentaryFacts",
    "GroundedReferentStoreProtocol",
    "SpeakerBusyProtocol",
]
