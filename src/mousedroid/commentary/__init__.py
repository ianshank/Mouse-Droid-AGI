"""Phase-0 grounded, novelty-gated spoken commentary subsystem.

The droid narrates its *situation* (spatial geometry, acoustic energy, motion,
battery) in Rocky's voice, fired only on statistically-novel, idle/safe moments,
entirely OUTSIDE the 30 Hz reactive control loop. Public types are exposed via
the protocol module; concrete types are wired through
:func:`mousedroid.factory.build_commentary`.
"""

from __future__ import annotations

from mousedroid.commentary.protocol import (
    CommentaryComposerProtocol,
    CommentaryEngineProtocol,
    CommentaryFacts,
    GroundedReferentStoreProtocol,
    SpeakerBusyProtocol,
)

__all__ = [
    "CommentaryComposerProtocol",
    "CommentaryEngineProtocol",
    "CommentaryFacts",
    "GroundedReferentStoreProtocol",
    "SpeakerBusyProtocol",
]
