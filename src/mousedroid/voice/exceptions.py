"""Voice subsystem exception types."""

from __future__ import annotations


class SpeakerUnavailableError(RuntimeError):
    """Raised when the USB speaker cannot be opened after all retry attempts.

    Callers (e.g. RockyVoiceEngine) should catch this and downgrade to a
    MockSpeaker so the orchestrator continues operating with a visible warning.
    """
