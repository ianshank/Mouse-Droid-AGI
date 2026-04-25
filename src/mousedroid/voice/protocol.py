"""Voice engine protocol — interface for personality-driven speech output."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VoiceEngineProtocol(Protocol):
    """Interface for voice engines that convert robot events to speech."""

    async def speak(self, event: str, context: dict[str, float] | None = None) -> None:
        """Queue a speech event for playback.

        Non-blocking: pushes onto an internal queue. Callers should
        fire-and-forget so the control loop is never delayed.

        Args:
            event: Semantic event name (e.g. ``"obstacle_detected"``).
            context: Optional context values (e.g. ``{"valence": 0.8}``).
        """
        ...

    async def start(self) -> None:
        """Start the voice engine background worker."""
        ...

    async def stop(self) -> None:
        """Stop the voice engine and drain the queue."""
        ...

    async def play_phrase(self, text: str) -> tuple[int, float]:
        """Immediately synthesize and play one phrase for validation flows.

        Args:
            text: Phrase to synthesize and play.

        Returns:
            Tuple of ``(samples_written, peak_abs_sample)``.
        """
        ...
