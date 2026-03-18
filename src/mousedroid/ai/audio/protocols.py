"""AI audio protocol interfaces — structural typing for all audio AI components.

All interfaces use ``@runtime_checkable`` structural typing following
the project's protocol-based dependency injection pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Transcription:
    """Speech-to-text transcription result.

    Attributes:
        text: Transcribed text.
        language: Detected language code (e.g. ``"en"``).
        confidence: Average transcription confidence in ``[0.0, 1.0]``.
        duration_s: Duration of audio that was transcribed.
    """

    text: str
    language: str
    confidence: float
    duration_s: float


@dataclass(frozen=True)
class SoundEvent:
    """A classified environmental sound event.

    Attributes:
        label: Sound class label (e.g. ``"speech"``, ``"alarm"``, ``"crash"``).
        confidence: Classification confidence in ``[0.0, 1.0]``.
        category: Top-level category (e.g. ``"human"``, ``"alert"``, ``"ambient"``).
    """

    label: str
    confidence: float
    category: str


@dataclass(frozen=True)
class AudioAIResult:
    """Combined output from all audio AI models for one processing cycle.

    Attributes:
        wake_detected: Whether the wake word was detected.
        transcription: ASR transcription (None if not triggered).
        sound_events: Classified environmental sounds.
        voice_command: Extracted voice command text (None if no command).
        timestamp: Monotonic timestamp.
    """

    wake_detected: bool
    transcription: Transcription | None
    sound_events: list[SoundEvent]
    voice_command: str | None
    timestamp: float


@runtime_checkable
class ASRProtocol(Protocol):
    """Interface for automatic speech recognition models."""

    async def transcribe(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> Transcription | None:
        """Transcribe audio to text.

        Args:
            audio: Audio samples, shape ``(n_samples,)``.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Transcription result, or ``None`` if no speech was detected
            or the audio segment is too short to transcribe.
        """
        ...

    async def start(self) -> None:
        """Load model and prepare for inference."""
        ...

    async def stop(self) -> None:
        """Release model resources."""
        ...


@runtime_checkable
class WakeWordProtocol(Protocol):
    """Interface for wake word detection models."""

    async def detect(self, audio: NDArray[np.int16]) -> bool:
        """Check whether the wake word is present in an audio chunk.

        Args:
            audio: Int16 PCM audio samples, shape ``(chunk_size,)``.

        Returns:
            True if wake word detected.
        """
        ...

    async def start(self) -> None:
        """Load model and prepare for inference."""
        ...

    async def stop(self) -> None:
        """Release model resources."""
        ...


@runtime_checkable
class SoundClassifierProtocol(Protocol):
    """Interface for environmental sound classification models."""

    async def classify(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> list[SoundEvent]:
        """Classify environmental sounds in an audio segment.

        Args:
            audio: Audio samples, shape ``(n_samples,)``.
            sample_rate: Audio sample rate in Hz.

        Returns:
            List of sound events above confidence threshold.
        """
        ...

    async def start(self) -> None:
        """Load model and prepare for inference."""
        ...

    async def stop(self) -> None:
        """Release model resources."""
        ...
