"""Audio AI pipeline — orchestrates all audio AI models.

Runs wake word detection, ASR transcription, and sound classification
on microphone audio and produces a unified ``AudioAIResult``.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.ai.audio.protocols import AudioAIResult, SoundEvent, Transcription
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.ai.audio.asr import WhisperASR
    from mousedroid.ai.audio.classifier import YAMNetClassifier
    from mousedroid.ai.audio.wake_word import OpenWakeWordDetector

_log = get_logger(__name__)


class AudioAIPipeline:
    """Unified audio pipeline running all AI models on microphone data.

    Wake word detection runs on every chunk (lightweight).
    ASR and sound classification run only when triggered:
    - ASR runs after wake word detection (for voice commands)
    - Sound classifier runs on periodic windows

    Parameters
    ----------
    asr:
        Whisper ASR engine or None to skip.
    wake_word:
        Wake word detector or None to skip.
    classifier:
        Sound classifier or None to skip.
    command_buffer_seconds:
        How many seconds of audio to buffer for ASR after wake word.
    """

    def __init__(
        self,
        asr: WhisperASR | None = None,
        wake_word: OpenWakeWordDetector | None = None,
        classifier: YAMNetClassifier | None = None,
        command_buffer_seconds: float = 3.0,
        sample_rate: int = 16000,
    ) -> None:
        self._asr = asr
        self._wake_word = wake_word
        self._classifier = classifier
        self._command_buffer_s = command_buffer_seconds
        self._sample_rate = sample_rate

        # State for wake-word → ASR pipeline
        self._wake_detected = False
        self._command_buffer: list[NDArray[np.float32]] = []
        self._command_start_t: float = 0.0

    async def start(self) -> None:
        """Start all configured audio AI models concurrently."""
        coros = []
        if self._asr is not None:
            coros.append(self._asr.start())
        if self._wake_word is not None:
            coros.append(self._wake_word.start())
        if self._classifier is not None:
            coros.append(self._classifier.start())
        if coros:
            await asyncio.gather(*coros)
        _log.info("audio_ai_pipeline_started", models=len(coros))

    async def stop(self) -> None:
        """Stop all configured audio AI models."""
        coros = []
        if self._asr is not None:
            coros.append(self._asr.stop())
        if self._wake_word is not None:
            coros.append(self._wake_word.stop())
        if self._classifier is not None:
            coros.append(self._classifier.stop())
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)
        _log.info("audio_ai_pipeline_stopped")

    async def process(
        self,
        audio_chunk: NDArray[np.float32],
        sample_rate: int = 16000,
    ) -> AudioAIResult:
        """Process an audio chunk through all AI models.

        Args:
            audio_chunk: Float32 audio, shape ``(N,)``, normalised to [-1, 1].
            sample_rate: Audio sample rate.

        Returns:
            Unified audio AI result.
        """
        now = time.time()
        wake_detected = False
        transcription: Transcription | None = None
        voice_command: str | None = None
        sound_events: list[SoundEvent] = []

        # --- Wake word detection (always-on, lightweight) ---
        if self._wake_word is not None:
            # Convert to int16 for OpenWakeWord
            int16_chunk = (audio_chunk * 32767).astype(np.int16)
            wake_detected = await self._wake_word.detect(int16_chunk)

        if wake_detected and not self._wake_detected:
            # Transition: wake word just detected
            self._wake_detected = True
            self._command_buffer = [audio_chunk]
            self._command_start_t = now
            _log.info("wake_word_triggered_starting_command_buffer")
        elif self._wake_detected:
            # Buffering audio after wake word
            self._command_buffer.append(audio_chunk)
            elapsed = now - self._command_start_t

            if elapsed >= self._command_buffer_s:
                # Buffer complete — run ASR
                if self._asr is not None and self._command_buffer:
                    full_audio = np.concatenate(self._command_buffer)
                    transcription = await self._asr.transcribe(full_audio, sample_rate)
                    if transcription is not None:
                        voice_command = transcription.text
                        _log.info(
                            "voice_command_recognised",
                            text=voice_command,
                            confidence=transcription.confidence,
                        )
                # Reset state
                self._wake_detected = False
                self._command_buffer = []

        # --- Sound classification (periodic) ---
        if self._classifier is not None:
            sound_events = await self._classifier.classify(audio_chunk, sample_rate)

        return AudioAIResult(
            wake_detected=wake_detected,
            transcription=transcription,
            sound_events=sound_events,
            voice_command=voice_command,
            timestamp=now,
        )
