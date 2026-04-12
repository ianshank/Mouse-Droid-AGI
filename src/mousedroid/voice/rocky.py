"""Rocky voice engine — Project Hail Mary personality for the Mouse Droid.

Rocky speaks English in a distinctive broken style: no articles, adjective
repetition for emphasis, blunt declarations, and enthusiastic exclamations.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger
from mousedroid.voice.phrase_bank import DEFAULT_PHRASES

if TYPE_CHECKING:
    from mousedroid.config.schema import VoiceConfig
    from mousedroid.hardware.protocols import SpeakerProtocol
    from mousedroid.voice.mock_tts import MockTTS
    from mousedroid.voice.tts import PiperTTS

_log = get_logger(__name__)


class Priority(IntEnum):
    """Speech request priority levels."""

    NORMAL = 0
    HIGH = 1
    EMERGENCY = 2


@dataclass(order=True)
class SpeechRequest:
    """A queued speech request with priority ordering.

    Higher priority requests are processed first. Within the same
    priority, earlier requests are processed first (FIFO).
    """

    priority: int
    timestamp: float = field(compare=True)
    text: str = field(compare=False)


# Rocky grammar transformation rules
_ARTICLES = {"the", "a", "an"}


def rocky_transform(text: str, intensity: float = 1.0) -> str:
    """Transform plain English into Rocky's speech style.

    Applies Rocky's characteristic grammar patterns:
    - Strips articles (the, a, an)
    - Repeats adjectives based on emotional intensity
    - Keeps sentences short and declarative

    Args:
        text: Plain English text.
        intensity: Emotional intensity 0.0-1.0, controls repetition.

    Returns:
        Rocky-style text.
    """
    words = text.split()
    result: list[str] = []

    for word in words:
        if word.lower() in _ARTICLES:
            continue
        result.append(word)

    transformed = " ".join(result)

    # Add repetition for emphasis at high intensity
    if intensity > 0.7 and transformed and not transformed.endswith("!"):
        transformed = transformed.rstrip(".") + "!"

    return transformed


class RockyVoiceEngine:
    """Rocky-personality voice engine for the Mouse Droid.

    Maintains an async priority queue of speech requests. A background
    task drains the queue, synthesises audio via TTS, and writes chunks
    to the speaker. A cooldown timer prevents the droid from talking
    non-stop.
    """

    def __init__(
        self,
        cfg: VoiceConfig,
        speaker: SpeakerProtocol,
        tts: PiperTTS | MockTTS,
    ) -> None:
        """Initialise the Rocky voice engine.

        Args:
            cfg: Voice engine configuration.
            speaker: Speaker driver for audio output.
            tts: Text-to-speech synthesiser.
        """
        self._cfg = cfg
        self._speaker = speaker
        self._tts = tts
        self._queue: asyncio.PriorityQueue[SpeechRequest] = asyncio.PriorityQueue(
            maxsize=cfg.queue_size,
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._last_speak_time: float = 0.0
        self._running = False

        # Build phrase bank: defaults + user overrides
        self._phrases: dict[str, list[str]] = dict(DEFAULT_PHRASES)
        for event, phrases in cfg.phrase_overrides.items():
            self._phrases[event] = phrases

        _log.info(
            "rocky_voice_engine_init",
            cooldown_s=cfg.cooldown_s,
            queue_size=cfg.queue_size,
            phrase_events=list(self._phrases.keys()),
        )

    async def speak(self, event: str, context: dict[str, float] | None = None) -> None:
        """Queue a speech event for playback.

        Non-blocking: drops the request if the queue is full.

        Args:
            event: Semantic event name (e.g. ``"obstacle_detected"``).
            context: Optional context (e.g. ``{"valence": 0.8}``).
        """
        phrases = self._phrases.get(event)
        if phrases is None:
            _log.debug("rocky_voice_unknown_event", voice_event=event)
            return

        text = random.choice(phrases)  # noqa: S311

        # Determine priority from event semantics
        if event == "emergency_stop":
            priority = Priority.EMERGENCY
        elif event in {"obstacle_detected", "error", "low_battery"}:
            priority = Priority.HIGH
        else:
            priority = Priority.NORMAL

        request = SpeechRequest(
            priority=-priority,  # Negate so higher priority sorts first
            timestamp=time.monotonic(),
            text=text,
        )

        try:
            self._queue.put_nowait(request)
            _log.debug("rocky_voice_queued", voice_event=event, text=text, priority=priority.name)
        except asyncio.QueueFull:
            if priority == Priority.EMERGENCY:
                # Emergency: clear queue and force it in
                await self._drain_queue()
                self._queue.put_nowait(request)
                _log.info("rocky_voice_emergency_queued", text=text)
            else:
                _log.debug("rocky_voice_queue_full", voice_event=event)

    async def start(self) -> None:
        """Start the speaker, TTS engine, and background worker."""
        await self._speaker.start()
        self._tts.start()
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        _log.info("rocky_voice_engine_started")

    async def stop(self) -> None:
        """Stop the background worker, TTS, and speaker."""
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
        self._tts.stop()
        await self._speaker.stop()
        _log.info("rocky_voice_engine_stopped")

    async def _worker(self) -> None:
        """Background task: drain the queue and play speech."""
        while self._running:
            try:
                request = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Respect cooldown (bypass for emergency)
            now = time.monotonic()
            elapsed = now - self._last_speak_time
            is_emergency = request.priority == -Priority.EMERGENCY
            if elapsed < self._cfg.cooldown_s and not is_emergency:
                _log.debug(
                    "rocky_voice_cooldown_skip",
                    text=request.text,
                    remaining_s=self._cfg.cooldown_s - elapsed,
                )
                continue

            await self._play(request.text)
            self._last_speak_time = time.monotonic()

    async def _play(self, text: str) -> None:
        """Synthesise and play a phrase through the speaker.

        Args:
            text: Rocky-style text to speak.
        """
        _log.info("rocky_voice_speaking", text=text)
        try:
            samples = await self._tts.synthesize(text)
            await self._write_samples(samples)
        except Exception:
            _log.warning("rocky_voice_play_failed", text=text, exc_info=True)

    async def _write_samples(self, samples: NDArray[np.float32]) -> None:
        """Write audio samples to the speaker in chunks.

        Args:
            samples: Full audio waveform as float32.
        """
        chunk_size = self._speaker.chunk_size
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            await self._speaker.write_chunk(chunk)

    async def _drain_queue(self) -> None:
        """Remove all items from the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
