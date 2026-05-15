"""Rocky voice engine — Project Hail Mary personality for the Mouse Droid.

Rocky speaks English in a distinctive broken style: no articles, adjective
repetition for emphasis, blunt declarations, and enthusiastic exclamations.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.common.rate_limit import TokenBucket
from mousedroid.common.time.protocol import RealClock
from mousedroid.logging.setup import get_logger
from mousedroid.telemetry.failure_recorder import NullFailureRecorder
from mousedroid.voice.exceptions import SpeakerUnavailableError
from mousedroid.voice.phrase_bank import DEFAULT_PHRASES

if TYPE_CHECKING:
    from mousedroid.common.time.protocol import ClockProtocol
    from mousedroid.config.schema import VoiceConfig
    from mousedroid.hardware.protocols import SpeakerProtocol
    from mousedroid.telemetry.failure_recorder import FailureRecorder
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


_ADJECTIVES = {
    "good",
    "bad",
    "big",
    "small",
    "happy",
    "safe",
    "clear",
    "new",
    "fast",
    "slow",
    "hot",
    "cold",
    "hard",
    "easy",
    "old",
    "young",
    "critical",
    "dangerous",
    "important",
    "strange",
    "interesting",
}
"""Common adjectives eligible for Rocky-style repetition."""

_INTENSITY_THRESHOLD_DEFAULT: float = 0.7
"""Default minimum intensity (from VoiceConfig.intensity_threshold)."""


def rocky_transform(
    text: str,
    intensity: float = 1.0,
    intensity_threshold: float = _INTENSITY_THRESHOLD_DEFAULT,
) -> str:
    """Transform plain English into Rocky's speech style.

    Applies Rocky's characteristic grammar patterns:

    - Strips articles (the, a, an)
    - Repeats recognised adjectives based on emotional intensity
    - Preserves capitalisation of the first word after article stripping
    - Adds exclamation mark at high intensity

    Args:
        text: Plain English text.
        intensity: Emotional intensity 0.0-1.0, controls repetition.
        intensity_threshold: Minimum intensity for effects (from VoiceConfig).

    Returns:
        Rocky-style text.
    """
    was_capitalised = bool(text) and text[0].isupper()
    words = text.split()
    result: list[str] = []

    for word in words:
        if word.lower() in _ARTICLES:
            continue
        # Repeat adjectives at high intensity
        if intensity > intensity_threshold and word.lower() in _ADJECTIVES:
            reps = 2 if intensity < 0.9 else 3
            result.extend([word] * reps)
        else:
            result.append(word)

    if not result:
        return ""

    # Preserve capitalisation when a leading article was stripped
    if was_capitalised:
        result[0] = result[0].capitalize()

    transformed = " ".join(result)

    # Add exclamation for emphasis at high intensity
    if intensity > intensity_threshold and not transformed.endswith("!"):
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
        failure_recorder: FailureRecorder | None = None,
        clock: ClockProtocol | None = None,
    ) -> None:
        """Initialise the Rocky voice engine.

        Args:
            cfg: Voice engine configuration.
            speaker: Speaker driver for audio output.
            tts: Text-to-speech synthesiser.
            failure_recorder: Optional :class:`FailureRecorder` used to
                report dropped events (rate-limit or cooldown). Defaults to
                :class:`NullFailureRecorder` (no-op) so unit tests and
                offline contexts work without telemetry wiring.
            clock: Optional :class:`ClockProtocol` for cooldown and
                rate-limit time. Defaults to :class:`RealClock`
                (production); inject :class:`MockClock` in tests to
                drive simulated time without wall-clock waits. Addresses
                Gemini high-priority review (PR #76).
        """
        from mousedroid.config.schema import SpeakerConfig

        self._cfg = cfg
        self._speaker = speaker
        self._tts = tts
        # Capture speaker config at construction so the degradation path in
        # start() can build a MockSpeaker without accessing private attrs.
        _cfg_attr = getattr(speaker, "_cfg", None)
        self._speaker_cfg: SpeakerConfig = (
            _cfg_attr if isinstance(_cfg_attr, SpeakerConfig) else SpeakerConfig.model_validate({})
        )
        self._failure_recorder: FailureRecorder = failure_recorder or NullFailureRecorder()
        self._clock: ClockProtocol = clock if clock is not None else RealClock()
        self._queue: asyncio.PriorityQueue[SpeechRequest] = asyncio.PriorityQueue(
            maxsize=cfg.queue_size,
        )
        self._worker_task: asyncio.Task[None] | None = None
        # Cooldown tracking. Events that have an entry in
        # ``cfg.cooldown_per_event`` are tracked individually in
        # ``_last_speak_time_per_event``; all other events share the global
        # ``_last_speak_time`` slot (preserving the legacy single-cooldown
        # semantics for unconfigured events).
        self._last_speak_time: float = 0.0
        self._last_speak_time_per_event: dict[str, float] = {}
        self._running = False

        # One token bucket per non-emergency priority class. Emergency events
        # are never rate-limited so they don't appear here. Uses the shared
        # ``TokenBucket`` from ``mousedroid.common.rate_limit`` (DRY — also
        # used by the MCP and REST control planes) with the engine's clock
        # injected so test runs are deterministic.
        self._buckets: dict[Priority, TokenBucket] = {
            Priority.HIGH: TokenBucket(
                rate_per_s=cfg.token_bucket_refill_rate,
                capacity=float(cfg.token_bucket_capacity),
                clock=self._clock,
            ),
            Priority.NORMAL: TokenBucket(
                rate_per_s=cfg.token_bucket_refill_rate,
                capacity=float(cfg.token_bucket_capacity),
                clock=self._clock,
            ),
        }

        # Build phrase bank: defaults + user overrides
        self._phrases: dict[str, list[str]] = dict(DEFAULT_PHRASES)
        for event, phrases in cfg.phrase_overrides.items():
            self._phrases[event] = phrases

        _log.info(
            "rocky_voice_engine_init",
            cooldown_s=cfg.cooldown_s,
            queue_size=cfg.queue_size,
            token_bucket_capacity=cfg.token_bucket_capacity,
            token_bucket_refill_rate=cfg.token_bucket_refill_rate,
            phrase_events=list(self._phrases.keys()),
        )

    def _get_cooldown(self, event: str) -> float:
        """Return effective cooldown for an event (per-event override or global)."""
        return self._cfg.cooldown_per_event.get(event, self._cfg.cooldown_s)

    async def speak(self, event: str, context: dict[str, float] | None = None) -> None:
        """Queue a speech event for playback.

        Non-blocking. Non-emergency events are gated by:

        - **Per-event cooldown** — if the event was spoken less than its
          effective cooldown ago, the request is dropped and recorded as
          ``event_dropped_cooldown``.
        - **Token-bucket backpressure** — each non-emergency priority class
          owns a token bucket; if the bucket is empty when the event
          arrives, the request is dropped and recorded as
          ``event_dropped_rate_limit``.

        Emergency events bypass both gates and forcibly drain the queue.

        Args:
            event: Semantic event name (e.g. ``"obstacle_detected"``).
            context: Optional context (e.g. ``{"valence": 0.8}``).
        """
        phrases = self._phrases.get(event)
        if not phrases:
            _log.debug("rocky_voice_no_phrases_for_event", voice_event=event)
            return

        # Phrase selection is for personality, not security — random.choice
        # is fine here. S311 flags non-crypto random for sensitive contexts.
        text = random.choice(phrases)  # noqa: S311

        # Apply Rocky personality transform using context intensity
        intensity = context.get("valence", 1.0) if context else 1.0
        effective_threshold = self._cfg.event_intensity_thresholds.get(
            event, self._cfg.intensity_threshold
        )
        text = rocky_transform(
            text,
            intensity=intensity,
            intensity_threshold=effective_threshold,
        )
        _log.debug(
            "rocky_voice_transform",
            voice_event=event,
            intensity=intensity,
            effective_threshold=effective_threshold,
            transformed_text=text,
        )

        # Determine priority from event semantics
        if event == "emergency_stop":
            priority = Priority.EMERGENCY
        elif event in {"obstacle_detected", "error", "low_battery"}:
            priority = Priority.HIGH
        else:
            priority = Priority.NORMAL

        # Per-event cooldown gate (non-emergency only). Burst-suppression
        # happens here at enqueue-time so the worker never has to throw
        # away cooled-down requests it already dequeued.
        #
        # Events with an explicit per-event cooldown are tracked
        # individually; all other events share the global last-speak slot,
        # preserving legacy single-cooldown semantics for unconfigured
        # events.
        if priority != Priority.EMERGENCY:
            cooldown_s = self._get_cooldown(event)
            has_per_event = event in self._cfg.cooldown_per_event
            last = (
                self._last_speak_time_per_event.get(event, 0.0)
                if has_per_event
                else self._last_speak_time
            )
            elapsed = self._clock.monotonic() - last
            if elapsed < cooldown_s:
                self._record_drop(
                    event=event,
                    reason="event_dropped_cooldown",
                    priority=priority,
                    extra={
                        "cooldown_s": float(cooldown_s),
                        "elapsed_s": float(elapsed),
                    },
                )
                return

            # Token-bucket backpressure gate (per priority class).
            bucket = self._buckets[priority]
            taken, _retry_hint_s = await bucket.take()
            if not taken:
                self._record_drop(
                    event=event,
                    reason="event_dropped_rate_limit",
                    priority=priority,
                )
                return

        request = SpeechRequest(
            priority=-priority,  # Negate so higher priority sorts first
            timestamp=self._clock.monotonic(),
            text=text,
        )

        # Reserve the cooldown slot at enqueue time so subsequent burst
        # events in the same cooldown window are suppressed even if the
        # worker hasn't drained the queue yet.
        if priority != Priority.EMERGENCY:
            now = self._clock.monotonic()
            if event in self._cfg.cooldown_per_event:
                self._last_speak_time_per_event[event] = now
            else:
                self._last_speak_time = now

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
                self._record_drop(
                    event=event,
                    reason="event_dropped_rate_limit",
                    priority=priority,
                    extra={"queue_full": 1},
                )

    def _record_drop(
        self,
        *,
        event: str,
        reason: str,
        priority: Priority,
        extra: dict[str, float | int] | None = None,
    ) -> None:
        """Emit structured log + failure-recorder metric for a dropped event.

        Args:
            event: Voice event name being dropped.
            reason: Machine-readable drop reason (``event_dropped_cooldown``
                or ``event_dropped_rate_limit``).
            priority: Priority class of the dropped event.
            extra: Optional additional structured fields.
        """
        log_kv: dict[str, str | int | float] = {
            "voice_event": event,
            "reason": reason,
            "priority": priority.name,
        }
        recorder_extra: dict[str, str | int | float] = {
            "event": event,
            "priority": priority.name,
        }
        if extra:
            for k, v in extra.items():
                log_kv[k] = v
                recorder_extra[k] = v

        _log.debug("voice_event_dropped", **log_kv)
        self._failure_recorder.record(
            subsystem="voice",
            reason=reason,
            level="warning",
            extra=recorder_extra,
        )

    async def start(self) -> None:
        """Start the speaker, TTS engine, and background worker.

        If the hardware speaker raises ``SpeakerUnavailable``, automatically
        downgrades to a ``MockSpeaker`` and logs a WARNING so the operator has
        a visible signal. The orchestrator continues; voice output is silenced.
        """
        try:
            await self._speaker.start()
        except SpeakerUnavailableError as exc:
            from mousedroid.hardware.audio.mock_speaker import MockSpeaker

            self._speaker = MockSpeaker(self._speaker_cfg)
            await self._speaker.start()
            _log.warning(
                "voice_speaker_degraded",
                reason=str(exc),
                fallback="MockSpeaker",
            )
            # TODO: wire voice_speaker_degraded_total Prometheus counter once
            # feat/observability-primitive lands (PR #2).
        self._tts.start()
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        _log.info("rocky_voice_engine_started")

    async def stop(self) -> None:
        """Stop the background worker, drain queued speech, TTS, and speaker."""
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
        await self._drain_queue()
        self._tts.stop()
        await self._speaker.stop()
        _log.info("rocky_voice_engine_stopped")

    async def play_phrase(self, text: str) -> tuple[int, float]:
        """Immediately synthesize and play one phrase.

        This bypasses the event queue and is intended for smoke checks and
        validation flows that need a deterministic one-shot playback.

        Args:
            text: Phrase to synthesize and play.

        Returns:
            Tuple of ``(samples_written, peak_abs_sample)``.

        Raises:
            RuntimeError: If synthesis returns no samples or the configured
                speaker runtime is unavailable.
        """
        if getattr(self._speaker, "_stream", object()) is None:
            raise RuntimeError("configured speaker device unavailable")

        samples = np.asarray(await self._tts.synthesize(text), dtype=np.float32)
        if samples.size == 0:
            raise RuntimeError("Rocky voice TTS returned no samples")

        peak_abs = float(np.max(np.abs(samples)))
        await self._write_samples(samples)
        return int(samples.size), peak_abs

    async def _worker(self) -> None:
        """Background task: drain the queue and play speech.

        Cooldown and rate-limit gating are now performed at enqueue time in
        :meth:`speak`, so the worker simply plays whatever it dequeues.
        """
        while self._running:
            try:
                request = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self._cfg.queue_poll_timeout_s,
                )
            except (TimeoutError, asyncio.TimeoutError):
                continue
            except asyncio.CancelledError:
                break

            await self._play(request.text)

    async def _play(self, text: str) -> None:
        """Synthesise and play a phrase through the speaker.

        Args:
            text: Rocky-style text to speak.
        """
        _log.info("rocky_voice_speaking", text=text)
        try:
            await self.play_phrase(text)
        except Exception:
            _log.warning("rocky_voice_play_failed", text=text, exc_info=True)

    async def _write_samples(self, samples: NDArray[np.float32]) -> None:
        """Write audio samples to the speaker in chunks.

        For stereo speakers, mono TTS output is duplicated across channels.
        Incomplete final chunks are zero-padded to ``frame_size``.

        Args:
            samples: Full audio waveform as float32 (mono from TTS).
        """
        if len(samples) == 0:
            _log.debug("rocky_voice_write_empty_samples")
            return

        channels = self._speaker.channels
        if channels > 1:
            # Duplicate mono TTS output across all channels
            samples = np.repeat(samples, channels)

        frame_size = self._speaker.chunk_size * channels
        for i in range(0, len(samples), frame_size):
            chunk = samples[i : i + frame_size]
            if len(chunk) < frame_size:
                chunk = np.pad(chunk, (0, frame_size - len(chunk)))
            await self._speaker.write_chunk(chunk)

    async def _drain_queue(self) -> None:
        """Remove all items from the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
