"""Microphone + speaker + Rocky voice runtime validation helpers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.validation.runtime._shared import _DEFAULT_SMOKE_PHRASE

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings


async def capture_microphone_chunk(cfg: Settings) -> NDArray[np.float32] | None:
    """Capture one chunk through the configured microphone driver.

    Returns ``None`` when the microphone is disabled in config.

    Args:
        cfg: Fully resolved settings.

    Returns:
        Captured audio chunk, or ``None`` when disabled.

    Raises:
        RuntimeError: If the configured microphone cannot open its runtime stream.
    """
    # Deferred, name-scoped import so a test's
    # ``monkeypatch.setattr(mousedroid.validation.runtime, "build_microphone",
    # ...)`` — which patches the package's own attribute — is observed here
    # even though this helper now lives in a submodule.
    from mousedroid.validation.runtime import build_microphone

    microphone = build_microphone(cfg)
    if microphone is None:
        return None

    await microphone.start()
    try:
        if getattr(microphone, "_stream", object()) is None:
            msg = "configured microphone device unavailable"
            raise RuntimeError(msg)
        chunk = await microphone.read_chunk()
        return np.asarray(chunk, dtype=np.float32)
    finally:
        await microphone.stop()


async def play_speaker_tone(
    cfg: Settings,
    *,
    duration_s: float = 0.3,
    frequency_hz: float = 440.0,
) -> int | None:
    """Play a short tone through the configured speaker driver.

    Args:
        cfg: Fully resolved settings.
        duration_s: Tone duration.
        frequency_hz: Tone frequency.

    Returns:
        Total number of interleaved samples written (``frames * channels``),
        or ``None`` when the speaker is disabled.

    Raises:
        RuntimeError: If the configured speaker cannot open its runtime stream.
    """
    # Deferred import — see ``capture_microphone_chunk`` for why this is
    # resolved through the package rather than imported at module level.
    from mousedroid.validation.runtime import build_speaker

    speaker = build_speaker(cfg)
    if speaker is None:
        return None

    await speaker.start()
    try:
        if getattr(speaker, "_stream", object()) is None:
            msg = "configured speaker device unavailable"
            raise RuntimeError(msg)

        channels = max(1, int(getattr(speaker, "channels", 1)))
        min_frames = max(1, round(float(speaker.sample_rate) * duration_s))
        total_frames = max(
            speaker.chunk_size,
            math.ceil(min_frames / speaker.chunk_size) * speaker.chunk_size,
        )
        time_axis = np.arange(total_frames, dtype=np.float32) / float(speaker.sample_rate)
        mono_tone = (0.2 * np.sin(2.0 * np.pi * frequency_hz * time_axis)).astype(np.float32)
        # Interleave identical tone across channels so each frame is `channels` samples.
        interleaved = np.repeat(mono_tone, channels) if channels > 1 else mono_tone

        samples_per_chunk = speaker.chunk_size * channels
        total_samples = total_frames * channels
        for start in range(0, total_samples, samples_per_chunk):
            chunk = interleaved[start : start + samples_per_chunk]
            if chunk.shape[0] < samples_per_chunk:
                chunk = np.pad(chunk, (0, samples_per_chunk - chunk.shape[0]))
            await speaker.write_chunk(chunk)

        return total_samples
    finally:
        await speaker.stop()


async def play_rocky_voice_phrase(
    cfg: Settings,
    *,
    phrase: str = _DEFAULT_SMOKE_PHRASE,
) -> tuple[int, float] | None:
    """Play a short Rocky voice phrase through the configured voice pipeline.

    Args:
        cfg: Fully resolved settings.
        phrase: Short phrase to synthesize and play.

    Returns:
        Tuple of ``(samples_written, peak_abs_sample)``, or ``None`` when the
        voice engine is disabled.

    Raises:
        RuntimeError: If the voice pipeline cannot load TTS or write to the
            configured speaker.
    """
    if not cfg.voice.enabled:
        return None

    # Deferred import — see ``capture_microphone_chunk`` for why this is
    # resolved through the package rather than imported at module level.
    from mousedroid.validation.runtime import build_speaker, build_voice_engine

    speaker = build_speaker(cfg)
    if speaker is None:
        raise RuntimeError("configured speaker unavailable for Rocky voice")

    engine = build_voice_engine(cfg, speaker=speaker)
    if engine is None:
        raise RuntimeError("Rocky voice engine unavailable")

    await engine.start()
    try:
        samples_written, peak_abs = await engine.play_phrase(phrase)
        if not cfg.mock_hardware and peak_abs <= 1e-6:
            raise RuntimeError("Rocky voice TTS returned silent audio")
        return samples_written, peak_abs
    finally:
        await engine.stop()
