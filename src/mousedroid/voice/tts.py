"""Text-to-speech wrapper for piper-tts.

Piper runs locally on ARM64 (Jetson) with no internet required.
Synthesis is offloaded to a thread to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from mousedroid.hardware.audio.constants import INT16_MAX_F
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import VoiceConfig

_log = get_logger(__name__)


class _PiperVoiceLike(Protocol):
    """Minimal interface required from a loaded piper voice."""

    def synthesize_wav(self, text: str) -> bytes: ...

    def synthesize(self, text: str, *args: Any, **kwargs: Any) -> Any: ...


class PiperTTS:
    """Piper text-to-speech synthesiser.

    Wraps ``piper.PiperVoice`` for async-safe speech synthesis.

    Piper-tts has two API generations:
    - New (>=0.0.7): ``synthesize_wav(text) -> bytes`` returns a complete WAV blob.
    - Legacy: ``synthesize(text, wav_file)`` writes PCM frames into a
      ``wave.Wave_write`` object opened by the caller.

    The API version is detected once in ``start()`` and cached so the hot path
    in ``_synthesize_sync`` never re-inspects the object on every call.
    """

    def __init__(self, cfg: VoiceConfig) -> None:
        """Initialise TTS from voice config.

        Args:
            cfg: Voice engine configuration.
        """
        self._cfg = cfg
        self._voice: _PiperVoiceLike | None = None
        self._use_wav_api: bool = False
        self._consecutive_failures: int = 0
        _log.info(
            "piper_tts_init",
            model_path=cfg.tts_model_path,
            sample_rate=cfg.tts_sample_rate,
            output_volume=cfg.output_volume,
        )

    def start(self) -> None:
        """Load the piper voice model and detect the API generation."""
        try:
            from piper import PiperVoice

            resolved_path = self._cfg.resolved_tts_model_path()
            if resolved_path is not None:
                source = (
                    "personality_map"
                    if self._cfg.personality in self._cfg.personality_to_model_map
                    else "tts_model_path"
                )
                _log.info(
                    "piper_tts_model_resolved",
                    personality=self._cfg.personality,
                    path=resolved_path,
                    source=source,
                )
                self._voice = PiperVoice.load(resolved_path)
                self._use_wav_api = hasattr(self._voice, "synthesize_wav")
                _log.info(
                    "piper_tts_model_loaded",
                    path=resolved_path,
                    api="synthesize_wav" if self._use_wav_api else "synthesize",
                )
                _log.info(
                    "voice_tts_api_selected",
                    api="synthesize_wav" if self._use_wav_api else "synthesize",
                )
            else:
                _log.warning("piper_tts_no_model_path")
        except ImportError:
            _log.warning("piper_tts_not_installed")
        except Exception:
            _log.warning("piper_tts_load_failed", exc_info=True)

    def stop(self) -> None:
        """Release the piper voice model."""
        self._voice = None
        _log.info("piper_tts_stopped")

    def _synthesize_via_wav(self, text: str) -> bytes:
        """Synthesise using the modern piper API that returns WAV bytes directly.

        Args:
            text: Text to synthesise.

        Returns:
            Raw WAV file bytes.
        """
        assert self._voice is not None  # guarded by caller
        return self._voice.synthesize_wav(text)

    def _synthesize_via_legacy(self, text: str) -> bytes:
        """Synthesise using the legacy piper API that writes into a wave writer.

        Args:
            text: Text to synthesise.

        Returns:
            Raw WAV file bytes.
        """
        import io
        import wave

        assert self._voice is not None  # guarded by caller
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(np.dtype(np.int16).itemsize)
            wav_file.setframerate(self._cfg.tts_sample_rate)
            self._voice.synthesize(text, wav_file)
        return wav_buffer.getvalue()

    def _synthesize_sync(self, text: str) -> NDArray[np.float32]:
        """Synthesise text to audio samples (blocking).

        Args:
            text: Text to speak.

        Returns:
            Audio samples as float32 array.
        """
        if self._voice is None:
            _log.debug("piper_tts_no_voice_returning_silence")
            return np.zeros(self._cfg.tts_sample_rate, dtype=np.float32)

        import io
        import wave

        try:
            if self._use_wav_api:
                wav_bytes = self._synthesize_via_wav(text)
            else:
                wav_bytes = self._synthesize_via_legacy(text)
            wav_buffer = io.BytesIO(wav_bytes)

            self._consecutive_failures = 0
        except Exception as exc:
            self._consecutive_failures += 1
            log_fn = (
                _log.error
                if self._consecutive_failures >= self._cfg.tts_failure_threshold
                else _log.warning
            )
            log_fn(
                "voice_tts_synthesize_failed",
                api="synthesize_wav" if self._use_wav_api else "synthesize",
                exc_type=type(exc).__name__,
                exc_msg=str(exc),
                consecutive_failures=self._consecutive_failures,
            )
            # TODO: wire voice_tts_synthesize_failures_total Prometheus counter
            # once feat/observability-primitive lands (PR #2).
            return np.zeros(self._cfg.tts_sample_rate, dtype=np.float32)

        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            n_frames = wav_file.getnframes()
            raw = wav_file.readframes(n_frames)
            width = wav_file.getsampwidth()

        if width == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / INT16_MAX_F
        else:
            samples = np.frombuffer(raw, dtype=np.float32).copy()

        if self._cfg.output_volume != 1.0:
            samples = np.clip(samples * np.float32(self._cfg.output_volume), -1.0, 1.0)

        return samples.astype(np.float32, copy=False)

    async def synthesize(self, text: str) -> NDArray[np.float32]:
        """Synthesise text to audio samples (async).

        Runs synthesis in a thread to avoid blocking the event loop.

        Args:
            text: Text to speak.

        Returns:
            Audio samples as float32 array.
        """
        return await asyncio.to_thread(self._synthesize_sync, text)
