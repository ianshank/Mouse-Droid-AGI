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

    def synthesize_wav(self, text: str, wav_file: Any) -> Any: ...

    def synthesize(self, text: str, *args: Any, **kwargs: Any) -> Any: ...


class PiperTTS:
    """Piper text-to-speech synthesiser.

    Wraps ``piper.PiperVoice`` for async-safe speech synthesis.
    """

    def __init__(self, cfg: VoiceConfig) -> None:
        """Initialise TTS from voice config.

        Args:
            cfg: Voice engine configuration.
        """
        self._cfg = cfg
        self._voice: _PiperVoiceLike | None = None
        _log.info(
            "piper_tts_init",
            model_path=cfg.tts_model_path,
            sample_rate=cfg.tts_sample_rate,
        )

    def start(self) -> None:
        """Load the piper voice model."""
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
                _log.info("piper_tts_model_loaded", path=resolved_path)
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

        wav_buffer = io.BytesIO()
        # piper-tts >=1.3 writes a complete WAV into a binary buffer. Older
        # versions wrote frames into a configured wave.Wave_write instead.
        synth_wav = getattr(self._voice, "synthesize_wav", None)
        if callable(synth_wav):
            _log.debug("piper_tts_synthesize_api", api="synthesize_wav")
            synth_wav(text, wav_buffer)
        else:
            _log.debug("piper_tts_synthesize_api", api="synthesize")
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(np.dtype(np.int16).itemsize)
                wav_file.setframerate(self._cfg.tts_sample_rate)
                self._voice.synthesize(text, wav_file)

        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            n_frames = wav_file.getnframes()
            raw = wav_file.readframes(n_frames)
            width = wav_file.getsampwidth()

        if width == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / INT16_MAX_F
        else:
            samples = np.frombuffer(raw, dtype=np.float32)

        return samples

    async def synthesize(self, text: str) -> NDArray[np.float32]:
        """Synthesise text to audio samples (async).

        Runs synthesis in a thread to avoid blocking the event loop.

        Args:
            text: Text to speak.

        Returns:
            Audio samples as float32 array.
        """
        return await asyncio.to_thread(self._synthesize_sync, text)
