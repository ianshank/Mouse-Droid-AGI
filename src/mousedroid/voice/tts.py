"""Text-to-speech wrapper for piper-tts.

Piper runs locally on ARM64 (Jetson) with no internet required.
Synthesis is offloaded to a thread to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from mousedroid.hardware.audio.constants import INT16_MAX_F
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import VoiceConfig

_log = get_logger(__name__)


class _PiperVoiceLike(Protocol):
    """Minimal interface required from a loaded piper voice.

    ``synthesize_wav`` is intentionally permissive: across piper-tts versions it
    is either ``synthesize_wav(text) -> bytes`` (0.0.7..0.x) or
    ``synthesize_wav(text, wav_file, ...)`` (piper 1.x, writes into a
    ``wave.Wave_write``). The concrete arity is resolved at load time.
    """

    def synthesize_wav(self, text: str, *args: Any, **kwargs: Any) -> Any: ...

    def synthesize(self, text: str, *args: Any, **kwargs: Any) -> Any: ...


def _synthesize_wav_needs_file(voice: object) -> bool:
    """Return True if ``synthesize_wav`` requires a ``wav_file`` argument.

    piper-tts 1.x exposes ``synthesize_wav(text, wav_file, ...)`` which writes a
    complete WAV into a caller-provided ``wave.Wave_write`` object, whereas
    0.0.7..0.x exposed ``synthesize_wav(text) -> bytes``. The signature is
    inspected once at load so the hot path never re-introspects. Falls back to
    ``False`` (the bytes API) when the signature can't be read (e.g. a C
    extension or a ``MagicMock``).

    Args:
        voice: A loaded piper voice object.

    Returns:
        ``True`` for the piper-1.x ``(text, wav_file)`` form, else ``False``.
    """
    import inspect

    synth_wav = getattr(voice, "synthesize_wav", None)
    if not callable(synth_wav):
        return False
    try:
        positional = [
            p
            for p in inspect.signature(synth_wav).parameters.values()
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        return False
    # ``self`` is already excluded for a bound method, so positional[0] is
    # ``text``; a required positional beyond it means the wav-file-writing API.
    return any(p.default is inspect.Parameter.empty for p in positional[1:])


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
        # piper 1.x ``synthesize_wav(text, wav_file)`` vs ``synthesize_wav(text)``
        # — resolved in ``start()`` and cached off the hot path.
        self._wav_needs_file: bool = False
        self._consecutive_failures: int = 0
        _log.info(
            "piper_tts_init",
            model_path=cfg.tts_model_path,
            sample_rate=cfg.tts_sample_rate,
            output_volume=cfg.output_volume,
        )

    @property
    def _api_label(self) -> str:
        """Human-readable label for the resolved synthesis API (for logs)."""
        if not self._use_wav_api:
            return "synthesize"
        return "synthesize_wav(text,wav_file)" if self._wav_needs_file else "synthesize_wav(text)"

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
                if self._use_wav_api:
                    self._wav_needs_file = _synthesize_wav_needs_file(self._voice)
                _log.info(
                    "piper_tts_model_loaded",
                    path=resolved_path,
                    api=self._api_label,
                )
                _log.info(
                    "voice_tts_api_selected",
                    api=self._api_label,
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
        """Synthesise using the ``synthesize_wav`` API (both arities).

        Handles both piper shapes resolved at load:
        - bytes API: ``synthesize_wav(text) -> bytes``.
        - piper 1.x: ``synthesize_wav(text, wav_file)`` writes a complete WAV
          (it sets the WAV format itself) into a caller-provided wave writer.

        Args:
            text: Text to synthesise.

        Returns:
            Raw WAV file bytes.
        """
        assert self._voice is not None  # guarded by caller
        if not self._wav_needs_file:
            return cast("bytes", self._voice.synthesize_wav(text))
        import io
        import wave

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)
        return wav_buffer.getvalue()

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
                api=self._api_label,
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
