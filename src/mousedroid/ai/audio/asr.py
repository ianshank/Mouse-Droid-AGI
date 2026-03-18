"""Whisper-based automatic speech recognition for Jetson.

Implements ``ASRProtocol`` using ``faster-whisper`` (CTranslate2 backend)
for efficient on-device speech-to-text with INT8 quantisation.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.ai.audio.protocols import Transcription
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import AudioAIConfig

_log = get_logger(__name__)

_fw: Any
try:
    from faster_whisper import WhisperModel as _WhisperModel  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _WhisperModel = None


class WhisperASR:
    """Faster-Whisper ASR implementing ``ASRProtocol``.

    Uses CTranslate2 for INT8 inference on Jetson. The ``tiny.en`` model
    uses ~75 MB of memory and provides acceptable accuracy for short
    voice commands.

    All blocking inference is delegated to ``asyncio.to_thread()``.
    """

    def __init__(self, cfg: AudioAIConfig) -> None:
        self._cfg = cfg
        self._model: Any = None

    async def start(self) -> None:
        """Load Whisper model."""
        if _WhisperModel is None:
            msg = "faster-whisper is not installed — install mousedroid[ai-audio]"
            raise RuntimeError(msg)
        await asyncio.to_thread(self._load_model)
        _log.info(
            "whisper_asr_started",
            model=self._cfg.asr_model,
            compute_type=self._cfg.asr_compute_type,
        )

    def _load_model(self) -> None:
        """Load CTranslate2 Whisper model (blocking)."""
        # faster-whisper uses CTranslate2 which can run on CPU with int8
        # or GPU with float16
        try:
            self._model = _WhisperModel(
                self._cfg.asr_model,
                device="cuda",
                compute_type=self._cfg.asr_compute_type,
            )
        except Exception:
            _log.warning("whisper_cuda_failed_falling_back_to_cpu", exc_info=True)
            self._model = _WhisperModel(
                self._cfg.asr_model,
                device="cpu",
                compute_type="int8",
            )

    async def stop(self) -> None:
        """Release model resources."""
        self._model = None
        _log.info("whisper_asr_stopped")

    async def transcribe(self, audio: NDArray[np.float32], sample_rate: int) -> Transcription | None:
        """Transcribe an audio segment.

        Args:
            audio: Float32 audio samples, shape ``(N,)``, normalised to [-1, 1].
            sample_rate: Audio sample rate (typically 16000).

        Returns:
            Transcription result, or None if no speech detected.
        """
        if self._model is None:
            return None

        if len(audio) < sample_rate * 0.3:
            # Less than 300ms of audio — too short to transcribe
            return None

        return await asyncio.to_thread(self._infer, audio, sample_rate)

    def _infer(self, audio: NDArray[np.float32], sample_rate: int) -> Transcription | None:
        """Run Whisper transcription (blocking)."""
        # faster-whisper expects 16kHz mono float32
        target_rate = self._cfg.asr_sample_rate_hz
        if sample_rate != target_rate:
            # Simple resampling — in production use scipy.signal.resample
            ratio = target_rate / sample_rate
            indices = np.arange(0, len(audio), 1 / ratio).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]

        segments, info = self._model.transcribe(
            audio,
            beam_size=self._cfg.asr_beam_size,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        texts = []
        total_prob = 0.0
        n_segments = 0
        for segment in segments:
            text = segment.text.strip()
            if text:
                texts.append(text)
                total_prob += segment.avg_log_prob
                n_segments += 1

        if not texts:
            return None

        full_text = " ".join(texts)
        avg_confidence = np.exp(total_prob / n_segments) if n_segments > 0 else 0.0
        duration = len(audio) / self._cfg.asr_sample_rate_hz

        return Transcription(
            text=full_text,
            language="en",
            confidence=float(avg_confidence),
            duration_s=duration,
        )
