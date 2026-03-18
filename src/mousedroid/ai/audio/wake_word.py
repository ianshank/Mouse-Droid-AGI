"""Wake word detection using OpenWakeWord.

Implements ``WakeWordProtocol`` for lightweight always-on keyword
spotting. Runs on CPU with minimal memory overhead (~20 MB).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import AudioAIConfig

_log = get_logger(__name__)

_oww: Any
try:
    import openwakeword as _oww  # type: ignore[import-untyped]
    from openwakeword.model import Model as _OWWModel  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _oww = None
    _OWWModel = None


class OpenWakeWordDetector:
    """OpenWakeWord detector implementing ``WakeWordProtocol``.

    Uses a pre-trained model for the configured wake phrase (default:
    ``"hey_jarvis"``). Requires 16 kHz mono INT16 audio chunks.

    CPU-only — no GPU memory overhead.
    """

    def __init__(self, cfg: AudioAIConfig) -> None:
        self._cfg = cfg
        self._model: Any = None
        self._threshold: float = cfg.wake_word_threshold

    async def start(self) -> None:
        """Load wake word model."""
        if _OWWModel is None:
            msg = "openwakeword is not installed — install mousedroid[ai-audio]"
            raise RuntimeError(msg)
        await asyncio.to_thread(self._load_model)
        _log.info(
            "wake_word_detector_started",
            model=self._cfg.wake_word_model,
            threshold=self._threshold,
        )

    def _load_model(self) -> None:
        """Load OpenWakeWord model (blocking)."""
        # Download only the configured wake-word model
        _oww.utils.download_models(models=[self._cfg.wake_word_model])
        self._model = _OWWModel(
            wakeword_models=[self._cfg.wake_word_model],
            inference_framework="onnx",
        )

    async def stop(self) -> None:
        """Release model resources."""
        self._model = None
        _log.info("wake_word_detector_stopped")

    async def detect(self, audio_chunk: NDArray[np.int16]) -> bool:
        """Check if the wake word is present in an audio chunk.

        Designed to be called continuously with small audio chunks
        (e.g. 1280 samples = 80 ms at 16 kHz).

        Args:
            audio_chunk: INT16 audio samples, shape ``(N,)``, 16 kHz.

        Returns:
            True if wake word detected above threshold.
        """
        if self._model is None:
            return False

        return await asyncio.to_thread(self._infer, audio_chunk)

    def _infer(self, audio_chunk: NDArray[np.int16]) -> bool:
        """Run wake word inference (blocking)."""
        prediction = self._model.predict(audio_chunk)
        # Check all model scores against threshold
        for model_name, scores in prediction.items():
            if isinstance(scores, dict):
                for score in scores.values():
                    if score >= self._threshold:
                        _log.info("wake_word_detected", model=model_name, score=score)
                        return True
            elif isinstance(scores, (int, float)) and scores >= self._threshold:
                _log.info("wake_word_detected", model=model_name, score=scores)
                return True
        return False
