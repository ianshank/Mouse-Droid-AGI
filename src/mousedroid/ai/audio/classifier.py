"""YAMNet environmental sound classifier.

Implements ``SoundClassifierProtocol`` using a YAMNet ONNX model
for classifying environmental sounds (e.g. crash, alarm, voice).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.ai.audio.protocols import SoundEvent
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import AudioAIConfig

_log = get_logger(__name__)

_ort: Any
try:
    import onnxruntime as _ort  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _ort = None

# Top-level YAMNet class categories for robotics
_SAFETY_CATEGORIES = {
    "alarm", "siren", "crash", "explosion", "glass_breaking",
    "scream", "shout", "crying", "horn", "buzzer",
}
_VOICE_CATEGORIES = {"speech", "singing", "conversation"}
_ENVIRONMENT_CATEGORIES = {
    "music", "silence", "wind", "rain", "thunder",
    "dog", "cat", "bird", "engine", "vehicle",
}

# YAMNet class names (subset — full list has 521 classes)
# We load from the model metadata or use indices
_YAMNET_CLASSES: list[str] = []


class YAMNetClassifier:
    """YAMNet sound classifier implementing ``SoundClassifierProtocol``.

    Uses ONNX Runtime for efficient inference. Classifies audio into
    safety-relevant, voice, and environmental sound categories.

    CPU-only — minimal memory overhead (~20 MB).
    """

    def __init__(self, cfg: AudioAIConfig) -> None:
        self._cfg = cfg
        self._session: Any = None
        self._classes: list[str] = []
        self._last_inference_t: float = 0.0
        self._min_interval: float = 1.0 / cfg.classifier_max_hz

    async def start(self) -> None:
        """Load YAMNet ONNX model."""
        if _ort is None:
            msg = "onnxruntime is not installed — install mousedroid[ai-audio]"
            raise RuntimeError(msg)
        await asyncio.to_thread(self._load_model)
        _log.info("yamnet_classifier_started", model=self._cfg.sound_classifier_model)

    def _load_model(self) -> None:
        """Load ONNX session (blocking)."""
        model_path = self._cfg.sound_classifier_model
        cache_dir = Path(self._cfg.sound_classifier_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        full_path = cache_dir / model_path
        if not full_path.exists():
            # Try loading from default model zoo path
            full_path = Path(model_path)

        if not full_path.exists():
            _log.warning("yamnet_model_not_found", path=str(full_path))
            return

        sess_opts = _ort.SessionOptions()
        sess_opts.inter_op_num_threads = 2
        sess_opts.intra_op_num_threads = 2
        self._session = _ort.InferenceSession(
            str(full_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        _log.info("yamnet_model_loaded", path=str(full_path))

    async def stop(self) -> None:
        """Release model resources."""
        self._session = None
        _log.info("yamnet_classifier_stopped")

    async def classify(self, audio: NDArray[np.float32], sample_rate: int) -> list[SoundEvent]:
        """Classify environmental sounds in an audio segment.

        Rate-limited to ``classifier_max_hz``.

        Args:
            audio: Float32 audio, shape ``(N,)``, normalised to [-1, 1].
            sample_rate: Audio sample rate (typically 16000).

        Returns:
            Top-K sound events sorted by confidence.
        """
        now = time.monotonic()
        if now - self._last_inference_t < self._min_interval:
            return []
        self._last_inference_t = now

        if self._session is None:
            return []

        return await asyncio.to_thread(self._infer, audio, sample_rate)

    def _infer(self, audio: NDArray[np.float32], sample_rate: int) -> list[SoundEvent]:
        """Run YAMNet classification (blocking)."""
        # YAMNet expects mono at the configured sample rate
        target_rate = self._cfg.classifier_sample_rate_hz
        if sample_rate != target_rate:
            ratio = target_rate / sample_rate
            indices = np.arange(0, len(audio), 1 / ratio).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]

        # Pad or trim to configured window
        target_len = int(self._cfg.classifier_window_ms * target_rate / 1000)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        # Run inference
        input_name = self._session.get_inputs()[0].name
        scores = self._session.run(None, {input_name: audio.reshape(1, -1)})[0]

        # Average over time frames if multiple
        if scores.ndim > 1:
            mean_scores = scores.mean(axis=0)
        else:
            mean_scores = scores.flatten()

        # Get top-K predictions
        top_k = min(self._cfg.classifier_top_k, len(mean_scores))
        top_indices = np.argsort(mean_scores)[::-1][:top_k]

        events: list[SoundEvent] = []
        for idx in top_indices:
            conf = float(mean_scores[idx])
            if conf < self._cfg.sound_classifier_confidence:  # Skip low confidence
                continue
            label = str(idx) if idx >= len(self._classes) else self._classes[idx]
            category = self._categorize(label)
            events.append(SoundEvent(label=label, confidence=conf, category=category))

        return events

    @staticmethod
    def _categorize(label: str) -> str:
        """Map a YAMNet class label to a high-level category."""
        label_lower = label.lower().replace(" ", "_")
        if label_lower in _SAFETY_CATEGORIES:
            return "safety"
        if label_lower in _VOICE_CATEGORIES:
            return "voice"
        if label_lower in _ENVIRONMENT_CATEGORIES:
            return "environment"
        return "other"
