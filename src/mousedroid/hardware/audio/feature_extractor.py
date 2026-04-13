"""Audio feature extraction — mel spectrogram for world-model input.

Converts raw PCM audio chunks into fixed-size feature vectors suitable
for the multimodal encoder.  Uses a simple mel filter bank computed with
numpy (no librosa dependency required at runtime).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import MicrophoneConfig

_log = get_logger(__name__)


def _mel_frequency(mel: float) -> float:
    """Convert mel-scale value to frequency in Hz.

    Args:
        mel: Mel-scale value.

    Returns:
        Frequency in Hz.
    """
    return float(700.0 * (10.0 ** (mel / 2595.0) - 1.0))


def _hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to mel scale.

    Args:
        hz: Frequency in Hz.

    Returns:
        Mel-scale value.
    """
    return float(2595.0 * np.log10(1.0 + hz / 700.0))


def _build_mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
) -> NDArray[np.float64]:
    """Build a mel-scale triangular filterbank matrix.

    Args:
        n_mels: Number of mel bands.
        n_fft: FFT window size.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Filterbank matrix, shape ``(n_mels, n_fft // 2 + 1)``.
    """
    n_freqs = n_fft // 2 + 1
    low_mel = _hz_to_mel(0.0)
    high_mel = _hz_to_mel(float(sample_rate) / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = np.array([_mel_frequency(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for j in range(left, center):
            if center > left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


class AudioFeatureExtractor:
    """Extract mel-spectrogram features from raw audio chunks.

    Produces a fixed-size float32 feature vector from a raw PCM audio
    chunk.  The output dimension equals ``n_mels * n_frames`` where
    ``n_frames`` depends on ``chunk_size``, ``n_fft``, and ``hop_length``.

    Args:
        cfg: Microphone configuration with mel parameters.
    """

    def __init__(self, cfg: MicrophoneConfig) -> None:
        self._cfg = cfg
        self._n_mels = cfg.n_mels
        self._n_fft = cfg.n_fft
        self._hop_length = cfg.hop_length
        self._sample_rate = cfg.sample_rate
        self._channels = cfg.channels

        self._filterbank = _build_mel_filterbank(
            n_mels=self._n_mels,
            n_fft=self._n_fft,
            sample_rate=self._sample_rate,
        )

        # Pre-compute expected output dimension.
        mono_chunk = cfg.chunk_size
        n_frames = max(1, 1 + (mono_chunk - self._n_fft) // self._hop_length)
        self._feature_dim = self._n_mels * n_frames

        _log.info(
            "audio_feature_extractor_init",
            n_mels=self._n_mels,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            feature_dim=self._feature_dim,
        )

    @property
    def feature_dim(self) -> int:
        """Output feature vector dimension."""
        return self._feature_dim

    def extract(self, audio_chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        """Extract mel-spectrogram features from a raw audio chunk.

        If the input is multi-channel, channels are averaged to mono first.

        Args:
            audio_chunk: Raw audio samples, shape ``(chunk_size * channels,)``.

        Returns:
            Feature vector, shape ``(feature_dim,)``, log-scaled and normalised.
        """
        # Mix down to mono if stereo.
        if self._channels > 1:
            mono = audio_chunk.reshape(-1, self._channels).mean(axis=1)
        else:
            mono = audio_chunk

        # STFT via overlapping windowed FFT.
        n_fft = self._n_fft
        hop = self._hop_length
        n_samples = len(mono)

        if n_samples < n_fft:
            # Pad if chunk is shorter than FFT window.
            mono = np.pad(mono, (0, n_fft - n_samples), mode="constant")
            n_samples = n_fft

        n_frames = max(1, 1 + (n_samples - n_fft) // hop)
        window = np.hanning(n_fft).astype(np.float32)

        power_spec_frames = []
        for i in range(n_frames):
            start = i * hop
            frame = mono[start : start + n_fft] * window
            spectrum = np.fft.rfft(frame.astype(np.float64))
            power = np.minimum(np.abs(spectrum) ** 2, 1e20)
            power_spec_frames.append(power)

        # Stack: (n_frames, n_fft//2+1)
        power_spectrogram = np.array(power_spec_frames, dtype=np.float64)

        # Apply mel filterbank: (n_mels, n_fft//2+1) @ (n_fft//2+1, n_frames) -> (n_mels, n_frames)
        # Suppress expected numerical warnings from sparse filterbank * near-zero values.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            mel_spec = self._filterbank @ power_spectrogram.T
        mel_spec = np.nan_to_num(mel_spec, nan=0.0, posinf=1e20, neginf=0.0)

        # Log scale with floor to avoid log(0).
        mel_spec = np.log(np.maximum(mel_spec, 1e-10))

        # Flatten to feature vector.
        features = mel_spec.T.flatten().astype(np.float32)

        # Truncate or pad to match expected dimension.
        if len(features) > self._feature_dim:
            features = features[: self._feature_dim]
        elif len(features) < self._feature_dim:
            features = np.pad(
                features,
                (0, self._feature_dim - len(features)),
                mode="constant",
            )

        # L2-normalise for stable input to the encoder.
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return cast(NDArray[np.float32], features)
