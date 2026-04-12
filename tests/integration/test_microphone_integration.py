"""Integration tests for the microphone -> feature extraction -> tensor pipeline."""

from __future__ import annotations

import numpy as np

from mousedroid.config.schema import MicrophoneConfig, Settings
from mousedroid.factory import build_microphone
from mousedroid.hardware.audio.feature_extractor import AudioFeatureExtractor
from mousedroid.hardware.audio.mock_microphone import MockMicrophone


def test_build_microphone_mock_integration():
    cfg = Settings(
        mock_hardware=True,
        microphone=MicrophoneConfig(sample_rate=22050, chunk_size=512),
    )
    mic = build_microphone(cfg)
    assert isinstance(mic, MockMicrophone)
    assert mic.sample_rate == 22050
    assert mic.chunk_size == 512


async def test_mock_microphone_lifecycle():
    cfg = Settings(
        mock_hardware=True,
        microphone=MicrophoneConfig(),
    )
    mic = build_microphone(cfg)
    assert mic is not None
    await mic.start()
    chunk = await mic.read_chunk()
    assert chunk.shape == (1024,)
    await mic.stop()


def test_build_microphone_disabled_integration():
    cfg = Settings(mock_hardware=True)
    mic = build_microphone(cfg)
    assert mic is None


def test_build_microphone_disabled_via_enabled_flag():
    """Microphone present but ``enabled=False`` returns None."""
    cfg = Settings(
        mock_hardware=True,
        microphone=MicrophoneConfig(enabled=False),
    )
    mic = build_microphone(cfg)
    assert mic is None


async def test_mic_capture_to_feature_extraction():
    """End-to-end: mock mic -> read_chunk -> feature extraction -> tensor."""
    mic_cfg = MicrophoneConfig()
    mic = MockMicrophone(mic_cfg)
    extractor = AudioFeatureExtractor(mic_cfg)

    await mic.start()
    raw_chunk = await mic.read_chunk()
    assert raw_chunk.shape == (mic_cfg.chunk_size,)

    features = extractor.extract(raw_chunk)
    assert features.dtype == np.float32
    assert features.shape == (extractor.feature_dim,)
    assert np.isfinite(features).all()

    # Verify L2 normalisation.
    norm = np.linalg.norm(features)
    assert abs(norm - 1.0) < 1e-5

    await mic.stop()


async def test_mic_set_chunk_to_features():
    """Custom audio data flows through the full pipeline."""
    mic_cfg = MicrophoneConfig(chunk_size=512)
    mic = MockMicrophone(mic_cfg)
    extractor = AudioFeatureExtractor(mic_cfg)

    # Set a known sine wave.
    t = np.linspace(0, 1, 512, endpoint=False, dtype=np.float32)
    sine = np.sin(2 * np.pi * 440 * t)
    mic.set_chunk(sine)

    await mic.start()
    raw = await mic.read_chunk()
    np.testing.assert_array_equal(raw, sine)

    features = extractor.extract(raw)
    assert features.shape == (extractor.feature_dim,)
    assert np.isfinite(features).all()
    await mic.stop()


async def test_stereo_mic_to_features():
    """Stereo microphone audio is correctly mixed down for feature extraction."""
    mic_cfg = MicrophoneConfig(channels=2)
    mic = MockMicrophone(mic_cfg)
    extractor = AudioFeatureExtractor(mic_cfg)

    await mic.start()
    raw = await mic.read_chunk()
    assert raw.shape == (mic_cfg.chunk_size * 2,)

    features = extractor.extract(raw)
    assert features.shape == (extractor.feature_dim,)
    await mic.stop()


def test_device_not_found_graceful():
    """USB mic with invalid device name still builds; failure is at start()."""
    mic_cfg = MicrophoneConfig(device_name="NonExistentDevice12345")
    cfg = Settings(
        mock_hardware=True,
        microphone=mic_cfg,
    )
    mic = build_microphone(cfg)
    # Mock microphone always succeeds — real hardware failure handled in start().
    assert mic is not None
    assert isinstance(mic, MockMicrophone)
