from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.config.schema import MicrophoneConfig
from mousedroid.hardware.audio.mock_microphone import MockMicrophone


@given(
    sample_rate=st.integers(min_value=1, max_value=192000),
    channels=st.integers(min_value=1, max_value=2),
    chunk_size=st.integers(min_value=1, max_value=8192),
)
@settings(max_examples=50)
async def test_mock_microphone_chunk_shape(sample_rate, channels, chunk_size):
    cfg = MicrophoneConfig(
        sample_rate=sample_rate,
        channels=channels,
        chunk_size=chunk_size,
    )
    mic = MockMicrophone(cfg)
    chunk = await mic.read_chunk()
    assert chunk.shape == (chunk_size * channels,)
    assert chunk.dtype == np.float32


@given(
    sample_rate=st.integers(min_value=1, max_value=192000),
)
@settings(max_examples=20)
def test_sample_rate_preserved(sample_rate):
    cfg = MicrophoneConfig(sample_rate=sample_rate)
    mic = MockMicrophone(cfg)
    assert mic.sample_rate == sample_rate


@given(
    channels=st.integers(min_value=1, max_value=2),
)
def test_channels_preserved(channels):
    cfg = MicrophoneConfig(channels=channels)
    mic = MockMicrophone(cfg)
    assert mic.channels == channels


@given(
    chunk_size=st.integers(min_value=1, max_value=8192),
)
@settings(max_examples=20)
def test_chunk_size_preserved(chunk_size):
    cfg = MicrophoneConfig(chunk_size=chunk_size)
    mic = MockMicrophone(cfg)
    assert mic.chunk_size == chunk_size


async def test_set_chunk_overrides_random():
    cfg = MicrophoneConfig(chunk_size=64)
    mic = MockMicrophone(cfg)
    custom = np.ones(64, dtype=np.float32) * 0.42
    mic.set_chunk(custom)
    result = await mic.read_chunk()
    np.testing.assert_array_equal(result, custom)
