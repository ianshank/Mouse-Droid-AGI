from __future__ import annotations

import numpy as np

from mousedroid.config.schema import MicrophoneConfig
from mousedroid.hardware.audio.mock_microphone import MockMicrophone


def _default_cfg() -> MicrophoneConfig:
    return MicrophoneConfig()


async def test_read_chunk_shape():
    mic = MockMicrophone(_default_cfg())
    chunk = await mic.read_chunk()
    assert chunk.shape == (1024,)


async def test_read_chunk_dtype():
    mic = MockMicrophone(_default_cfg())
    chunk = await mic.read_chunk()
    assert chunk.dtype == np.float32


async def test_read_chunk_stereo():
    cfg = MicrophoneConfig(channels=2)
    mic = MockMicrophone(cfg)
    chunk = await mic.read_chunk()
    assert chunk.shape == (2048,)


async def test_read_chunk_custom_size():
    cfg = MicrophoneConfig(chunk_size=512)
    mic = MockMicrophone(cfg)
    chunk = await mic.read_chunk()
    assert chunk.shape == (512,)


async def test_set_chunk():
    mic = MockMicrophone(_default_cfg())
    data = np.ones(1024, dtype=np.float32) * 0.5
    mic.set_chunk(data)
    chunk = await mic.read_chunk()
    np.testing.assert_array_equal(chunk, data)


async def test_start_stop():
    mic = MockMicrophone(_default_cfg())
    assert not mic.started
    await mic.start()
    assert mic.started
    await mic.stop()
    assert not mic.started


def test_sample_rate():
    mic = MockMicrophone(_default_cfg())
    assert mic.sample_rate == 16000


def test_channels():
    mic = MockMicrophone(_default_cfg())
    assert mic.channels == 1


def test_chunk_size():
    mic = MockMicrophone(_default_cfg())
    assert mic.chunk_size == 1024


def test_custom_sample_rate():
    cfg = MicrophoneConfig(sample_rate=44100)
    mic = MockMicrophone(cfg)
    assert mic.sample_rate == 44100


async def test_read_chunk_not_all_zero():
    mic = MockMicrophone(_default_cfg())
    chunk = await mic.read_chunk()
    # Random data should not all be zero
    assert not np.allclose(chunk, 0.0)
