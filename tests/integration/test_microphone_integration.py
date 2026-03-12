from __future__ import annotations

from mousedroid.config.schema import MicrophoneConfig, Settings
from mousedroid.factory import build_microphone
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
