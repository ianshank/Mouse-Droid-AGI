"""Unit tests for MockSpeaker — mirrors test_mock_microphone.py."""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.config.schema import SpeakerConfig
from mousedroid.hardware.audio.mock_speaker import MockSpeaker


def _default_cfg() -> SpeakerConfig:
    return SpeakerConfig()


@pytest.mark.asyncio
async def test_write_chunk_captures_data() -> None:
    """Written chunks are captured for test assertions."""
    speaker = MockSpeaker(_default_cfg())
    chunk = np.ones(1024, dtype=np.float32)
    await speaker.write_chunk(chunk)
    written = speaker.get_written_chunks()
    assert len(written) == 1
    np.testing.assert_array_equal(written[0], chunk)


@pytest.mark.asyncio
async def test_write_chunk_copies_data() -> None:
    """Captured chunks are copies, not references."""
    speaker = MockSpeaker(_default_cfg())
    chunk = np.ones(1024, dtype=np.float32)
    await speaker.write_chunk(chunk)
    chunk[:] = 0.0  # Mutate original
    written = speaker.get_written_chunks()
    assert written[0].sum() == 1024.0  # Copy unchanged


@pytest.mark.asyncio
async def test_write_multiple_chunks() -> None:
    """Multiple chunks are captured in order."""
    speaker = MockSpeaker(_default_cfg())
    for i in range(3):
        await speaker.write_chunk(np.full(1024, float(i), dtype=np.float32))
    written = speaker.get_written_chunks()
    assert len(written) == 3
    assert written[0][0] == 0.0
    assert written[1][0] == 1.0
    assert written[2][0] == 2.0


@pytest.mark.asyncio
async def test_clear() -> None:
    """Clear removes all captured chunks."""
    speaker = MockSpeaker(_default_cfg())
    await speaker.write_chunk(np.ones(1024, dtype=np.float32))
    speaker.clear()
    assert len(speaker.get_written_chunks()) == 0


@pytest.mark.asyncio
async def test_lifecycle() -> None:
    """Start/stop track started state."""
    speaker = MockSpeaker(_default_cfg())
    assert not speaker.started
    await speaker.start()
    assert speaker.started
    await speaker.stop()
    assert not speaker.started


def test_properties_from_config() -> None:
    """Properties reflect config values."""
    cfg = SpeakerConfig(sample_rate=44100, channels=2, chunk_size=512)
    speaker = MockSpeaker(cfg)
    assert speaker.sample_rate == 44100
    assert speaker.channels == 2
    assert speaker.chunk_size == 512


def test_default_properties() -> None:
    """Default config produces expected values."""
    speaker = MockSpeaker(_default_cfg())
    assert speaker.sample_rate == 22050
    assert speaker.channels == 1
    assert speaker.chunk_size == 1024
