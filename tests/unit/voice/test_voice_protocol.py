"""Tests for runtime protocol compliance — isinstance checks."""

from __future__ import annotations

from mousedroid.config.schema import SpeakerConfig, VoiceConfig
from mousedroid.hardware.audio.mock_speaker import MockSpeaker
from mousedroid.hardware.protocols import SpeakerProtocol
from mousedroid.voice.mock_tts import MockTTS
from mousedroid.voice.protocol import VoiceEngineProtocol
from mousedroid.voice.rocky import RockyVoiceEngine


def test_mock_speaker_implements_speaker_protocol() -> None:
    """MockSpeaker satisfies SpeakerProtocol at runtime."""
    cfg = SpeakerConfig()
    speaker = MockSpeaker(cfg)
    assert isinstance(speaker, SpeakerProtocol)


def test_usb_speaker_implements_speaker_protocol() -> None:
    """UsbSpeaker satisfies SpeakerProtocol at runtime."""
    from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

    cfg = SpeakerConfig()
    speaker = UsbSpeaker(cfg)
    assert isinstance(speaker, SpeakerProtocol)


def test_rocky_voice_engine_implements_voice_protocol() -> None:
    """RockyVoiceEngine satisfies VoiceEngineProtocol at runtime."""
    voice_cfg = VoiceConfig(enabled=True, tts_sample_rate=22050)
    speaker_cfg = SpeakerConfig()
    speaker = MockSpeaker(speaker_cfg)
    tts = MockTTS(voice_cfg)
    engine = RockyVoiceEngine(voice_cfg, speaker, tts)
    assert isinstance(engine, VoiceEngineProtocol)
