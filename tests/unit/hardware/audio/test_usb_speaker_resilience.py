"""Unit tests for UsbSpeaker retry-on-init and fail-loud write behavior.

PyAudio is mocked via ``sys.modules`` patching so no real audio hardware is
needed. Each test constructs a minimal fake pyaudio module with configurable
``open()`` behavior so that we can exercise the retry loop and OSError path
without requiring ALSA or a USB device.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import structlog.testing

from mousedroid.config.schema import SpeakerConfig, VoiceConfig
from mousedroid.voice.exceptions import SpeakerUnavailableError


def _make_fake_pyaudio(pa_instance: MagicMock) -> ModuleType:
    """Build a minimal fake pyaudio module wrapping *pa_instance*."""
    m = ModuleType("pyaudio")
    m.PyAudio = MagicMock(return_value=pa_instance)  # type: ignore[attr-defined]
    m.paFloat32 = 1  # type: ignore[attr-defined]
    m.paInt16 = 8  # type: ignore[attr-defined]
    return m


def _speaker_cfg(**kw: object) -> SpeakerConfig:
    """Return a SpeakerConfig with optional overrides."""
    return SpeakerConfig(**kw)  # type: ignore[arg-type]


class TestUsbSpeakerInit:
    """Retry-on-init tests."""

    async def test_speaker_initializes_when_device_present(self) -> None:
        """No exception when pa.open() succeeds on first attempt."""
        mock_stream = MagicMock()
        mock_pa = MagicMock()
        mock_pa.open.return_value = mock_stream
        mock_pa.get_device_count.return_value = 0  # skip device search
        cfg = _speaker_cfg(reconnect_max_attempts=3, device_index=0)

        fake = _make_fake_pyaudio(mock_pa)
        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "pyaudio", fake)
            from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

            speaker = UsbSpeaker(cfg)
            with structlog.testing.capture_logs():
                await speaker.start()

        assert speaker._stream is not None
        mock_pa.open.assert_called_once()

    async def test_speaker_raises_unavailable_after_retries_exhausted(self) -> None:
        """SpeakerUnavailableError raised after reconnect_max_attempts failures."""
        mock_pa = MagicMock()
        mock_pa.open.side_effect = OSError("no device")
        mock_pa.get_device_count.return_value = 0
        cfg = _speaker_cfg(
            reconnect_max_attempts=3,
            reconnect_backoff_initial_s=0.001,
            reconnect_backoff_max_s=0.005,
            device_index=0,
        )

        fake = _make_fake_pyaudio(mock_pa)
        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "pyaudio", fake)
            from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

            speaker = UsbSpeaker(cfg)
            with (
                structlog.testing.capture_logs(),
                pytest.raises(SpeakerUnavailableError, match="3 attempts"),
            ):
                await speaker.start()

        assert mock_pa.open.call_count == 3

    async def test_speaker_raises_unavailable_when_pyaudio_missing(self) -> None:
        """SpeakerUnavailableError raised immediately when pyaudio is not installed."""
        cfg = _speaker_cfg(device_index=0)

        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "pyaudio", None)  # type: ignore[arg-type]
            from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

            speaker = UsbSpeaker(cfg)
            with (
                structlog.testing.capture_logs(),
                pytest.raises(SpeakerUnavailableError, match="pyaudio not installed"),
            ):
                await speaker.start()


class TestUsbSpeakerWrite:
    """Fail-loud write tests."""

    async def test_write_chunk_raises_unavailable_on_ioerror(self) -> None:
        """SpeakerUnavailableError propagated when _write_raw raises OSError."""
        mock_stream = MagicMock()
        mock_stream.write.side_effect = OSError("device disconnected")
        mock_stream.get_write_available.return_value = 9999
        mock_pa = MagicMock()
        mock_pa.open.return_value = mock_stream
        mock_pa.get_device_count.return_value = 0
        cfg = _speaker_cfg(device_index=0)

        fake = _make_fake_pyaudio(mock_pa)
        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "pyaudio", fake)
            from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

            speaker = UsbSpeaker(cfg)
            with structlog.testing.capture_logs() as logs:
                await speaker.start()
                with pytest.raises(SpeakerUnavailableError):
                    await speaker.write_chunk(np.zeros(1024, dtype=np.float32))

        errors = [e for e in logs if e.get("event") == "usb_speaker_write_error"]
        assert len(errors) == 1


class TestRockyVoiceDegradation:
    """RockyVoiceEngine downgrades to MockSpeaker when UsbSpeaker raises."""

    async def test_rocky_voice_downgrades_to_mock_on_unavailable(self) -> None:
        """start() replaces speaker with MockSpeaker and logs voice_speaker_degraded."""
        from mousedroid.hardware.audio.mock_speaker import MockSpeaker
        from mousedroid.voice.rocky import RockyVoiceEngine

        cfg = VoiceConfig()
        speaker_cfg = SpeakerConfig()

        bad_speaker = MagicMock()
        bad_speaker.start = AsyncMock(side_effect=SpeakerUnavailableError("no device"))
        bad_speaker._cfg = speaker_cfg

        mock_tts = MagicMock()
        mock_tts.start = MagicMock()

        engine = RockyVoiceEngine(cfg, bad_speaker, mock_tts)  # type: ignore[arg-type]

        with structlog.testing.capture_logs() as logs:
            await engine.start()

        assert isinstance(engine._speaker, MockSpeaker)
        degraded = [e for e in logs if e.get("event") == "voice_speaker_degraded"]
        assert len(degraded) == 1
        assert degraded[0].get("log_level") == "warning"

        await engine.stop()
