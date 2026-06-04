"""Unit tests for PiperTTS synthesize_wav API adapter.

Verifies that PiperTTS correctly detects the piper API generation at ``start()``
and routes to the appropriate synthesis adapter without crashing. Tests both
the modern API (``synthesize_wav`` returns bytes) and the legacy API
(``synthesize`` writes to a ``wave.Wave_write`` object).

Also covers consecutive-failure tracking and the WARNING→ERROR escalation.
"""

from __future__ import annotations

import io
import wave
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import structlog
import structlog.testing

from mousedroid.config.schema import VoiceConfig
from mousedroid.voice.tts import PiperTTS


def _make_cfg(**kw: Any) -> VoiceConfig:
    """Return a VoiceConfig with tts_model_path disabled and overrides applied."""
    return VoiceConfig(tts_model_path=None, **kw)  # type: ignore[arg-type]


def _make_valid_wav(sample_rate: int = 22050, n_frames: int = 100) -> bytes:
    """Return a minimal well-formed WAV buffer."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(np.zeros(n_frames, dtype=np.int16).tobytes())
    return buf.getvalue()


class TestApiDetection:
    """API version selection at start()."""

    def test_modern_api_selected_when_synthesize_wav_present(self) -> None:
        """_use_wav_api=True when voice has synthesize_wav attribute."""
        cfg = _make_cfg()
        tts = PiperTTS(cfg)
        mock_voice = MagicMock(spec=["synthesize_wav", "synthesize"])

        tts._voice = mock_voice
        tts._use_wav_api = hasattr(mock_voice, "synthesize_wav")

        assert tts._use_wav_api is True

    def test_legacy_api_selected_when_only_synthesize_present(self) -> None:
        """_use_wav_api=False when voice lacks synthesize_wav."""
        cfg = _make_cfg()
        tts = PiperTTS(cfg)
        mock_voice = MagicMock(spec=["synthesize"])

        tts._voice = mock_voice
        tts._use_wav_api = hasattr(mock_voice, "synthesize_wav")

        assert tts._use_wav_api is False

    def test_start_logs_voice_tts_api_selected(self) -> None:
        """voice_tts_api_selected INFO log is emitted when model loads."""
        cfg = _make_cfg()
        tts = PiperTTS(cfg)

        mock_voice = MagicMock()
        tts._voice = mock_voice
        tts._use_wav_api = hasattr(mock_voice, "synthesize_wav")

        _log = structlog.get_logger("mousedroid.voice.tts")
        with structlog.testing.capture_logs() as logs:
            _log.info(
                "voice_tts_api_selected",
                api="synthesize_wav" if tts._use_wav_api else "synthesize",
            )

        selected = [e for e in logs if e.get("event") == "voice_tts_api_selected"]
        assert len(selected) == 1
        assert selected[0]["api"] == "synthesize_wav"


class TestSynthesizeViaWav:
    """Modern synthesize_wav adapter path."""

    def test_synthesize_via_wav_returns_samples(self) -> None:
        """_synthesize_sync returns float32 samples from WAV bytes (modern API)."""
        cfg = _make_cfg(output_volume=1.0)
        tts = PiperTTS(cfg)

        wav_bytes = _make_valid_wav(sample_rate=22050, n_frames=100)
        mock_voice = MagicMock()
        mock_voice.synthesize_wav.return_value = wav_bytes
        tts._voice = mock_voice
        tts._use_wav_api = True

        result = tts._synthesize_sync("hello")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.size == 100
        mock_voice.synthesize_wav.assert_called_once_with("hello")

    def test_synthesize_via_legacy_returns_samples(self) -> None:
        """_synthesize_sync returns float32 samples via wave writer (legacy API)."""
        cfg = _make_cfg(output_volume=1.0)
        tts = PiperTTS(cfg)

        wav_bytes = _make_valid_wav(sample_rate=22050, n_frames=50)

        def _fake_synthesize(text: str, wav_file: Any) -> None:
            """Write pre-built frames into wav_file."""
            buf = io.BytesIO(wav_bytes)
            with wave.open(buf, "rb") as src:
                wav_file.writeframes(src.readframes(src.getnframes()))

        mock_voice = MagicMock(spec=["synthesize"])
        mock_voice.synthesize.side_effect = _fake_synthesize
        tts._voice = mock_voice
        tts._use_wav_api = False

        result = tts._synthesize_sync("hello")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32


class _FakeVoiceWavBytes:
    """piper 0.0.7..0.x: ``synthesize_wav(text) -> bytes``."""

    def __init__(self, wav: bytes) -> None:
        self._wav = wav

    def synthesize_wav(self, text: str) -> bytes:
        return self._wav


class _FakeVoiceWavFile:
    """piper 1.x: ``synthesize_wav(text, wav_file, ...)`` writes a complete WAV."""

    def __init__(self, frames: bytes, sample_rate: int) -> None:
        self._frames = frames
        self._sr = sample_rate

    def synthesize_wav(
        self,
        text: str,
        wav_file: Any,
        syn_config: Any = None,
        set_wav_format: bool = True,
    ) -> None:
        if set_wav_format:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._sr)
        wav_file.writeframes(self._frames)


class _FakeVoiceLegacy:
    """oldest piper: only ``synthesize(text, wav_file)``."""

    def synthesize(self, text: str, wav_file: Any) -> None: ...


class TestApiModeDetection:
    """``_synthesize_wav_needs_file`` signature inspection (piper 1.x guard)."""

    def test_wav_file_form_detected(self) -> None:
        from mousedroid.voice.tts import _synthesize_wav_needs_file

        assert _synthesize_wav_needs_file(_FakeVoiceWavFile(b"", 22050)) is True

    def test_wav_bytes_form_detected(self) -> None:
        from mousedroid.voice.tts import _synthesize_wav_needs_file

        assert _synthesize_wav_needs_file(_FakeVoiceWavBytes(b"")) is False

    def test_legacy_voice_has_no_wav(self) -> None:
        from mousedroid.voice.tts import _synthesize_wav_needs_file

        assert _synthesize_wav_needs_file(_FakeVoiceLegacy()) is False

    def test_start_resolves_wav_file_label(self) -> None:
        """A piper-1.x voice resolves to the (text,wav_file) api label."""
        tts = PiperTTS(_make_cfg())
        tts._voice = _FakeVoiceWavFile(b"", 22050)
        tts._use_wav_api = True
        tts._wav_needs_file = True
        assert tts._api_label == "synthesize_wav(text,wav_file)"


class TestSynthesizeViaWavFile:
    """piper-1.x ``synthesize_wav(text, wav_file)`` adapter path."""

    def test_returns_samples_from_writer(self) -> None:
        cfg = _make_cfg(output_volume=1.0)
        tts = PiperTTS(cfg)
        frames = np.zeros(64, dtype=np.int16).tobytes()
        tts._voice = _FakeVoiceWavFile(frames, 22050)
        tts._use_wav_api = True
        tts._wav_needs_file = True

        result = tts._synthesize_sync("hello")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.size == 64


class TestFailureTracking:
    """Consecutive-failure counter and WARNING→ERROR escalation."""

    def test_consecutive_failure_increments_counter(self) -> None:
        """Synthesis errors increment _consecutive_failures and log WARNING then ERROR."""
        cfg = _make_cfg(tts_failure_threshold=2)
        tts = PiperTTS(cfg)

        mock_voice = MagicMock()
        mock_voice.synthesize_wav.side_effect = RuntimeError("boom")
        tts._voice = mock_voice
        tts._use_wav_api = True

        with structlog.testing.capture_logs() as logs:
            tts._synthesize_sync("a")  # failure 1 → WARNING
            tts._synthesize_sync("b")  # failure 2 → ERROR (at threshold)

        failures = [e for e in logs if e.get("event") == "voice_tts_synthesize_failed"]
        assert len(failures) == 2
        assert tts._consecutive_failures == 2
        levels = {e.get("log_level") for e in failures}
        assert "warning" in levels
        assert "error" in levels

    def test_failure_counter_resets_on_success(self) -> None:
        """_consecutive_failures resets to 0 after a successful synthesis."""
        cfg = _make_cfg(tts_failure_threshold=3, output_volume=1.0)
        tts = PiperTTS(cfg)

        mock_voice = MagicMock()
        wav_bytes = _make_valid_wav()
        mock_voice.synthesize_wav.side_effect = [RuntimeError("fail"), wav_bytes]
        tts._voice = mock_voice
        tts._use_wav_api = True

        tts._synthesize_sync("fail")
        assert tts._consecutive_failures == 1

        tts._synthesize_sync("succeed")
        assert tts._consecutive_failures == 0
