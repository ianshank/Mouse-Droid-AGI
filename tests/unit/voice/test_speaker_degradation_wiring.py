"""Wiring tests for the voice_speaker_degraded_total counter.

Exercises the two write sites end-to-end through the real classes:

* ``RockyVoiceEngine.start`` catching ``SpeakerUnavailableError`` and downgrading
  to a MockSpeaker → ``subsystem="rocky_fallback"``.
* ``UsbSpeaker.start`` exhausting its reconnect retries → ``subsystem="usb_speaker"``.

Both assert the ``metrics is None`` path is a no-op so offline constructions keep
working without telemetry wiring.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from mousedroid.config.schema import MetricsConfig, SpeakerConfig, VoiceConfig
from mousedroid.hardware.audio.usb_speaker import UsbSpeaker
from mousedroid.telemetry.metrics import MetricsRegistry
from mousedroid.voice.exceptions import SpeakerUnavailableError
from mousedroid.voice.mock_tts import MockTTS
from mousedroid.voice.rocky import RockyVoiceEngine


class _UnavailableSpeaker:
    """Speaker stand-in whose ``start`` always reports the hardware as gone."""

    def __init__(self) -> None:
        self._cfg = SpeakerConfig.model_validate({})

    @property
    def sample_rate(self) -> int:
        return self._cfg.sample_rate

    async def start(self) -> None:
        raise SpeakerUnavailableError("no speaker")

    async def stop(self) -> None:  # pragma: no cover - not reached in these tests
        return None


# --------------------------------------------------------------------------- #
# RockyVoiceEngine fallback path
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_rocky_fallback_increments_counter() -> None:
    cfg = VoiceConfig()
    metrics = MetricsRegistry(MetricsConfig())
    engine = RockyVoiceEngine(cfg, _UnavailableSpeaker(), MockTTS(cfg), metrics=metrics)

    await engine.start()
    try:
        out = metrics.render_prometheus()
        assert 'mousedroid_voice_speaker_degraded_total{subsystem="rocky_fallback"} 1' in out
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_rocky_fallback_without_metrics_is_noop() -> None:
    cfg = VoiceConfig()
    engine = RockyVoiceEngine(cfg, _UnavailableSpeaker(), MockTTS(cfg))  # metrics=None

    await engine.start()
    try:
        assert engine is not None  # reached the fallback without raising
    finally:
        await engine.stop()


# --------------------------------------------------------------------------- #
# UsbSpeaker retry-exhaustion path
# --------------------------------------------------------------------------- #
def _install_failing_pyaudio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake ``pyaudio`` whose ``PyAudio()`` always raises OSError.

    ``UsbSpeaker.start`` imports pyaudio *locally*, so a ``sys.modules`` insert
    is picked up without any module reload (avoids the cv2-style reload hazard).
    """

    def _raise() -> Any:
        raise OSError("device missing")

    fake = SimpleNamespace(PyAudio=_raise, paFloat32=1, paInt16=8)
    monkeypatch.setitem(sys.modules, "pyaudio", fake)


def _one_shot_speaker_cfg() -> SpeakerConfig:
    # A single attempt means the backoff sleep is never reached, so no wait.
    return SpeakerConfig.model_validate({"reconnect_max_attempts": 1})


@pytest.mark.asyncio
async def test_usb_speaker_retry_exhaustion_increments_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_failing_pyaudio(monkeypatch)
    metrics = MetricsRegistry(MetricsConfig())
    speaker = UsbSpeaker(_one_shot_speaker_cfg(), metrics=metrics)

    with pytest.raises(SpeakerUnavailableError):
        await speaker.start()

    out = metrics.render_prometheus()
    assert 'mousedroid_voice_speaker_degraded_total{subsystem="usb_speaker"} 1' in out


@pytest.mark.asyncio
async def test_usb_speaker_retry_exhaustion_without_metrics_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_failing_pyaudio(monkeypatch)
    speaker = UsbSpeaker(_one_shot_speaker_cfg())  # metrics=None

    with pytest.raises(SpeakerUnavailableError):
        await speaker.start()
