from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.hardware
def test_pyaudio_available() -> None:
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        pytest.skip("pyaudio not installed")


@pytest.mark.hardware
def test_usb_speaker_detected(jetson_settings) -> None:
    try:
        import pyaudio
    except ImportError:
        pytest.skip("pyaudio not installed")

    cfg = jetson_settings.speaker
    if cfg is None or not cfg.enabled:
        pytest.skip("speaker disabled in config")

    needle = cfg.device_name.lower()

    pa = pyaudio.PyAudio()
    try:
        matches: list[str] = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = str(info.get("name", ""))
            if int(info.get("maxOutputChannels", 0)) > 0 and needle in name.lower():
                matches.append(f"[{i}] {name}")
    finally:
        pa.terminate()

    if not matches:
        pytest.skip(f"no output device matching '{cfg.device_name}' — USB speaker not connected?")

    assert matches, "matches list unexpectedly empty after skip guard"


@pytest.mark.hardware
async def test_usb_speaker_write_chunk(jetson_settings) -> None:
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        pytest.skip("pyaudio not installed")

    from mousedroid.hardware.audio.usb_speaker import UsbSpeaker

    cfg = jetson_settings.speaker
    if cfg is None or not cfg.enabled:
        pytest.skip("speaker disabled in config")

    speaker = UsbSpeaker(cfg)

    try:
        await speaker.start()
    except Exception as exc:
        pytest.skip(f"USB speaker unavailable: {exc}")

    # start() degrades silently when no device is found.
    if getattr(speaker, "_stream", None) is None:
        await speaker.stop()
        pytest.skip(f"USB speaker '{cfg.device_name}' not found — graceful no-op mode engaged")

    try:
        freq_hz = 440.0
        duration_chunks = 4
        total_samples = cfg.chunk_size * duration_chunks
        t = np.arange(total_samples, dtype=np.float32) / float(cfg.sample_rate)
        tone = (0.1 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)

        for start in range(0, total_samples, cfg.chunk_size):
            chunk = tone[start : start + cfg.chunk_size]
            await speaker.write_chunk(chunk)
    finally:
        await speaker.stop()
