from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.hardware
def test_pyaudio_available():
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        pytest.skip("pyaudio not installed")


@pytest.mark.hardware
def test_usb_mic_detected(jetson_settings):
    try:
        import pyaudio
    except ImportError:
        pytest.skip("pyaudio not installed")

    cfg = jetson_settings.microphone
    if cfg is None or not cfg.enabled:
        pytest.skip("microphone disabled in config")

    needle = cfg.device_name.lower()

    pa = pyaudio.PyAudio()
    try:
        found = False
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if (
                int(info.get("maxInputChannels", 0)) > 0
                and needle in str(info.get("name", "")).lower()
            ):
                found = True
                break
        assert found, f"No audio input device matching '{cfg.device_name}' found"
    finally:
        pa.terminate()


@pytest.mark.hardware
async def test_usb_mic_capture(jetson_settings):
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        pytest.skip("pyaudio not installed")

    from mousedroid.hardware.audio.usb_microphone import UsbMicrophone

    cfg = jetson_settings.microphone
    if cfg is None or not cfg.enabled:
        pytest.skip("microphone disabled in config")

    mic = UsbMicrophone(cfg)
    try:
        await mic.start()
        if getattr(mic, "_stream", None) is None:
            pytest.skip(f"USB microphone '{cfg.device_name}' not available")
        chunk = await mic.read_chunk()
        assert chunk.dtype == np.float32
        assert chunk.shape == (cfg.chunk_size * cfg.channels,)
    finally:
        await mic.stop()
