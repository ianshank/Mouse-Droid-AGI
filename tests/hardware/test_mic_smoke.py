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
def test_usb_mic_detected():
    try:
        import pyaudio
    except ImportError:
        pytest.skip("pyaudio not installed")

    pa = pyaudio.PyAudio()
    try:
        found = False
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) > 0:
                found = True
                break
        assert found, "No audio input device found"
    finally:
        pa.terminate()


@pytest.mark.hardware
async def test_usb_mic_capture():
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        pytest.skip("pyaudio not installed")

    from mousedroid.config.schema import MicrophoneConfig
    from mousedroid.hardware.audio.usb_microphone import UsbMicrophone

    cfg = MicrophoneConfig(chunk_size=512, sample_rate=16000, channels=1)
    mic = UsbMicrophone(cfg)
    try:
        await mic.start()
        chunk = await mic.read_chunk()
        assert chunk.dtype == np.float32
        assert chunk.shape == (512,)
    finally:
        await mic.stop()
