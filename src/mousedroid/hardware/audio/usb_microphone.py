"""SuziePi USB 2.0 Mini Microphone driver.

Implements ``AudioProtocol`` using PyAudio for real USB audio capture.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import MicrophoneConfig

_log = get_logger(__name__)


class UsbMicrophone:
    """SuziePi USB 2.0 Mini Microphone implementing ``AudioProtocol``.

    Uses PyAudio to capture audio from a USB microphone device.
    Auto-detects the device by name if ``device_index`` is not specified.
    """

    def __init__(self, cfg: MicrophoneConfig) -> None:
        """Initialise USB microphone from config.

        Args:
            cfg: Microphone configuration.
        """
        self._cfg = cfg
        self._pa: Any = None
        self._stream: Any = None
        _log.info(
            "usb_microphone_init",
            sample_rate=cfg.sample_rate,
            channels=cfg.channels,
            chunk_size=cfg.chunk_size,
            device_name=cfg.device_name,
        )

    def _find_device_index(self) -> int | None:
        """Find the device index matching the configured device name.

        Returns:
            Device index or None if not found.
        """
        import pyaudio

        pa = pyaudio.PyAudio()
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                name = str(info.get("name", ""))
                max_input = int(info.get("maxInputChannels", 0))
                if self._cfg.device_name.lower() in name.lower() and max_input > 0:
                    _log.info("usb_microphone_found", index=i, name=name)
                    return i
        finally:
            pa.terminate()
        return None

    async def start(self) -> None:
        """Open the PyAudio stream for capture."""
        import pyaudio

        self._pa = pyaudio.PyAudio()

        device_index = self._cfg.device_index
        if device_index is None:
            device_index = self._find_device_index()
            if device_index is None:
                _log.warning(
                    "usb_microphone_not_found",
                    device_name=self._cfg.device_name,
                )

        fmt = pyaudio.paFloat32 if self._cfg.format == "float32" else pyaudio.paInt16

        self._stream = self._pa.open(
            format=fmt,
            channels=self._cfg.channels,
            rate=self._cfg.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self._cfg.chunk_size,
        )
        _log.info("usb_microphone_started", device_index=device_index)

    async def stop(self) -> None:
        """Close the PyAudio stream and terminate."""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        _log.info("usb_microphone_stopped")

    async def read_chunk(self) -> NDArray[np.float32]:
        """Read one chunk of audio from the USB microphone.

        Returns:
            Audio samples as float32, shape ``(chunk_size * channels,)``.
        """
        if self._stream is None:
            msg = "Microphone stream not started"
            raise RuntimeError(msg)

        loop = asyncio.get_running_loop()
        raw_data: bytes = await loop.run_in_executor(
            None,
            self._stream.read,
            self._cfg.chunk_size,
        )

        if self._cfg.format == "int16":
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            samples = np.frombuffer(raw_data, dtype=np.float32)

        return samples

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        return self._cfg.sample_rate

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self._cfg.channels

    @property
    def chunk_size(self) -> int:
        """Number of samples per chunk."""
        return self._cfg.chunk_size
