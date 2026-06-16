"""USB audio input driver (Wonrabai USB Sound Card or compatible).

Implements ``AudioProtocol`` using PyAudio for real USB audio capture.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.hardware.audio._device_discovery import find_pyaudio_device_index
from mousedroid.hardware.audio.constants import INT16_MAX_F
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import MicrophoneConfig

_log = get_logger(__name__)


class UsbMicrophone:
    """USB audio input driver implementing ``AudioProtocol``.

    Uses PyAudio to capture audio from a USB microphone device.
    Auto-detects the device by name if ``device_index`` is not specified.
    Handles missing PyAudio or ALSA errors gracefully, leaving
    the microphone in a disabled (silence) state with a warning.
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
        """Find the input device index matching the configured device name.

        Returns:
            Device index or None if not found.
        """
        return find_pyaudio_device_index(
            self._cfg.device_name,
            want_input=True,
            log_event="usb_microphone_found",
        )

    async def start(self) -> None:
        """Open the PyAudio stream for capture.

        Handles missing PyAudio or ALSA errors gracefully, leaving
        the microphone in a disabled (silence) state with a warning.
        """
        try:
            import pyaudio
        except ImportError:
            _log.warning(
                "usb_microphone_unavailable",
                reason="pyaudio_import_failed",
                device_name=self._cfg.device_name,
            )
            return

        try:
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
        except OSError as exc:
            if self._pa is not None:
                self._pa.terminate()
            self._pa = None
            self._stream = None
            _log.warning(
                "usb_microphone_unavailable",
                reason="pyaudio_open_failed",
                error=str(exc),
                device_name=self._cfg.device_name,
            )

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

        Returns silence if the stream failed to start (graceful degradation).

        Returns:
            Audio samples as float32, shape ``(chunk_size * channels,)``.
        """
        silence = np.zeros(
            self._cfg.chunk_size * self._cfg.channels,
            dtype=np.float32,
        )

        if self._stream is None:
            return silence

        try:
            loop = asyncio.get_running_loop()
            raw_data: bytes = await loop.run_in_executor(
                None,
                self._stream.read,
                self._cfg.chunk_size,
            )
        except OSError as exc:
            _log.warning(
                "usb_microphone_read_failed",
                error=str(exc),
                device_name=self._cfg.device_name,
            )
            self._stream = None
            return silence

        if self._cfg.format == "int16":
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / INT16_MAX_F
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
