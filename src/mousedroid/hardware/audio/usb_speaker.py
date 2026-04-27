"""USB speaker output driver.

Implements ``SpeakerProtocol`` using PyAudio for real USB audio playback.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mousedroid.hardware.audio.constants import INT16_MAX_F
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import SpeakerConfig

_log = get_logger(__name__)


class UsbSpeaker:
    """USB speaker implementing ``SpeakerProtocol``.

    Uses PyAudio to play audio through a USB speaker device.
    Auto-detects the device by name if ``device_index`` is not specified.
    """

    def __init__(self, cfg: SpeakerConfig) -> None:
        """Initialise USB speaker from config.

        Args:
            cfg: Speaker configuration.
        """
        self._cfg = cfg
        self._pa: Any = None
        self._stream: Any = None
        _log.info(
            "usb_speaker_init",
            sample_rate=cfg.sample_rate,
            channels=cfg.channels,
            chunk_size=cfg.chunk_size,
            device_name=cfg.device_name,
        )

    def _find_device_index(self) -> int | None:
        """Find the output device index matching the configured device name.

        Returns:
            Device index or None if not found.
        """
        import pyaudio

        pa = pyaudio.PyAudio()
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                name = str(info.get("name", ""))
                max_output = int(info.get("maxOutputChannels", 0))
                if self._cfg.device_name.lower() in name.lower() and max_output > 0:
                    _log.info("usb_speaker_found", index=i, name=name)
                    return i
        finally:
            pa.terminate()
        return None

    async def start(self) -> None:
        """Open the PyAudio stream for playback.

        Handles missing PyAudio or ALSA errors gracefully, leaving
        the speaker in a disabled (no-op write) state with a warning.
        """
        try:
            import pyaudio
        except ImportError:
            _log.warning(
                "usb_speaker_unavailable",
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
                        "usb_speaker_not_found",
                        device_name=self._cfg.device_name,
                    )

            fmt = pyaudio.paFloat32 if self._cfg.format == "float32" else pyaudio.paInt16

            self._stream = self._pa.open(
                format=fmt,
                channels=self._cfg.channels,
                rate=self._cfg.sample_rate,
                output=True,
                output_device_index=device_index,
                frames_per_buffer=self._cfg.chunk_size,
            )
            _log.info("usb_speaker_started", device_index=device_index)
        except OSError as exc:
            if self._pa is not None:
                self._pa.terminate()
            self._pa = None
            self._stream = None
            _log.warning(
                "usb_speaker_unavailable",
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
        _log.info("usb_speaker_stopped")

    async def write_chunk(self, samples: NDArray[np.float32]) -> None:
        """Write one chunk of audio to the USB speaker.

        No-op if the speaker failed to start (graceful degradation).
        Clamps samples to [-1.0, 1.0) before int16 conversion to
        prevent overflow distortion.

        Args:
            samples: Audio samples as float32, shape ``(chunk_size * channels,)``.
        """
        if self._stream is None:
            return

        channel_count = max(1, self._cfg.channels)
        if samples.shape[0] % channel_count != 0:
            _log.warning(
                "usb_speaker_channel_misalignment",
                sample_count=int(samples.shape[0]),
                channels=channel_count,
            )
            msg = (
                "Audio sample count must be divisible by configured speaker channels: "
                f"{samples.shape[0]} vs {channel_count}"
            )
            raise ValueError(msg)

        if self._cfg.format == "int16":
            clamped = np.clip(samples, -1.0, np.nextafter(np.float32(1.0), np.float32(0.0)))
            raw_data = (clamped * INT16_MAX_F).astype(np.int16).tobytes()
        else:
            raw_data = samples.astype(np.float32).tobytes()

        frames_to_write = max(1, int(samples.shape[0]) // channel_count)
        get_write_available = getattr(self._stream, "get_write_available", None)

        if callable(get_write_available):
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._cfg.write_timeout_s
            while True:
                available_raw = get_write_available()
                if not isinstance(available_raw, int | float):
                    break
                if int(available_raw) >= frames_to_write:
                    break
                if loop.time() >= deadline:
                    await self.stop()
                    _log.warning(
                        "usb_speaker_write_timeout",
                        frames_requested=frames_to_write,
                        frames_available=int(available_raw),
                        timeout_s=self._cfg.write_timeout_s,
                    )
                    msg = (
                        "USB speaker write timed out waiting for buffer availability "
                        f"after {self._cfg.write_timeout_s:.2f}s"
                    )
                    raise RuntimeError(msg)
                await asyncio.sleep(self._cfg.write_poll_interval_s)

        try:
            await asyncio.to_thread(self._write_raw, raw_data)
        except Exception as exc:
            await self.stop()
            _log.warning("usb_speaker_write_failed", error=str(exc))
            raise RuntimeError(f"USB speaker write failed: {exc}") from exc

    def _write_raw(self, raw_data: bytes) -> None:
        """Synchronously write raw bytes to the underlying PyAudio stream."""
        if self._stream is None:
            msg = "USB speaker stream unavailable"
            raise RuntimeError(msg)
        try:
            self._stream.write(raw_data, exception_on_underflow=False)
        except TypeError:
            self._stream.write(raw_data)

    @property
    def sample_rate(self) -> int:
        """Audio output sample rate in Hz."""
        return self._cfg.sample_rate

    @property
    def channels(self) -> int:
        """Number of audio output channels."""
        return self._cfg.channels

    @property
    def chunk_size(self) -> int:
        """Number of samples per output chunk."""
        return self._cfg.chunk_size
