"""Shared PyAudio device-discovery helper for USB audio drivers.

Both :class:`~mousedroid.hardware.audio.usb_microphone.UsbMicrophone` and
:class:`~mousedroid.hardware.audio.usb_speaker.UsbSpeaker` auto-detect their
device by name. The enumeration loop is identical apart from which channel
direction is required (input vs output) and which structlog event is emitted
on a match — this module hosts that single implementation so the two drivers
cannot drift.

``pyaudio`` is an optional hardware dependency, so it is imported lazily inside
the helper (mirroring the original per-driver methods) rather than at module
import time.
"""

from __future__ import annotations

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


def find_pyaudio_device_index(
    device_name: str,
    *,
    want_input: bool,
    log_event: str,
) -> int | None:
    """Find the index of the first PyAudio device matching name and direction.

    Enumerates every device PyAudio reports and returns the index of the first
    whose name contains ``device_name`` (case-insensitive substring match) and
    that exposes at least one channel in the required direction
    (``maxInputChannels > 0`` when ``want_input`` is true, otherwise
    ``maxOutputChannels > 0``). On a match, ``log_event`` is logged at INFO with
    ``index`` and ``name`` fields. A fresh ``pyaudio.PyAudio()`` instance is
    created for the enumeration and always terminated before returning.

    Args:
        device_name: Substring matched case-insensitively against each device
            name (typically ``cfg.device_name``).
        want_input: When ``True`` require an input-capable device
            (``maxInputChannels > 0``); when ``False`` require an output-capable
            device (``maxOutputChannels > 0``).
        log_event: structlog event name emitted on a successful match
            (e.g. ``"usb_microphone_found"`` or ``"usb_speaker_found"``).

    Returns:
        The device index of the first match, or ``None`` if no device matches.
    """
    import pyaudio

    channel_field = "maxInputChannels" if want_input else "maxOutputChannels"

    pa = pyaudio.PyAudio()
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = str(info.get("name", ""))
            max_channels = int(info.get(channel_field, 0))
            if device_name.lower() in name.lower() and max_channels > 0:
                _log.info(log_event, index=i, name=name)
                return i
    finally:
        pa.terminate()
    return None
