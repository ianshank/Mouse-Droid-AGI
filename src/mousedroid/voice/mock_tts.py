"""Mock TTS for testing — returns silence and records calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import VoiceConfig

_log = get_logger(__name__)


class MockTTS:
    """Mock text-to-speech for testing.

    Returns fixed-length silence and records all synthesis requests.
    """

    def __init__(self, cfg: VoiceConfig) -> None:
        """Initialise mock TTS.

        Args:
            cfg: Voice engine configuration.
        """
        self._cfg = cfg
        self._calls: list[str] = []
        _log.info("mock_tts_init")

    def start(self) -> None:
        """No-op start."""
        _log.info("mock_tts_started")

    def stop(self) -> None:
        """No-op stop."""
        _log.info("mock_tts_stopped")

    async def synthesize(self, text: str) -> NDArray[np.float32]:
        """Return silence and record the call.

        Args:
            text: Text that would be spoken.

        Returns:
            Silence array of one second duration.
        """
        self._calls.append(text)
        return np.zeros(self._cfg.tts_sample_rate, dtype=np.float32)

    def get_calls(self) -> list[str]:
        """Return all texts passed to ``synthesize``.

        Returns:
            List of synthesised text strings.
        """
        return list(self._calls)

    def clear(self) -> None:
        """Clear recorded calls."""
        self._calls.clear()
