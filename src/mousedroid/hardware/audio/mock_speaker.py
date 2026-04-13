"""Mock USB speaker for testing and simulation.

Implements ``SpeakerProtocol`` with captured output for test assertions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import SpeakerConfig

_log = get_logger(__name__)


class MockSpeaker:
    """Mock USB speaker implementing ``SpeakerProtocol``.

    Captures written audio chunks for test verification.
    """

    def __init__(self, cfg: SpeakerConfig) -> None:
        """Initialise mock speaker from config.

        Args:
            cfg: Speaker configuration.
        """
        self._cfg = cfg
        self._started = False
        self._written_chunks: list[NDArray[np.float32]] = []
        _log.info(
            "mock_speaker_init",
            sample_rate=cfg.sample_rate,
            channels=cfg.channels,
            chunk_size=cfg.chunk_size,
        )

    async def write_chunk(self, samples: NDArray[np.float32]) -> None:
        """Capture audio chunk for test assertions.

        Args:
            samples: Audio samples, shape ``(chunk_size * channels,)``.
        """
        self._written_chunks.append(samples.copy())

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

    async def start(self) -> None:
        """Start mock audio playback."""
        self._started = True
        _log.info("mock_speaker_started")

    async def stop(self) -> None:
        """Stop mock audio playback."""
        self._started = False
        _log.info("mock_speaker_stopped")

    @property
    def started(self) -> bool:
        """Whether the mock speaker is currently started."""
        return self._started

    def get_written_chunks(self) -> list[NDArray[np.float32]]:
        """Return all captured audio chunks.

        Returns:
            List of audio chunk arrays written so far.
        """
        return list(self._written_chunks)

    def clear(self) -> None:
        """Clear all captured audio chunks."""
        self._written_chunks.clear()
