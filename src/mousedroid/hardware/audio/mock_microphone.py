"""Mock USB microphone for testing and simulation.

Implements ``AudioProtocol`` with configurable return values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import MicrophoneConfig

_log = get_logger(__name__)


class MockMicrophone:
    """Mock USB microphone implementing ``AudioProtocol``.

    Returns configurable audio chunks for test control.
    """

    def __init__(self, cfg: MicrophoneConfig) -> None:
        """Initialise mock microphone from config.

        Args:
            cfg: Microphone configuration.
        """
        self._cfg = cfg
        self._chunk: NDArray[np.float32] | None = None
        self._started = False
        _log.info(
            "mock_microphone_init",
            sample_rate=cfg.sample_rate,
            channels=cfg.channels,
            chunk_size=cfg.chunk_size,
        )

    async def read_chunk(self) -> NDArray[np.float32]:
        """Return the configured mock audio chunk.

        Returns:
            Audio samples, shape ``(chunk_size * channels,)``.
        """
        if self._chunk is not None:
            return self._chunk
        return np.random.default_rng().standard_normal(
            self._cfg.chunk_size * self._cfg.channels,
        ).astype(np.float32)

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

    async def start(self) -> None:
        """Start mock audio capture."""
        self._started = True
        _log.info("mock_microphone_started")

    async def stop(self) -> None:
        """Stop mock audio capture."""
        self._started = False
        _log.info("mock_microphone_stopped")

    @property
    def started(self) -> bool:
        """Whether the mock microphone is currently started."""
        return self._started

    def set_chunk(self, data: NDArray[np.float32]) -> None:
        """Set the audio chunk returned by ``read_chunk``.

        Args:
            data: Audio samples to return.
        """
        self._chunk = data
