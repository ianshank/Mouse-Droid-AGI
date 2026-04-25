"""Hardware abstraction protocols for vision, distance, audio, and LiDAR sensors.

All hardware interfaces use ``@runtime_checkable`` structural typing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mousedroid.hardware.display.expressions import Expression
    from mousedroid.sensing.lidar_scan import LidarScan


@runtime_checkable
class VisionProtocol(Protocol):
    """Interface for all vision drivers (IMX500, mock, future cameras)."""

    async def capture_features(self) -> NDArray[np.float32]:
        """Capture and extract vision feature vector.

        Returns:
            Feature vector, shape ``(feature_dim,)``.
        """
        ...

    @property
    def feature_dim(self) -> int:
        """Output feature vector dimension."""
        ...

    async def start(self) -> None:
        """Start camera capture pipeline."""
        ...

    async def stop(self) -> None:
        """Stop camera capture pipeline."""
        ...


@runtime_checkable
class RawFrameSourceProtocol(Protocol):
    """Optional capability for vision drivers that can emit raw JPEG frames.

    Implemented by :class:`MockCamera` for offline dashboard validation and
    by on-NPU drivers that can tee the pre-inference frame. Consumers
    should duck-type via ``isinstance(camera, RawFrameSourceProtocol)``
    so drivers without raw-frame support remain usable.
    """

    async def capture_raw_jpeg(self) -> bytes | None:
        """Return a JPEG-encoded snapshot of the current scene.

        Returns:
            JPEG bytes, or ``None`` when no frame is available (e.g. the
            first call before the pipeline has warmed up).
        """
        ...


@runtime_checkable
class DistanceSensorProtocol(Protocol):
    """Interface for single-point distance sensors (HC-SR04, etc)."""

    async def read_distance_m(self) -> float:
        """Read distance measurement.

        Returns:
            Distance in metres. Returns max_range_m if no echo detected.
        """
        ...

    @property
    def max_range_m(self) -> float:
        """Maximum detection range in metres."""
        ...

    @property
    def min_range_m(self) -> float:
        """Minimum detection range in metres."""
        ...


@runtime_checkable
class AudioProtocol(Protocol):
    """Interface for all audio input drivers (USB microphone, mock, etc)."""

    async def read_chunk(self) -> NDArray[np.float32]:
        """Read one chunk of audio samples.

        Returns:
            Audio samples, shape ``(chunk_size * channels,)``.
        """
        ...

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        ...

    @property
    def channels(self) -> int:
        """Number of audio channels (1=mono, 2=stereo)."""
        ...

    @property
    def chunk_size(self) -> int:
        """Number of samples per chunk."""
        ...

    async def start(self) -> None:
        """Start audio capture stream."""
        ...

    async def stop(self) -> None:
        """Stop audio capture stream."""
        ...


@runtime_checkable
class SpeakerProtocol(Protocol):
    """Interface for all audio output drivers (USB speaker, mock, etc)."""

    async def write_chunk(self, samples: NDArray[np.float32]) -> None:
        """Write one chunk of audio samples to the speaker.

        Args:
            samples: Audio samples, shape ``(chunk_size * channels,)``.
        """
        ...

    @property
    def sample_rate(self) -> int:
        """Audio output sample rate in Hz."""
        ...

    @property
    def channels(self) -> int:
        """Number of audio output channels (1=mono, 2=stereo)."""
        ...

    @property
    def chunk_size(self) -> int:
        """Number of samples per output chunk."""
        ...

    async def start(self) -> None:
        """Start audio playback stream."""
        ...

    async def stop(self) -> None:
        """Stop audio playback stream."""
        ...


@runtime_checkable
class LidarProtocol(Protocol):
    """Interface for 2D LiDAR scanners (FHL-LD19, etc).

    Unlike :class:`DistanceSensorProtocol` (single-point), LiDAR returns
    a full 360-degree scan per read.
    """

    async def read_scan(self) -> LidarScan:
        """Read a full 360-degree scan.

        Returns:
            A :class:`LidarScan` containing angles, distances, and
            confidences for every measured point in one rotation.
        """
        ...

    @property
    def max_range_m(self) -> float:
        """Maximum detection range in metres."""
        ...

    @property
    def min_range_m(self) -> float:
        """Minimum detection range in metres."""
        ...

    @property
    def scan_frequency_hz(self) -> float:
        """Nominal scan rotation frequency in Hz."""
        ...

    async def start(self) -> None:
        """Start LiDAR motor and data acquisition."""
        ...

    async def stop(self) -> None:
        """Stop LiDAR motor and data acquisition."""
        ...


@runtime_checkable
class FaceDisplayProtocol(Protocol):
    """Interface for the MSE-6 face display (SSD1306 OLED, mock, etc.).

    Renders discrete facial expressions and short status text. All methods
    are ``async`` so blocking I²C writes can be bridged with
    ``asyncio.to_thread`` without stalling the orchestrator loop.
    """

    async def start(self) -> None:
        """Initialise the display device and any background animation tasks."""
        ...

    async def stop(self) -> None:
        """Shut down background tasks and release the display device."""
        ...

    async def show_expression(self, expression: Expression) -> None:
        """Render a discrete facial expression.

        Args:
            expression: Target :class:`Expression` value to render.
        """
        ...

    async def show_text(self, message: str) -> None:
        """Render a short single-line status message (used for boot banner)."""
        ...
