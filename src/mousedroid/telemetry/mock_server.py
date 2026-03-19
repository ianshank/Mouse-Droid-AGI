"""Mock telemetry server — for testing without network binding.

Implements ``TelemetryServerProtocol``. Records published frames for
assertion in tests. No actual HTTP or WebSocket connections.
"""

from __future__ import annotations

from mousedroid.logging.setup import get_logger
from mousedroid.telemetry.protocol import TelemetryFrame

_log = get_logger(__name__)


class MockTelemetryServer:
    """Mock telemetry server for testing — no network binding.

    Implements ``TelemetryServerProtocol``. Records lifecycle events
    for assertion in tests.
    """

    def __init__(self) -> None:
        """Initialise mock server."""
        self._started = False
        self._client_count = 0
        self._frames: list[TelemetryFrame] = []

    def record_frame(self, frame: TelemetryFrame) -> None:
        """Record a telemetry frame for later inspection in tests.

        Args:
            frame: The ``TelemetryFrame`` instance to record.
        """
        self._frames.append(frame)

    async def start(self) -> None:
        """Mark server as started."""
        self._started = True
        _log.info("mock_telemetry_server_started")

    async def stop(self) -> None:
        """Mark server as stopped."""
        self._started = False
        _log.info("mock_telemetry_server_stopped")

    @property
    def client_count(self) -> int:
        """Always returns 0 for mock server."""
        return self._client_count

    @property
    def is_running(self) -> bool:
        """Whether the mock server was started."""
        return self._started

    @property
    def received_frames(self) -> list[TelemetryFrame]:
        """Frames recorded by the mock server.

        Returns:
            List of received ``TelemetryFrame`` objects.
        """
        return list(self._frames)
