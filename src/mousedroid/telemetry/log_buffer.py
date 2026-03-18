"""Log ring buffer — structlog processor that captures entries for streaming.

Installed into the structlog processor chain to intercept log events and
store them in a bounded ring buffer. Clients can retrieve recent entries
via REST or subscribe for live streaming via WebSocket.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
from collections import deque
from typing import Any

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class LogRingBuffer:
    """Thread-safe ring buffer that captures structlog entries.

    Installed as a structlog processor. Stores the last N log entries
    for REST/WebSocket retrieval. Passes events through unchanged
    (transparent processor).

    Args:
        maxlen: Maximum number of log entries to retain.
    """

    def __init__(self, maxlen: int = 200) -> None:
        """Initialise the ring buffer.

        Args:
            maxlen: Maximum entries to retain.
        """
        self._buffer: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Structlog processor — capture event_dict into buffer.

        Args:
            logger: The logger instance.
            method_name: The log method name (info, warning, etc.).
            event_dict: The structured log event dictionary.

        Returns:
            The event_dict unchanged (passthrough).
        """
        entry = copy.copy(event_dict)
        self._buffer.append(entry)

        for sub_queue in self._subscribers:
            with contextlib.suppress(asyncio.QueueFull):
                sub_queue.put_nowait(entry)

        return event_dict

    def get_recent(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the N most recent log entries.

        Args:
            n: Number of entries to return.

        Returns:
            List of log event dictionaries, most recent last.
        """
        entries = list(self._buffer)
        return entries[-n:] if n < len(entries) else entries

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Create a new subscriber queue for live log streaming.

        Returns:
            An ``asyncio.Queue`` that receives new log entries.
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a subscriber queue.

        Args:
            queue: The subscriber queue to remove.
        """
        with contextlib.suppress(ValueError):
            self._subscribers.remove(queue)

    @property
    def size(self) -> int:
        """Current number of entries in the buffer.

        Returns:
            Number of stored entries.
        """
        return len(self._buffer)
