"""Cloud integration protocols — interfaces for all GCP sinks and exporters.

Every concrete implementation lives behind a ``@runtime_checkable Protocol``
so that the orchestrator and factory never import GCP SDK types directly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from mousedroid.experience.protocol import ExperienceProtocol


@runtime_checkable
class CloudTelemetrySinkProtocol(Protocol):
    """Async sink that publishes telemetry frames and experience records to the cloud."""

    async def start(self) -> None:
        """Initialise client connections (called during orchestrator startup)."""
        ...

    async def publish_telemetry(self, frame_dict: dict[str, Any]) -> None:
        """Publish a serialised telemetry frame.

        Args:
            frame_dict: Dictionary representation of a ``TelemetryFrame``.
        """
        ...

    async def publish_experience(self, record: ExperienceProtocol) -> None:
        """Publish a single experience record.

        Args:
            record: Experience record implementing ``ExperienceProtocol``.
        """
        ...

    async def flush(self) -> None:
        """Flush any buffered messages to the cloud."""
        ...

    async def close(self) -> None:
        """Release resources and close connections."""
        ...


@runtime_checkable
class CloudExperienceExporterProtocol(Protocol):
    """Batch exporter that uploads LMDB experience shards to Cloud Storage."""

    async def start(self) -> None:
        """Initialise client and start periodic export (called during orchestrator startup)."""
        ...

    async def export_pending(self) -> int:
        """Export pending experience records to cloud storage.

        Returns:
            Number of records exported in this batch.
        """
        ...

    async def close(self) -> None:
        """Release resources and close connections."""
        ...


@runtime_checkable
class CloudLoggingSinkProtocol(Protocol):
    """structlog processor that forwards log events to Cloud Logging."""

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a structlog event and forward to Cloud Logging.

        Args:
            logger: The wrapped logger object.
            method_name: The name of the log method called.
            event_dict: The structured event dictionary.

        Returns:
            The event dictionary (unchanged — pass-through processor).
        """
        ...


@runtime_checkable
class CloudMetricsExporterProtocol(Protocol):
    """Periodically exports ``MetricsRegistry`` data to Cloud Monitoring."""

    async def export_once(self) -> None:
        """Run a single export cycle."""
        ...

    async def start(self) -> None:
        """Start the periodic export background task."""
        ...

    async def stop(self) -> None:
        """Stop the periodic export background task and release resources."""
        ...
