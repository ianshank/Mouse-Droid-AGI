"""Cloud integration protocols — interfaces for all GCP sinks and exporters.

Every concrete implementation lives behind a ``@runtime_checkable Protocol``
so that the orchestrator and factory never import GCP SDK types directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mousedroid.experience.protocol import ExperienceProtocol


@dataclass(frozen=True)
class PendingWeightUpdate:
    """Verified OTA weight artifact awaiting an orchestrator-side atomic swap.

    Produced by :class:`WeightUpdatePollerProtocol` implementations and
    consumed by ``MouseDroidOrchestrator._apply_pending_weight_update``.
    Frozen (``@dataclass(frozen=True)``) so the orchestrator can treat it
    as a value-class snapshot — no mutation after creation. Slot generation
    is intentionally NOT enabled because operators occasionally subclass
    this dataclass in tests to add extra metadata fields without paying
    the ``__slots__`` ergonomic cost.

    Attributes:
        repo_id: HuggingFace Hub repo ID the artifact came from.
        filename: Filename within the repo (e.g. ``"policy.onnx"``).
        revision: HF Hub commit SHA the artifact was pinned to.
        sha256: Hex-encoded SHA-256 digest verified against the published
            manifest before the update became pending.
        local_path: Local filesystem path the artifact was atomically
            renamed into (final destination).
        downloaded_at: Wall-clock seconds since the epoch when the download
            finished (set with ``time.time()``).
        engine_type: Engine identifier this artifact targets. One of
            ``"policy"`` or ``"world_model"``. Drives the orchestrator's
            swap dispatch + the ``engine_type`` Prometheus label.
    """

    repo_id: str
    filename: str
    revision: str
    sha256: str
    local_path: Path
    downloaded_at: float
    engine_type: str


@runtime_checkable
class WeightUpdatePollerProtocol(Protocol):
    """Polls HF Hub for newer model weights and gates atomic orchestrator swap.

    Implementations download artifacts in the background and surface a
    verified :class:`PendingWeightUpdate` via :attr:`pending_update` for the
    orchestrator to consume at a tick boundary. ``acknowledge_swap`` clears
    the slot once the orchestrator has applied the update so the same
    revision is not re-applied on the next tick.
    """

    async def start(self) -> None:
        """Begin the background poll loop."""
        ...

    async def stop(self) -> None:
        """Stop the poll loop + cancel any in-flight downloads."""
        ...

    @property
    def pending_update(self) -> PendingWeightUpdate | None:
        """Latest verified update awaiting orchestrator swap; ``None`` when no update."""
        ...

    def acknowledge_swap(self, update: PendingWeightUpdate) -> None:
        """Clear the pending slot after the orchestrator has applied ``update``."""
        ...


@runtime_checkable
class GCSBlobProtocol(Protocol):
    """Structural protocol for the subset of GCS blob APIs we use."""

    def upload_from_string(self, data: bytes) -> object:
        """Upload raw bytes to the object store."""
        ...


@runtime_checkable
class GCSBucketProtocol(Protocol):
    """Structural protocol for the subset of GCS bucket APIs we use."""

    def blob(self, blob_name: str) -> GCSBlobProtocol:
        """Return a blob handle for ``blob_name``."""
        ...


@runtime_checkable
class GCSClientProtocol(Protocol):
    """Structural protocol for the subset of GCS client APIs we use."""

    def bucket(self, bucket_name: str) -> GCSBucketProtocol:
        """Return a bucket handle for ``bucket_name``."""
        ...


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
