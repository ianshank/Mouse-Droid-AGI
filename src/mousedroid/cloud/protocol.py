"""Cloud integration protocols — interfaces for all GCP sinks and exporters.

Every concrete implementation lives behind a ``@runtime_checkable Protocol``
so that the orchestrator and factory never import GCP SDK types directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from mousedroid.experience.protocol import ExperienceProtocol

# Canonical engine-type discriminator for OTA weight-update routing. New
# engines (e.g. ``"affect"``) extend this Literal and the orchestrator's
# ``_apply_one_pending_update`` dispatch in lock-step so a typo at any
# call site fails mypy --strict at the boundary rather than landing as a
# silent ``cloud_weight_update_unknown_engine_type`` dead-letter at
# runtime. The string values are part of the
# ``PendingWeightUpdate.engine_type`` public contract — Prometheus
# ``engine_type`` labels and the ``HuggingFaceWeightUpdatePoller(...,
# engine_type=...)`` constructor argument both rely on them.
EngineType: TypeAlias = Literal["policy", "world_model"]

#: Canonical string for the VLA-policy engine. Use over bare ``"policy"``
#: in production code paths so renames stay greppable.
ENGINE_TYPE_POLICY: EngineType = "policy"
#: Canonical string for the world-model engine.
ENGINE_TYPE_WORLD_MODEL: EngineType = "world_model"


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
    engine_type: EngineType


@runtime_checkable
class WeightUpdatePollerProtocol(Protocol):
    """Polls HF Hub for newer model weights and gates atomic orchestrator swap.

    Implementations download artifacts in the background and surface a
    verified :class:`PendingWeightUpdate` via :attr:`pending_update` for the
    orchestrator to consume at a tick boundary. ``acknowledge_swap`` clears
    the slot once the orchestrator has applied the update so the same
    revision is not re-applied on the next tick.

    Note for external implementors: the public surface is intentionally
    minimal so pollers predating the Tier C1.2 multi-engine mapping
    continue to satisfy this protocol structurally. The optional
    ``engine_type`` property used by the orchestrator's per-engine
    dispatch lives on :class:`EngineTypedWeightUpdatePollerProtocol`
    (an extension protocol). The orchestrator's legacy-kwarg fold-in
    path queries ``getattr(poller, "engine_type", getattr(poller,
    "_engine_type", "policy"))`` so external pollers may declare the
    extended protocol, expose the legacy private ``_engine_type``
    attribute, or omit both (defaulting to the policy engine).
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
class EngineTypedWeightUpdatePollerProtocol(WeightUpdatePollerProtocol, Protocol):
    """Extension protocol for pollers that expose ``engine_type`` (Tier C1.2).

    The base :class:`WeightUpdatePollerProtocol` deliberately omits this
    property so external pollers written before Tier C1.2 still satisfy
    it structurally. The Tier C1.2 multi-engine factory + dispatch path
    queries ``engine_type`` via the orchestrator's ``getattr`` fallback
    chain, so implementing this extension is optional — but recommended
    for new pollers because it gives both ``mypy --strict`` callers and
    ``isinstance(poller, EngineTypedWeightUpdatePollerProtocol)``
    runtime checks a precise signal.
    """

    @property
    def engine_type(self) -> EngineType:
        """Engine discriminator the orchestrator dispatches on.

        Exposed on this extension protocol so the orchestrator no longer
        needs to reach into a private ``_engine_type`` attribute when
        folding a legacy single-poller kwarg into the C1.2 dual-poller
        mapping.
        """
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
    """structlog processor that forwards log events to Cloud Logging.

    Unlike the other cloud protocols, this instance's ``start``/``close``
    lifecycle is driven by ``main.py`` directly (not the orchestrator) —
    ``configure_logging()`` runs synchronously before ``build_orchestrator()``
    is ever called, so the sink must be built and threaded through
    ``configure_logging()`` and ``main.py``'s own entry points instead.
    """

    async def start(self) -> None:
        """Initialise the Cloud Logging client."""
        ...

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

    async def close(self) -> None:
        """Release Cloud Logging resources."""
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


@runtime_checkable
class CloudFirestoreSyncProtocol(Protocol):
    """Periodically syncs episodic memory entries to Cloud Firestore."""

    async def start(self) -> None:
        """Initialise the Firestore client and start the sync loop."""
        ...

    async def sync_once(self) -> int:
        """Run a single sync cycle.

        Returns:
            Number of episodes synced.
        """
        ...

    async def close(self) -> None:
        """Stop the sync loop and release resources."""
        ...
