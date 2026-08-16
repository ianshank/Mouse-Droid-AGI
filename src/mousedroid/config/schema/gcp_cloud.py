"""GCP Digital Twin and Tier C1 cloud-retraining configuration models.

Pub/Sub, GCS, Cloud Logging/Monitoring, Firestore, Vertex AI training, and
GKE simulation sub-blocks composed by the top-level ``GCPConfig``; plus the
HuggingFace Hub OTA weight-update poller (``CloudConfig``), orthogonal to
the GCP data pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from mousedroid.config.schema._primitives import _settings_default_factory
from mousedroid.config.schema.misc import CircuitBreakerConfig, RetryConfig


class GCPPubSubConfig(BaseModel):
    """Google Cloud Pub/Sub configuration for telemetry and experience export."""

    telemetry_topic: str = Field(
        "mousedroid-telemetry",
        description="Pub/Sub topic for telemetry frames",
    )
    experience_topic: str = Field(
        "mousedroid-experience",
        description="Pub/Sub topic for experience records",
    )
    batch_max_messages: int = Field(
        100,
        gt=0,
        description="Max messages per Pub/Sub publish batch",
    )
    batch_max_bytes: int = Field(
        1_048_576,
        gt=0,
        description="Max bytes per publish batch (default 1 MB)",
    )
    batch_max_latency_s: float = Field(
        1.0,
        gt=0,
        description="Max batch latency before flush (s)",
    )
    publish_timeout_s: float = Field(
        10.0,
        gt=0,
        description="Timeout in seconds for individual publish futures",
    )
    ordering_key: str = Field(
        "mousedroid-0",
        description="Message ordering key for ordered delivery",
    )


class GCPStorageConfig(BaseModel):
    """Google Cloud Storage configuration for experience archival."""

    bucket: str = Field(
        "mousedroid-experience",
        description="GCS bucket name for experience shards",
    )
    prefix: str = Field(
        "experience/v1",
        description="Object key prefix for experience data",
    )
    upload_batch_size: int = Field(
        1000,
        gt=0,
        description="Number of experience records per GCS shard file",
    )
    upload_interval_s: float = Field(
        300.0,
        gt=0,
        description="Seconds between GCS shard uploads",
    )
    compression: Literal["none", "gzip", "zstd"] = Field(
        "gzip",
        description="Shard file compression algorithm",
    )


class GCPLoggingConfig(BaseModel):
    """Google Cloud Logging sink configuration."""

    enabled: bool = Field(True, description="Forward structlog events to Cloud Logging")
    log_name: str = Field("mousedroid", description="Cloud Logging log name")
    min_level: str = Field("INFO", description="Minimum log level to forward to cloud")


class GCPMonitoringConfig(BaseModel):
    """Google Cloud Monitoring configuration for metrics export."""

    enabled: bool = Field(True, description="Export metrics to Cloud Monitoring")
    export_interval_s: float = Field(
        60.0,
        gt=0,
        description="Seconds between metric export batches",
    )
    metric_prefix: str = Field(
        "custom.googleapis.com/mousedroid",
        description="Cloud Monitoring custom metric type prefix",
    )


class GCPFirestoreConfig(BaseModel):
    """Firestore configuration for episodic memory synchronisation."""

    enabled: bool = Field(False, description="Sync episodic memory to Firestore")
    collection: str = Field(
        "mousedroid_episodes",
        description="Firestore collection for episode documents",
    )
    sync_interval_s: float = Field(
        120.0,
        gt=0,
        description="Seconds between episodic memory sync batches",
    )
    sync_batch_size: int = Field(
        10,
        gt=0,
        description="Max episodes to sync per batch",
    )


class GCPTrainingConfig(BaseModel):
    """Vertex AI cloud training pipeline configuration."""

    training_bucket: str = Field(
        "mousedroid-training",
        description="GCS bucket for training datasets and checkpoints",
    )
    pipeline_region: str = Field(
        "us-central1",
        description="Vertex AI pipeline region",
    )
    machine_type: str = Field(
        "a2-highgpu-1g",
        description="Training VM machine type (A100 GPU)",
    )
    accelerator_type: str = Field(
        "NVIDIA_TESLA_A100",
        description="GPU accelerator type for training",
    )
    accelerator_count: int = Field(1, gt=0, description="Number of GPUs per training job")
    max_run_hours: float = Field(
        4.0,
        gt=0,
        description="Maximum pipeline runtime in hours",
    )
    schedule_cron: str = Field(
        "0 2 * * *",
        description="Cloud Scheduler cron expression (UTC) for nightly retraining",
    )
    huggingface_repo: str = Field(
        "ianshank/mousedroid-weights",
        pattern=r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$",
        description="HuggingFace Hub repo for weight push after training",
    )
    ewc_enabled: bool = Field(
        True,
        description="Enable EWC Fisher matrix update step in pipeline",
    )


class GCPSimulationConfig(BaseModel):
    """GKE parallel simulation configuration for safety validation."""

    gke_cluster: str = Field(
        "mousedroid-sim",
        description="GKE Autopilot cluster name for sim pods",
    )
    region: str = Field("us-central1", description="GKE cluster region")
    max_parallel_pods: int = Field(
        50,
        gt=0,
        description="Maximum concurrent simulation pods",
    )
    sim_ticks_per_scenario: int = Field(
        300,
        gt=0,
        description="Orchestrator ticks per scenario (300 = 10 s at 30 Hz)",
    )
    results_bucket: str = Field(
        "mousedroid-sim-results",
        description="GCS bucket for simulation campaign results",
    )
    image: str = Field(
        "gcr.io/mousedroid-twin/mousedroid:sim",
        description="Container image for simulation pods",
    )


class GCPConfig(BaseModel):
    """GCP Digital Twin umbrella configuration.

    When ``None`` in ``Settings``, all GCP features are disabled and the droid
    operates in fully autonomous offline mode with zero cloud dependency.
    """

    project_id: str = Field(..., description="GCP project ID (required)")
    credentials_path: Path | None = Field(
        None,
        description="Service account key path (None = use ADC / metadata server)",
    )
    robot_id: str = Field(
        "droid-001",
        description="Unique identifier for this robot instance",
    )
    pubsub: GCPPubSubConfig = Field(
        default_factory=_settings_default_factory(GCPPubSubConfig),
    )
    storage: GCPStorageConfig = Field(
        default_factory=_settings_default_factory(GCPStorageConfig),
    )
    logging: GCPLoggingConfig = Field(
        default_factory=_settings_default_factory(GCPLoggingConfig),
    )
    monitoring: GCPMonitoringConfig = Field(
        default_factory=_settings_default_factory(GCPMonitoringConfig),
    )
    firestore: GCPFirestoreConfig = Field(
        default_factory=_settings_default_factory(GCPFirestoreConfig),
    )
    training: GCPTrainingConfig | None = Field(
        None,
        description="Cloud training pipeline config (None = no cloud training)",
    )
    simulation: GCPSimulationConfig | None = Field(
        None,
        description="GKE simulation config (None = no cloud simulation)",
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=lambda: CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_s=60.0,
            half_open_max_calls=1,
        ),
        description="Circuit breaker for cloud API calls (tuned for higher latency)",
    )
    retry: RetryConfig = Field(
        default_factory=lambda: RetryConfig(
            max_attempts=3,
            base_delay_s=2.0,
            max_delay_s=60.0,
            exponential_base=2.0,
            jitter_fraction=0.1,
        ),
        description="Retry config for cloud API calls",
    )
    metrics_labels: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Deployment labels (e.g. env, region, fleet) attached to cloud "
            "exports. Keys/values must be non-empty strings. Backwards "
            "compatible: empty dict by default."
        ),
    )

    @model_validator(mode="after")
    def _validate_required_cloud_fields(self) -> GCPConfig:
        """Enforce non-empty identifiers when the digital twin is enabled.

        ``GCPConfig`` itself is optional on :class:`Settings`; when present
        it must identify the project, robot, and every destination that
        downstream sinks will target so they never silently publish to
        empty topic / bucket names.

        Returns:
            The validated instance (unchanged when valid).

        Raises:
            ValueError: If any required identifier is empty / whitespace.
        """
        required: dict[str, str] = {
            "project_id": self.project_id,
            "robot_id": self.robot_id,
            "pubsub.telemetry_topic": self.pubsub.telemetry_topic,
            "pubsub.experience_topic": self.pubsub.experience_topic,
            "storage.bucket": self.storage.bucket,
        }
        empty = [key for key, value in required.items() if not value or not value.strip()]
        if empty:
            raise ValueError("GCPConfig requires non-empty values for: " + ", ".join(sorted(empty)))

        for label_key, label_value in self.metrics_labels.items():
            if not label_key or not label_key.strip():
                raise ValueError("GCPConfig.metrics_labels keys must be non-empty")
            if not isinstance(label_value, str) or not label_value.strip():
                raise ValueError(
                    f"GCPConfig.metrics_labels[{label_key!r}] must be a non-empty string"
                )

        if self.pubsub.telemetry_topic == self.pubsub.experience_topic:
            raise ValueError("GCPConfig.pubsub.telemetry_topic and experience_topic must differ")
        return self


#: Default ``world_model_repo_id`` literal. Defined as a module-level constant
#: so the field default and the
#: ``_warn_on_default_world_model_repo`` validator share one canonical value
#: and a future rename touches one place. This is the maintainer's personal
#: HF Hub repo — operators MUST override before enabling the world-model
#: poller in production.
_WORLD_MODEL_DEFAULT_REPO_ID: str = "ianshank/mousedroid-dual-stream-rssm"


class WeightUpdatePollConfig(BaseModel):
    """Configuration for the HuggingFace Hub OTA weight-update poller.

    Default ``poll_interval_s = 0.0`` disables the poller entirely so
    existing YAML files load with byte-identical pre-Tier-C1 behaviour.
    """

    poll_interval_s: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Background poll interval in seconds. ``0.0`` disables the poller "
            "entirely (default — preserves byte-identical pre-Tier-C1 "
            "behaviour). Operators flip this to e.g. 300.0 to poll every "
            "five minutes for new artifacts."
        ),
    )
    policy_repo_id: str = Field(
        "ianshank/mousedroid-policy-v2",
        description="HuggingFace Hub repo ID containing the trained policy artifact.",
    )
    policy_filename: str = Field(
        "policy.onnx",
        description="Filename within ``policy_repo_id`` of the policy artifact.",
    )
    world_model_repo_id: str = Field(
        _WORLD_MODEL_DEFAULT_REPO_ID,
        description="HuggingFace Hub repo ID containing the trained world-model artifact.",
    )
    world_model_filename: str = Field(
        "observe_step.onnx",
        description="Filename within ``world_model_repo_id`` of the observe_step ONNX export.",
    )
    cache_dir: str = Field(
        "weights/cloud_updates",
        description=(
            "Local directory the poller writes verified artifacts into. "
            "Resolved relative to the runtime CWD unless absolute."
        ),
    )
    sha256_manifest_filename: str = Field(
        "sha256.txt",
        description=(
            "Filename inside the HF repo carrying the expected hex-encoded "
            "SHA-256 digest for the downloaded artifact. Single-line file "
            "containing only the hex digest. SAFETY-CRITICAL: a download is "
            "refused if the local SHA does not match this manifest."
        ),
    )
    reset_state_on_swap: bool = Field(
        True,
        description=(
            "Reset h/z to zeros after swap. Default ``True`` because the "
            "orchestrator's ``tick()`` body runs ``_update_world_model`` "
            "BEFORE ``_select_action``, so a swap mid-sprint leaves the next "
            "tick's ``observe_step`` receiving ``(h, z)`` computed by the OLD "
            "world model. Zeroing the recurrent state on swap is the only "
            "way to avoid that one-tick cross-model contamination — trade-off "
            "is one tick of context loss, which is acceptable for an OTA "
            "event operators expect to happen at minute-scale, not 30 Hz."
        ),
    )
    download_timeout_s: float = Field(
        60.0,
        gt=0.0,
        description="Per-download wall-clock timeout (seconds).",
    )
    max_retries: int = Field(
        3,
        ge=0,
        description="Maximum retry attempts per artifact (forwarded to weights_manager).",
    )
    world_model_enabled: bool = Field(
        False,
        description=(
            "Enable a second OTA poller targeting ``world_model_repo_id`` / "
            "``world_model_filename``. Default ``False`` preserves "
            "byte-identical pre-C1.2 behaviour — only the policy poller is "
            "built. Operators flip to ``True`` when the world-model export "
            "pipeline is producing artifacts to OTA-deploy."
        ),
    )
    upload_extensions: tuple[str, ...] = Field(
        (".onnx", ".pt", ".npz", ".json", ".safetensors"),
        description=(
            "File extensions ``training/upload_weights.py::sync_gcs_to_hf`` "
            "publishes to HF Hub when running the cloud-trainer leg of the "
            "OTA loop. Default includes ``.onnx`` + ``.safetensors`` so the "
            "world-model export and HF-native weight formats round-trip "
            "without operator intervention. Stored as a hashable ``tuple`` "
            "(not a ``set``) so the Pydantic schema stays hashable."
        ),
    )
    gcs_artifact_prefix: str = Field(
        "trained/",
        min_length=1,
        description=(
            "Object prefix inside ``gcp.training.training_bucket`` that the "
            "``--from-gcs`` CLI mode lists. Trailing slash preserved verbatim "
            "(forwarded to ``bucket.list_blobs(prefix=...)``). Default "
            "matches the cloud trainer's output convention. MUST be non-empty: "
            "an empty / whitespace prefix would enumerate the entire training "
            "bucket and publish every matching artifact extension to HF Hub — "
            "a high-impact operator footgun. Enforced both by ``min_length=1`` "
            "and the ``_reject_blank_gcs_artifact_prefix`` validator below."
        ),
    )

    @field_validator("gcs_artifact_prefix", mode="after")
    @classmethod
    def _reject_blank_gcs_artifact_prefix(cls, value: str) -> str:
        """Reject whitespace-only prefixes (Copilot MED follow-up, PR #98).

        ``min_length=1`` blocks the literal empty string but lets a whitespace
        prefix like ``"  /"`` slip through, which would also list the bucket
        root once ``bucket.list_blobs`` strips it. Strip + non-empty check is
        the only safe gate.
        """
        if not value.strip():
            msg = (
                "cloud.weight_update.gcs_artifact_prefix must be a non-blank string; "
                "an empty / whitespace prefix would publish every artifact in the "
                "training bucket to HF Hub. Set it to e.g. 'trained/' or a "
                "fleet-specific subpath."
            )
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _warn_on_default_world_model_repo(self) -> WeightUpdatePollConfig:
        """Warn when ``world_model_enabled=True`` but the repo is left at default.

        The default ``world_model_repo_id`` is the maintainer's personal HF
        Hub repo. An operator who flips ``world_model_enabled`` without
        explicitly overriding the repo + filename would silently OTA-deploy
        weights from that repo into production — a footgun the validator
        surfaces at config-load time rather than after the first poll cycle.
        The validator only logs; it does NOT raise, so operators who *intend*
        to consume the default repo (the maintainer themselves, e2e tests)
        keep working.

        Returns:
            The unchanged instance.
        """
        if self.world_model_enabled and self.world_model_repo_id == _WORLD_MODEL_DEFAULT_REPO_ID:
            # Local import — avoid circular-import risk during settings build.
            from mousedroid.logging.setup import get_logger

            _log = get_logger(__name__)
            _log.warning(
                "world_model_poller_using_default_repo",
                repo_id=self.world_model_repo_id,
                hint=(
                    "Set ``cloud.weight_update.world_model_repo_id`` to your "
                    "fleet's HF Hub repo to avoid silently OTA-deploying "
                    "from the maintainer's personal repo."
                ),
            )
        return self


class CloudConfig(BaseModel):
    """Tier C1 cloud retraining loop umbrella configuration.

    Owns the OTA weight-update poller block. Orthogonal to :class:`GCPConfig`
    which covers the Pub/Sub / GCS data pipeline.
    """

    weight_update: WeightUpdatePollConfig = Field(
        default_factory=_settings_default_factory(WeightUpdatePollConfig),
        description="HuggingFace Hub OTA weight-update poller configuration.",
    )
