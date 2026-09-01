"""Telemetry and observability configuration models.

The WiFi/Ethernet telemetry server (REST + WebSocket), bearer-token auth,
Prometheus metrics export, and the training-side experiment logger.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, field_validator

from mousedroid.config.schema._primitives import StrictBaseModel


class TelemetryAuthConfig(StrictBaseModel):
    """Bearer token authentication configuration for the telemetry server.

    When enabled, requires a valid ``Authorization: Bearer <token>`` header
    on all requests except those matching ``exempt_paths``. The token value
    is read from the environment variable named by ``token_env_var``.
    """

    auth_enabled: bool = Field(False, description="Enable bearer token authentication")
    token_env_var: str = Field(
        "MOUSEDROID_TELEMETRY_TOKEN",
        description="Environment variable name containing the bearer token",
    )
    allowed_origins: list[str] = Field(
        default_factory=list,
        description="CORS allowed origins for auth middleware (empty=unrestricted)",
    )
    exempt_paths: list[str] = Field(
        default_factory=lambda: ["/health", "/metrics"],
        description=(
            "Paths that bypass authentication. Each entry must start with '/' "
            "and contain only lowercase letters, digits, hyphens, underscores, "
            "and forward slashes."
        ),
    )

    @field_validator("exempt_paths")
    @classmethod
    def _validate_exempt_paths(cls, paths: list[str]) -> list[str]:
        """Reject paths with traversal components, query strings, or unusual chars.

        Also reject (a) trailing slashes on non-root entries and (b)
        empty segments (``//``). The middleware uses segment-exact
        matching so ``/health/`` and ``/health`` would be different
        exemptions — silently accepting a trailing slash would make
        operator misconfigurations invisible. Addresses Gemini /
        Copilot review (PR #78).

        Prevents config typos that could widen the exemption surface (e.g.
        '/healthz' unintentionally exempting '/health') from going unnoticed.
        """
        import re

        exempt_re = re.compile(r"^/[a-z0-9_/\-]*$")
        for path in paths:
            if not exempt_re.match(path):
                raise ValueError(
                    f"exempt_paths entry {path!r} is invalid: must start with '/' "
                    "and contain only [a-z0-9_/-] (no query strings, no '..')"
                )
            if len(path) > 1 and path.endswith("/"):
                raise ValueError(
                    f"exempt_paths entry {path!r} must not have a trailing slash "
                    "(non-root). Use '/health' rather than '/health/'."
                )
            if "//" in path:
                raise ValueError(
                    f"exempt_paths entry {path!r} contains an empty segment ('//'). "
                    "Use single-slash boundaries only."
                )
        return paths


class TelemetryConfig(StrictBaseModel):
    """WiFi/Ethernet telemetry server configuration for remote monitoring.

    When enabled, exposes REST and WebSocket endpoints for real-time
    sensor data, log streaming, and health metrics. Binds to all
    network interfaces by default (WiFi + Ethernet + localhost).
    """

    enabled: bool = Field(False, description="Enable telemetry server")
    force_real_server: bool = Field(
        False,
        description=(
            "When True, use the real aiohttp TelemetryServer even with "
            "mock_hardware=True. Useful for local dashboard validation."
        ),
    )
    raw_frame_hz: float = Field(
        10.0,
        gt=0,
        le=60,
        description=(
            "Target frame rate (Hz) for the /camera/stream MJPEG endpoint "
            "when the camera driver supports raw-frame capture."
        ),
    )
    vision_feature_max_samples: int = Field(
        256,
        gt=0,
        le=4096,
        description=(
            "Maximum number of vision-feature samples encoded into each "
            "TelemetryFrame.vision_features payload. Larger feature "
            "vectors are uniformly strided down to this size before "
            "serialisation, keeping dashboard bandwidth bounded."
        ),
    )
    host: str = Field(
        "0.0.0.0",  # noqa: S104 — intentional all-interfaces default for the rover WiFi dashboard
        description="Server bind address (0.0.0.0 = all interfaces)",
    )
    port: int = Field(8080, gt=0, le=65535, description="Server port")
    port_discovery_strategy: Literal["fixed", "fallback_range", "kernel_assigned"] = Field(
        "fixed",
        description=(
            "Port binding strategy. 'fixed': bind exactly to port (raises on conflict). "
            "'fallback_range': try port, port+1, ..., port+port_discovery_max_attempts. "
            "'kernel_assigned': bind to port 0 and let the OS assign a free port."
        ),
    )
    port_discovery_max_attempts: int = Field(
        10,
        gt=0,
        le=100,
        description=(
            "Number of consecutive ports to try when port_discovery_strategy='fallback_range'."
        ),
    )
    preferred_interface: str | None = Field(
        None,
        description=(
            "[Reserved] Preferred network interface for mDNS (e.g. wlan0, eth0) — "
            "not wired to runtime"
        ),
    )
    ws_path: str = Field("/ws", description="WebSocket endpoint path")
    api_prefix: str = Field("/api/v1", description="REST API prefix")
    publish_hz: float = Field(
        10.0,
        gt=0,
        le=60,
        description="Telemetry publish rate (Hz)",
    )
    max_clients: int = Field(10, gt=0, description="Maximum concurrent WebSocket clients")
    queue_size: int = Field(64, gt=0, description="Internal publish queue depth")
    serialization: Literal["json", "msgpack"] = Field(
        "json",
        description="WebSocket serialization format",
    )
    api_key: SecretStr | None = Field(
        None,
        min_length=1,
        description=(
            "Optional API key (None=disabled). An empty string is rejected rather "
            "than silently accepted as a real key — the legacy X-API-Key middleware "
            "(telemetry/server/_lifecycle.py) would otherwise treat an empty "
            "configured key as matching an unauthenticated request's empty header."
        ),
    )
    mdns_enabled: bool = Field(True, description="Enable mDNS/Zeroconf discovery")
    mdns_service_name: str = Field(
        "MouseDroid Telemetry",
        description="mDNS service display name",
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins",
    )
    log_stream_buffer: int = Field(200, gt=0, description="Ring buffer size for log entries")
    metrics_path: str = Field(
        "/metrics",
        description=(
            "Legacy scrape endpoint path for direct TelemetryServer construction. "
            "Settings.metrics.path is the canonical configuration source."
        ),
    )
    auth: TelemetryAuthConfig | None = Field(
        None,
        description="Bearer token authentication config (None=disabled)",
    )

    # ------------------------------------------------------------------
    # PR #4: live streaming, mock visibility, serialization negotiation,
    # mDNS readiness, and sensor-liveness fields. All optional, all with
    # safe defaults so existing YAML files load unchanged.
    # ------------------------------------------------------------------
    lidar_raw_publish_hz: float = Field(
        5.0,
        gt=0,
        le=30,
        description=(
            "Target broadcast rate (Hz) for the /ws/v1/lidar/raw WebSocket "
            "stream. The LD19 driver runs at ~10 Hz natively; this rate "
            "downsamples on the server side before fan-out to clients."
        ),
    )
    lidar_raw_queue_size: int = Field(
        16,
        gt=0,
        le=1024,
        description=(
            "Internal queue depth for raw LiDAR scan publishing. When the "
            "queue is full new scans are dropped (non-blocking) to keep the "
            "control loop responsive."
        ),
    )
    lidar_raw_ws_path: str = Field(
        "/ws/v1/lidar/raw",
        description=(
            "WebSocket path for the raw LiDAR scan stream. Versioned so "
            "future protocol breaks land on /ws/v2/* without breaking "
            "existing dashboards."
        ),
    )
    mock_force_real_when_enabled: bool = Field(
        True,
        description=(
            "When True (default) and mock_hardware=True, the factory still "
            "builds the real aiohttp TelemetryServer on localhost instead of "
            "the no-op MockTelemetryServer so the dashboard can be exercised "
            "locally. Existing tests that construct MockTelemetryServer "
            "directly remain unaffected; the legacy force_real_server flag "
            "still wins when set explicitly."
        ),
    )
    mock_telemetry_source_enabled: bool = Field(
        True,
        description=(
            "When True and mock_hardware=True, factory wires a "
            "MockTelemetrySource that synthesises plausible scan + camera "
            "data into the publisher so the dashboard renders meaningful "
            "patterns without a real rover attached."
        ),
    )
    msgpack_client_lib_url: str = Field(
        "https://github.com/msgpack/msgpack-javascript",
        description=(
            "Public URL pointing to a msgpack JS decoder. Surfaced in the "
            "dashboard error banner when the server is configured for "
            "msgpack but the connecting client lacks a decoder."
        ),
    )
    mdns_register_timeout_s: float = Field(
        5.0,
        gt=0,
        le=60,
        description=(
            "Maximum time TelemetryServer.start() waits for the mDNS "
            "register call (in a thread pool) to complete or fail before "
            "continuing startup. Timeout is non-fatal: server keeps "
            "running, mDNS becomes best-effort and the failure is "
            "recorded via FailureRecorder."
        ),
    )
    ws_protocol_version: int = Field(
        1,
        ge=1,
        le=99,
        description=(
            "Server-side WebSocket protocol version advertised in the "
            "handshake hello-ack. Clients should send their accepted "
            "versions in the hello message."
        ),
    )
    ws_handshake_timeout_s: float = Field(
        2.0,
        gt=0,
        le=30,
        description=(
            "Maximum time to wait for the optional client hello negotiation "
            "message before falling back to the server-configured "
            "serialization. Keeps the path backwards-compatible with "
            "non-negotiating clients."
        ),
    )
    sensor_liveness_stale_s: float = Field(
        2.0,
        gt=0,
        le=60,
        description=(
            "Age threshold (seconds) above which a sensor's data is "
            "reported as 'stale' rather than 'live' in the liveness map. "
            "Tune per deployment based on the slowest sensor's expected "
            "update rate."
        ),
    )


class MetricsConfig(StrictBaseModel):
    """Prometheus-compatible metrics export configuration.

    Controls metrics endpoint enablement, naming, and scrape path.  All
    metric names are derived from ``namespace`` so nothing is hardcoded
    outside this class.
    """

    enabled: bool = Field(True, description="Enable /metrics endpoint")
    path: str = Field("/metrics", description="HTTP path for Prometheus scrape endpoint")
    namespace: str = Field(
        "mousedroid",
        description="Prefix applied to all metric names (e.g. mousedroid_loop_time_ms)",
    )
    export_interval_s: float = Field(
        10.0, gt=0, description="[Reserved] Background export interval (s) — not wired to runtime"
    )
    # Individual metric enable/disable toggles (all default-on)
    track_loop_time: bool = Field(True, description="Expose loop_time_ms gauge")
    track_battery: bool = Field(True, description="Expose battery_voltage_v gauge")
    track_ws_clients: bool = Field(True, description="Expose ws_client_count gauge")
    track_frame_drops: bool = Field(True, description="Expose frame_drop_total counter")
    track_safety_violations: bool = Field(
        True, description="Expose safety_violations_total counter"
    )
    track_gpu_temp: bool = Field(True, description="Expose gpu_temp_celsius gauge")
    track_llm_translations: bool = Field(
        True,
        description="Expose llm_translation counters and latency histogram",
    )
    track_lidar: bool = Field(
        True,
        description=(
            "Expose lidar_sector_distance_m (labeled), lidar_min_distance_m, "
            "and lidar_scan_points gauges"
        ),
    )
    track_memory_tier: bool = Field(True, description="Expose memory tier gauges")
    track_voice_events: bool = Field(True, description="Expose voice event counter")
    track_llm_latency: bool = Field(True, description="Expose LLM mission parse latency")
    track_llm_gateway: bool = Field(
        True,
        description=(
            "Expose deliberative LLM-gateway observability for the Anthropic "
            "Claude tier: token-usage counter (labels: model, token_type), "
            "round-trip latency histogram, per-tier served counter (labels: "
            "tier, outcome), and a latency-budget-exceeded counter (label: "
            "model). Emitted only when the gateway runs with a MetricsRegistry "
            "and actually translates — safe to leave on."
        ),
    )
    track_curiosity: bool = Field(True, description="Expose curiosity intrinsic reward gauge")
    track_sensor_recovery: bool = Field(True, description="Expose sensor recovery counter")
    track_cloud: bool = Field(
        True,
        description=(
            "Expose cloud digital twin metrics: publish counters, publish "
            "latency histogram, circuit breaker state, and experience "
            "export backlog gauges. Emitted only when a cloud sink is "
            "actually wired into the orchestrator — safe to leave on."
        ),
    )
    track_mcp: bool = Field(
        True,
        description=(
            "Expose MCP server metrics: request counter, per-tool call "
            "counter (label: tool, result), and request latency histogram. "
            "Emitted only when the MCP server is actually built — safe to "
            "leave on."
        ),
    )
    track_openclaw_memory: bool = Field(
        True,
        description=(
            "Expose MCP memory query latency histogram for OpenClaw integration. Safe to leave on."
        ),
    )
    track_on_device_learning: bool = Field(
        True,
        description=(
            "Expose the Phase-6 on-device-learning revert counter "
            "(label: reason). Pure-add: omitted from /metrics until the first "
            "revert, so default deployments render byte-identically. Safe to "
            "leave on."
        ),
    )
    track_growth_distillation: bool = Field(
        True,
        description=(
            "Expose the growth-pillar distillation counter (label: outcome). "
            "Pure-add: omitted from /metrics until the first distillation cycle, "
            "so default deployments render byte-identically. Safe to leave on."
        ),
    )
    track_voice_degradation: bool = Field(
        True,
        description=(
            "Expose the voice-subsystem degradation counters: "
            "``voice_speaker_degraded_total`` (label: subsystem — the USB "
            "speaker exhausted its reconnect retries or the engine fell back "
            "to a MockSpeaker) and ``voice_tts_synthesize_failures_total`` "
            "(label: api — a Piper synthesis call raised). Pure-add: each "
            "family is omitted from /metrics until its first increment, so "
            "default deployments render byte-identically. Safe to leave on."
        ),
    )
    loop_latency_buckets_ms: tuple[float, ...] = Field(
        (1.0, 2.5, 5.0, 10.0, 20.0, 33.0, 50.0, 100.0, 200.0, float("inf")),
        description="Histogram bucket boundaries for control-loop latency (ms)",
    )
    llm_latency_buckets_ms: tuple[float, ...] = Field(
        (25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, float("inf")),
        description="Histogram bucket boundaries for LLM translation latency (ms)",
    )
    llm_gateway_latency_buckets_ms: tuple[float, ...] = Field(
        (50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, float("inf")),
        description=(
            "Histogram bucket boundaries for cloud LLM-gateway round-trip "
            "latency (ms). Wider than llm_latency_buckets_ms because cloud "
            "Claude round-trips are seconds, not ms. The 500 ms default "
            "latency_target_ms and the 5000 ms cloud-pilot overlay value "
            "(config/jetson_claude_pilot.yaml) both land on bucket boundaries."
        ),
    )
    mcp_latency_buckets_ms: tuple[float, ...] = Field(
        (5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0, float("inf")),
        description="Histogram bucket boundaries for MCP request latency (ms)",
    )
    mcp_memory_query_latency_buckets_ms: tuple[float, ...] = Field(
        (5.0, 10.0, 25.0, 50.0, 100.0, 150.0, 250.0, 500.0, 1000.0, float("inf")),
        description="Histogram bucket boundaries for MCP memory query latency (ms)",
    )
    vla_inference_seconds_buckets: tuple[float, ...] = Field(
        (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, float("inf")),
        description=(
            "Histogram bucket boundaries for VLA policy inference latency (seconds). "
            "Phase 3b: covers the 30 Hz orchestrator budget (~33 ms) up to long-tail "
            "fallbacks beyond 1 s. Operator-tunable per deployment."
        ),
    )
    world_model_observe_step_seconds_buckets: tuple[float, ...] = Field(
        (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, float("inf")),
        description=(
            "Histogram bucket boundaries for DualStreamRSSM.observe_step latency "
            "(seconds). Default envelope covers <1 ms baseline up to long-tail "
            "PyTorch ticks beyond 100 ms. The 10 ms target on Orin Nano (with "
            "cfg.world_model.engine=onnx_trt + TensorRT EP) lands within the "
            "(0.005, 0.01] bucket; the portable dev gate is 33 ms (30 Hz tick). "
            "Operator-tunable per deployment."
        ),
    )
    cloud_weight_update_download_seconds_buckets: tuple[float, ...] = Field(
        (0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, float("inf")),
        description=(
            "Histogram bucket boundaries for OTA weight-update download latency "
            "(seconds). Tier C1: covers cellular-fleet downloads on the order of "
            "tens of MB. Operator-tunable per deployment."
        ),
    )
    mission_duration_seconds_buckets: tuple[float, ...] = Field(
        (1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, float("inf")),
        description=(
            "Histogram bucket boundaries for mission active duration (seconds). "
            "Tier C2 (C2.3): covers short single-objective missions (< 1 min) "
            "through multi-minute autonomous navigation runs (> 10 min)."
        ),
    )

    @field_validator(
        "loop_latency_buckets_ms",
        "llm_latency_buckets_ms",
        "llm_gateway_latency_buckets_ms",
        "mcp_latency_buckets_ms",
        "mcp_memory_query_latency_buckets_ms",
        "vla_inference_seconds_buckets",
        "world_model_observe_step_seconds_buckets",
        "cloud_weight_update_download_seconds_buckets",
        "mission_duration_seconds_buckets",
    )
    @classmethod
    def _validate_histogram_buckets(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Enforce monotonically ascending, strictly positive bucket boundaries.

        A trailing ``float("inf")`` sentinel is permitted (and conventional for
        Prometheus histograms) but not required — the registry appends one at
        runtime if missing. When present, ``float("inf")`` MUST be the last
        element; an ``inf`` in any other position would yield surprising bucket
        cardinality after the runtime ``sorted(...)`` call in ``MetricsRegistry``.
        Negative, zero, or duplicate boundaries would silently corrupt bucket
        accumulation, so they're rejected at schema-load time.
        """
        inf = float("inf")
        if not value:
            msg = "histogram bucket tuple must be non-empty"
            raise ValueError(msg)
        # Reject ``inf`` anywhere except the trailing position.
        inf_positions = [i for i, b in enumerate(value) if b == inf]
        if inf_positions and inf_positions != [len(value) - 1]:
            msg = (
                f"histogram bucket boundaries may only contain +inf as the "
                f"trailing sentinel; got {value!r}"
            )
            raise ValueError(msg)
        finite = [b for b in value if b != inf]
        if finite != sorted(finite):
            msg = f"histogram bucket boundaries must be monotonically ascending; got {value!r}"
            raise ValueError(msg)
        if any(b <= 0.0 for b in finite):
            msg = f"histogram bucket boundaries must be strictly positive; got {value!r}"
            raise ValueError(msg)
        if len(set(finite)) != len(finite):
            msg = f"histogram bucket boundaries must be unique (no duplicates); got {value!r}"
            raise ValueError(msg)
        return value


# sqlite tracking URIs that name no database file. SQLAlchemy resolves each
# to an in-memory store, so every run written to one is discarded at process
# exit with no error. Compared against the lower-cased, stripped value.
# ``sqlite:///:memory:`` is deliberately absent -- that spelling is explicit
# and legitimate. Mirrors the factory's ``_IN_MEMORY_SQLITE_PATHS``, which
# encodes the same fact one layer down as path components rather than URIs.
_EPHEMERAL_SQLITE_URIS: frozenset[str] = frozenset({"sqlite://", "sqlite:///"})


class ExperimentLoggerConfig(StrictBaseModel):
    """Experiment-logger configuration for training runs (per-step + per-phase metrics).

    Wired into :class:`PipelineOrchestrator` and :class:`OfflineRLTrainer`
    via :func:`mousedroid.factory.build_experiment_logger`. Defaults to OFF
    (``backend="none"``) so a YAML predating this feature loads unchanged
    (CLAUDE.md invariant #6). Selecting ``backend="mlflow"`` requires the
    ``mousedroid[mlflow]`` extras (``mlflow-skinny`` plus ``sqlalchemy`` and
    ``alembic``, which mlflow's default sqlite tracking store needs even
    with the skinny client); a missing dep degrades gracefully to the NoOp
    logger with a structured warning.
    """

    backend: Literal["none", "mlflow"] = Field(
        "none",
        description=(
            "Experiment-logger backend. ``none`` (default) selects the NoOp "
            "logger — byte-identical to pre-feature behavior. ``mlflow`` "
            "selects the MlflowClient-backed logger writing to "
            "``tracking_uri`` (default ``sqlite:///mlflow.db``)."
        ),
    )
    tracking_uri: str = Field(
        "sqlite:///mlflow.db",
        min_length=1,
        description=(
            "MLflow tracking URI. ``sqlite:///mlflow.db`` (default) writes "
            "to a local SQLite database — mlflow's own recommended local "
            "backend. mlflow 3.x rejects the plain file-store backend "
            "unless ``MLFLOW_ALLOW_FILE_STORE=true`` is set, which "
            "``mlflow_logger.py`` does unconditionally at import, so the "
            "``file:`` store still works; defaulting to sqlite removes the "
            "dependence on that escape hatch rather than unblocking "
            "anything. Set a ``file:`` URI to use the legacy directory-tree "
            "store, or ``http://host:port`` for a remote tracking server. "
            "Both local schemes (``file:`` and ``sqlite:///``) have a "
            "relative path pinned to an absolute one at factory resolution "
            "time, so the effective location is reported in the "
            "``experiment_logger_tracking_uri_resolved`` log event; remote "
            "and in-memory sqlite URIs pass through unchanged. A relative "
            "path is still resolved against whichever directory the process "
            "was launched from — set an absolute URI to make the database "
            "location independent of that (artifacts have a separate root, "
            "which mlflow keeps at ``./mlruns`` regardless). This is a plain "
            "str, not a SecretStr, so a credentialed remote URI has its "
            "userinfo redacted before reaching any log event."
        ),
    )

    @field_validator("tracking_uri")
    @classmethod
    def _reject_blank_or_in_memory_tracking_uri(cls, value: str) -> str:
        """Reject URIs that silently discard every metric written to them.

        Two values validate as plain non-empty strings but make the logger
        a silent black hole rather than failing loudly, which is worse than
        either working or crashing:

        * whitespace-only / empty — mlflow falls back to its own ambient
          default (or ``MLFLOW_TRACKING_URI``), so runs land somewhere the
          operator never configured.
        * ``sqlite://`` (two slashes) and ``sqlite:///`` (three, but an
          empty path) — both are SQLAlchemy's in-memory database. Verified
          directly rather than reasoned about: ``PRAGMA database_list``
          reports an empty file for each, so every run is written to RAM
          and discarded at process exit with no error at any point.
          ``sqlite:///`` matters *more* than the two-slash form, because it
          is what an operator lands on after following this validator's own
          advice to "use three slashes" and forgetting the filename. The
          factory already classifies it as in-memory
          (``_IN_MEMORY_SQLITE_PATHS`` contains ``""``); accepting it here
          would leave the schema contradicting the factory.

        ``sqlite:///:memory:`` is deliberately still allowed: it is the
        explicit, unambiguous way to ask for an in-memory store (tests do),
        whereas the two bare forms are nearly always a truncated path.

        Returns the *stripped* value, which is load-bearing rather than
        cosmetic: a leading space defeats the scheme match in
        ``_resolve_tracking_uri`` entirely, so ``" sqlite:///mlflow.db"``
        would silently skip pinning.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "tracking_uri must not be empty or whitespace-only; use an "
                "explicit URI such as 'sqlite:///mlflow.db' (the default)"
            )
        if stripped.lower() in _EPHEMERAL_SQLITE_URIS:
            raise ValueError(
                f"tracking_uri {stripped!r} is SQLAlchemy's in-memory database — "
                "every logged run would be silently discarded at process exit. "
                "Use 'sqlite:///mlflow.db' (three slashes AND a filename) or, "
                "if an in-memory store is genuinely intended, the explicit "
                "'sqlite:///:memory:'"
            )
        return stripped

    experiment_name: str = Field(
        "mousedroid",
        min_length=1,
        description="MLflow experiment name (created if missing).",
    )
    run_name: str | None = Field(
        None,
        description=(
            "Optional human-readable run name for the parent (pipeline) run. "
            "When ``None`` the logger falls back to its configured default "
            '(this field) or the ``"pipeline"`` sentinel.'
        ),
    )
    log_step_every_n: int = Field(
        1,
        gt=0,
        description=(
            "Per-update-step metric throttle. ``1`` (default) logs every "
            "update_step call. Set higher for very-long training runs to "
            "reduce store-write overhead."
        ),
    )
    log_artifacts: bool = Field(
        True,
        description=(
            "When True, the orchestrator logs the resolved Settings JSON "
            "snapshot as a parent-run artifact at start, plus the per-phase "
            "checkpoint file as a child-run artifact on phase completion."
        ),
    )


class ObservabilityConfig(StrictBaseModel):
    """Top-level observability configuration for the training stack.

    Currently contains the experiment-logger sub-config; future fields
    (training-side Prometheus metrics, W&B integration, etc.) land here
    to keep ``Settings`` flat.
    """

    experiment_logger: ExperimentLoggerConfig = Field(
        default_factory=ExperimentLoggerConfig,
        description="Per-run experiment-logger config (MLflow file backend).",
    )
