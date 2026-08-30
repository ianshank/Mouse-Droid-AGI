"""Factory builders — metrics, experiment logging, weight-update, telemetry.

Metrics registry, experiment logger, weight-update pollers, failure recorder,
telemetry publisher/server.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mousedroid.cloud.protocol import (
    ENGINE_TYPE_POLICY,
    ENGINE_TYPE_WORLD_MODEL,
)
from mousedroid.hardware.protocols import VisionProtocol
from mousedroid.logging.redaction import redact_uri_credentials, redact_uris_in_text
from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.cloud.protocol import (
        PendingWeightUpdate,
        WeightUpdatePollerProtocol,
    )
    from mousedroid.config.schema import (
        Settings,
    )
    from mousedroid.health.monitor import HealthMonitor
    from mousedroid.learning.on_device.hot_swap import OnDeviceWeightUpdateSource
    from mousedroid.orchestrator.mission_dispatcher import MissionDispatcherProtocol
    from mousedroid.telemetry.failure_recorder import FailureRecorder
    from mousedroid.telemetry.log_buffer import LogRingBuffer
    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.telemetry.protocol import TelemetryPublisherProtocol, TelemetryServerProtocol
    from mousedroid.training.observability import ExperimentLoggerProtocol

_log = get_logger(__name__)

_PINNED_URI_SCHEMES: tuple[tuple[str, str], ...] = (
    ("file:", "file:"),
    ("sqlite:///", "sqlite:///"),
)

_IN_MEMORY_SQLITE_PATHS: frozenset[str] = frozenset({"", ":memory:"})


def build_metrics_registry(cfg: Settings) -> MetricsRegistry | None:
    """Build the shared Prometheus metrics registry when metrics are enabled.

    Args:
        cfg: Root settings.

    Returns:
        Shared ``MetricsRegistry`` or ``None`` when metrics are disabled.
    """
    if not cfg.metrics.enabled:
        return None

    from mousedroid.telemetry.metrics import MetricsRegistry

    return MetricsRegistry(cfg.metrics)


def build_experiment_logger(cfg: Settings) -> ExperimentLoggerProtocol:
    """Build the shared experiment logger for training pipelines.

    Mirrors :func:`build_metrics_registry`'s shape: returns a NEVER-None
    protocol type so callers can drop the ``logger is not None`` guard.
    The NoOp implementation is the default and is byte-identically a no-op,
    so threading the logger through the orchestrator/trainer is free when
    observability is disabled.

    Resolution order:

    1. ``cfg.observability is None`` (the pre-feature default) →
       :class:`NoOpExperimentLogger`.
    2. ``cfg.observability.experiment_logger.backend == "none"`` →
       :class:`NoOpExperimentLogger`.
    3. ``cfg.observability.experiment_logger.backend == "mlflow"`` AND
       the ``[mlflow]`` extras are installed →
       :class:`MlflowExperimentLogger`.
    4. ``cfg.observability.experiment_logger.backend == "mlflow"`` AND
       ``mlflow-skinny`` is NOT installed →
       :class:`NoOpExperimentLogger` (with a structured warning, so
       operators see the misconfiguration without crashing the run).

    Relative local-store URIs — both ``file:`` and the default ``sqlite:///``
    — are pinned to an absolute path at build time by
    :func:`_resolve_tracking_uri` (see there for the precise, narrower-than-
    obvious guarantee that provides). The resolved URI is logged as
    ``experiment_logger_tracking_uri_resolved`` before construction, so a
    later store failure can be traced to a concrete path.

    Args:
        cfg: Root settings.

    Returns:
        A logger conforming to :class:`ExperimentLoggerProtocol`.
    """
    from mousedroid.training.observability import NoOpExperimentLogger

    if cfg.observability is None:
        return NoOpExperimentLogger()
    logger_cfg = cfg.observability.experiment_logger
    if logger_cfg.backend == "none":
        return NoOpExperimentLogger()

    if logger_cfg.backend == "mlflow":
        try:
            from mousedroid.training.observability.mlflow_logger import (
                MlflowExperimentLogger,
            )

            tracking_uri = _resolve_tracking_uri(logger_cfg.tracking_uri)
            # Emitted BEFORE construction so it survives an init failure: the
            # store's own errors ("unable to open database file", "file is not
            # a database") name no path, so without this an operator seeing
            # "no runs visible" cannot tell where the backend actually points.
            #
            # Redacted because tracking_uri is a plain str (not SecretStr) and
            # a remote store is legitimately spelled with inline credentials --
            # ``http://user:password@host:5000``. Redaction masks only the
            # userinfo component, so the scheme/host/path this event exists to
            # report survive intact; the local sqlite/file default is
            # byte-identical to its input.
            _log.info(
                "experiment_logger_tracking_uri_resolved",
                configured_uri=redact_uri_credentials(logger_cfg.tracking_uri),
                resolved_uri=redact_uri_credentials(tracking_uri),
                experiment_name=logger_cfg.experiment_name,
            )
            return MlflowExperimentLogger(
                tracking_uri=tracking_uri,
                experiment_name=logger_cfg.experiment_name,
                run_name=logger_cfg.run_name,
            )
        except ImportError as exc:
            _log.warning(
                "experiment_logger_mlflow_extras_missing",
                configured_uri=redact_uri_credentials(logger_cfg.tracking_uri),
                error_type=type(exc).__name__,
                # Redacted too, not just the URI field: mlflow's
                # UnsupportedModelRegistryStoreURIException quotes the
                # offending tracking URI back verbatim, password included.
                error=redact_uris_in_text(str(exc)),
            )
            return NoOpExperimentLogger()
        except Exception as exc:
            # Construction can also fail on a bad tracking_uri, an unreachable
            # store, or permission errors. Degrade to NoOp (with a distinct
            # warning) rather than crashing the whole training run — observability
            # is best-effort, never load-bearing. The URI is included because the
            # underlying store exceptions do not carry it.
            _log.warning(
                "experiment_logger_mlflow_init_failed",
                configured_uri=redact_uri_credentials(logger_cfg.tracking_uri),
                error_type=type(exc).__name__,
                # See the sibling branch: store exceptions echo the URI.
                error=redact_uris_in_text(str(exc)),
            )
            return NoOpExperimentLogger()

    # Exhaustive Literal coverage; reached only on schema additions without a
    # corresponding factory branch.
    _log.warning(
        "experiment_logger_unknown_backend",
        backend=logger_cfg.backend,
    )
    return NoOpExperimentLogger()


def _resolve_tracking_uri(raw: str) -> str:
    """Pin a relative local-filesystem tracking URI to an absolute path.

    Covers both local backends: ``file:`` (the legacy directory-tree store)
    and ``sqlite:///`` (the default, see ``ExperimentLoggerConfig``).

    What the pin does and does not guarantee, stated precisely because the
    obvious stronger claim is false (both halves verified directly, not
    assumed): it does **not** make a relative URI working-directory
    independent — a process launched from a different directory still pins
    against its own CWD, and mlflow caches its store per URI string per
    process so an in-process ``chdir()`` is invisible either way. What it
    guarantees is that the *effective* path is absolute and therefore
    reportable, which is what ``experiment_logger_tracking_uri_resolved``
    logs so an operator can tell which database a run actually reached. The
    operator-facing cure for a split across launch directories is an
    absolute ``tracking_uri``; see ``docs/runbooks/mlflow-local-ui.md``.

    Remote URIs (``http``, ``https``, ``databricks``) and in-memory sqlite
    URIs (``sqlite://``, ``sqlite:///:memory:``) pass through unchanged, as
    does any URI whose path is already absolute.

    Scheme matching is case-insensitive: URI schemes are case-insensitive
    per RFC 3986 and mlflow lowercases them via ``urlparse`` before
    dispatching, so a configured ``FILE:./mlruns`` still selects the file
    store downstream — matching case-sensitively here would silently skip
    the pin for exactly those URIs.

    Args:
        raw: Tracking URI exactly as configured.

    Returns:
        The URI with a relative local path pinned absolute, else ``raw``.
    """
    lowered = raw.lower()
    for scheme, rebuilt_prefix in _PINNED_URI_SCHEMES:
        if not lowered.startswith(scheme):
            continue
        path_part = raw[len(scheme) :]
        if scheme.startswith("sqlite") and path_part in _IN_MEMORY_SQLITE_PATHS:
            return raw
        return f"{rebuilt_prefix}{Path(path_part).resolve()}"
    return raw


def build_weight_update_pollers(
    cfg: Settings,
    *,
    metrics: MetricsRegistry | None = None,
) -> Mapping[str, WeightUpdatePollerProtocol]:
    """Build a mapping of ``engine_type`` -> poller (Tier C1.2).

    Returns ``{}`` when ``cfg.cloud.weight_update.poll_interval_s <= 0.0``,
    preserving byte-identical pre-Tier-C1 behaviour. Always includes the
    ``"policy"`` entry when polling is enabled; includes ``"world_model"``
    only when ``cfg.cloud.weight_update.world_model_enabled is True``.

    Dict insertion order is ``policy`` -> ``world_model`` so the orchestrator
    consumes pending updates deterministically.

    Args:
        cfg: Root settings.
        metrics: Shared metrics registry; forwarded to each poller for
            download / mismatch / latency observability.

    Returns:
        Mapping from ``engine_type`` to its
        :class:`WeightUpdatePollerProtocol` implementation. Empty when OTA
        polling is disabled. Insertion order is guaranteed to be ``policy``
        before ``world_model`` so the orchestrator consumes updates
        deterministically.
    """
    if cfg.cloud.weight_update.poll_interval_s <= 0.0:
        return {}

    from pathlib import Path as _Path

    from mousedroid.cloud.weight_update_poller import HuggingFaceWeightUpdatePoller

    # Per-engine cache subdirectory layout (Copilot MED): both pollers
    # download a ``sha256.txt`` manifest into their cache dir each cycle.
    # When both share the same root, the world-model poller's manifest
    # writer races the policy poller's writer and produces spurious
    # mismatches. Giving each poller a per-engine subdir under the
    # configured root preserves the operator-facing
    # ``cfg.cloud.weight_update.cache_dir`` knob (root unchanged) while
    # eliminating the collision. The subdir name reuses the typed
    # ``EngineType`` literal so a future engine addition only needs to
    # extend the enum.
    cache_root = _Path(cfg.cloud.weight_update.cache_dir)
    pollers: dict[str, WeightUpdatePollerProtocol] = {
        ENGINE_TYPE_POLICY: HuggingFaceWeightUpdatePoller(
            cfg.cloud.weight_update,
            repo_id=cfg.cloud.weight_update.policy_repo_id,
            filename=cfg.cloud.weight_update.policy_filename,
            engine_type=ENGINE_TYPE_POLICY,
            metrics=metrics,
            cache_dir_override=cache_root / ENGINE_TYPE_POLICY,
        ),
    }
    if cfg.cloud.weight_update.world_model_enabled:
        pollers[ENGINE_TYPE_WORLD_MODEL] = HuggingFaceWeightUpdatePoller(
            cfg.cloud.weight_update,
            repo_id=cfg.cloud.weight_update.world_model_repo_id,
            filename=cfg.cloud.weight_update.world_model_filename,
            engine_type=ENGINE_TYPE_WORLD_MODEL,
            metrics=metrics,
            cache_dir_override=cache_root / ENGINE_TYPE_WORLD_MODEL,
        )
    _log.info(
        "weight_update_pollers_built",
        engines=list(pollers.keys()),
        poll_interval_s=cfg.cloud.weight_update.poll_interval_s,
        cache_root=str(cache_root),
    )
    return pollers


def build_weight_update_loader(
    cfg: Settings,
) -> Callable[[PendingWeightUpdate], object] | None:
    """Build the optional Tier C1 OTA artifact loader.

    The loader is invoked by the orchestrator inside
    ``_apply_pending_weight_update`` to materialise the downloaded artifact
    into a live engine BEFORE the reference swap.

    Returns ``None`` when the OTA poller is disabled or when no production
    loader is wired (the test suite injects its own loader). When the
    poller IS enabled but the loader returns ``None``, the orchestrator
    emits ``cloud_weight_update_swap_skipped_no_loader`` and leaves the
    live model untouched — operators decide what to do.

    Returns:
        Callable that maps :class:`PendingWeightUpdate` to a new engine
        object, or ``None``.
    """
    if cfg.cloud.weight_update.poll_interval_s <= 0.0:
        return None
    # Production loader wiring is engine-specific (ONNX runtime / TensorRT
    # session reload). Tier C1 ships the seam; the operator pulls the
    # concrete loader through configuration in a follow-up PR.
    return None


def _compose_weight_update_loader(
    cloud_loader: Callable[[PendingWeightUpdate], object] | None,
    on_device_source: OnDeviceWeightUpdateSource,
) -> Callable[[PendingWeightUpdate], object]:
    """Compose the WS-E4 hot-loop loader: PURE return for on-device updates.

    The orchestrator invokes ``_weight_update_loader`` SYNCHRONOUSLY inside
    ``tick()``. For an on-device hot-swap update the engine was already
    constructed OFF the hot loop (in the source's ``refresh_once``), so the
    composed loader is a PURE reference return —
    :meth:`OnDeviceWeightUpdateSource.take_materialized` does NO I/O. Any other
    (cloud OTA) update is delegated to ``cloud_loader``; when no cloud loader is
    wired AND the update is not owned by the on-device source, the composed
    loader raises so the orchestrator's broad-except logs
    ``cloud_weight_update_swap_failed`` and leaves the live model untouched
    (fail-closed) rather than swapping in a bogus engine.

    Args:
        cloud_loader: The Tier C1 cloud OTA loader (or ``None`` when no cloud
            loader is wired — the default).
        on_device_source: The WS-E4 source that pre-materialised the engine.

    Returns:
        The composed ``(PendingWeightUpdate) -> engine`` loader.
    """

    def _loader(update: PendingWeightUpdate) -> object:
        if on_device_source.owns(update):
            # PURE reference return — the engine was built off the hot loop.
            return on_device_source.take_materialized(update)
        if cloud_loader is not None:
            return cloud_loader(update)
        msg = (
            "no loader wired for weight update "
            f"(engine_type={update.engine_type!r}, revision={update.revision!r})"
        )
        raise RuntimeError(msg)

    return _loader


def build_failure_recorder(
    cfg: Settings,
    metrics: MetricsRegistry | None = None,
) -> FailureRecorder:
    """Build a failure recorder wired to the given metrics registry.

    Returns a :class:`~mousedroid.telemetry.failure_recorder.PrometheusFailureRecorder`
    when *metrics* is non-None (telemetry enabled), otherwise a
    :class:`~mousedroid.telemetry.failure_recorder.NullFailureRecorder`.

    Args:
        cfg: Root settings (reserved for future per-subsystem gating).
        metrics: Shared metrics registry, or ``None`` when telemetry is
            disabled.  Pass the result of :func:`build_metrics_registry`.

    Returns:
        A ``FailureRecorder`` implementation appropriate for the environment.
    """
    from mousedroid.telemetry.failure_recorder import NullFailureRecorder, PrometheusFailureRecorder

    if metrics is not None:
        _log.debug("failure_recorder_built", backend="prometheus")
        return PrometheusFailureRecorder(metrics)

    _log.debug("failure_recorder_built", backend="null")
    return NullFailureRecorder()


def build_telemetry_publisher(cfg: Settings) -> TelemetryPublisherProtocol | None:
    """Build telemetry publisher if telemetry is enabled.

    Args:
        cfg: Root settings.

    Returns:
        ``TelemetryPublisher`` or ``None`` if telemetry disabled.
    """
    if not cfg.telemetry.enabled:
        return None
    from mousedroid.telemetry.publisher import TelemetryPublisher

    _log.info("telemetry_publisher_built", publish_hz=cfg.telemetry.publish_hz)
    return TelemetryPublisher(cfg.telemetry)


def build_telemetry_server(
    cfg: Settings,
    publisher: TelemetryPublisherProtocol | None,
    health_monitor: HealthMonitor,
    log_buffer: LogRingBuffer | None = None,
    metrics_registry: MetricsRegistry | None = None,
    camera: VisionProtocol | None = None,
    mission_dispatcher: MissionDispatcherProtocol | None = None,
) -> TelemetryServerProtocol | None:
    """Build telemetry server if telemetry is enabled.

    Args:
        cfg: Root settings.
        publisher: Telemetry publisher to consume frames from.
        health_monitor: Health monitor for health endpoint.
        log_buffer: Optional log ring buffer for log streaming.
        metrics_registry: Optional shared metrics registry reused by other runtime components.
        camera: Optional vision driver; used as a raw-frame source for
            the MJPEG ``/camera/stream`` endpoint when it also implements
            :class:`RawFrameSourceProtocol`.
        mission_dispatcher: Optional :class:`MissionDispatcherProtocol`.
            When supplied together with an enabled ``cfg.openclaw``, the
            ``POST /api/v1/mission`` endpoint is registered.

    Returns:
        ``TelemetryServer`` or ``None`` if telemetry disabled.
    """
    if not cfg.telemetry.enabled or publisher is None:
        return None

    # PR #4: when mock_hardware is on, prefer building the real
    # aiohttp server bound to localhost so the dashboard is exercisable
    # end-to-end without rover hardware. ``force_real_server=True``
    # (legacy override) still wins. ``mock_force_real_when_enabled=False``
    # restores the no-op MockTelemetryServer for tests that prefer it.
    if cfg.mock_hardware and not cfg.telemetry.force_real_server:
        if not cfg.telemetry.mock_force_real_when_enabled:
            from mousedroid.telemetry.mock_server import MockTelemetryServer

            _log.info("telemetry_mock_server_built")
            return MockTelemetryServer()
        _log.info("telemetry_real_server_in_mock_mode")

    # D4: validate bearer token is present in env when auth is enabled.
    auth_cfg = cfg.telemetry.auth
    if auth_cfg is not None and auth_cfg.auth_enabled:
        import os

        from mousedroid.telemetry.exceptions import TelemetryConfigError

        token = os.environ.get(auth_cfg.token_env_var, "")
        if not token:
            raise TelemetryConfigError(
                f"telemetry auth_enabled=True but ${auth_cfg.token_env_var} is unset or empty; "
                "export the token or set auth_enabled=False"
            )

    shared_metrics_registry = metrics_registry
    metrics_path = cfg.metrics.path
    telemetry_metrics_path_default = type(cfg.telemetry).model_fields["metrics_path"].default
    metrics_path_default = type(cfg.metrics).model_fields["path"].default
    if (
        metrics_path == metrics_path_default
        and cfg.telemetry.metrics_path != telemetry_metrics_path_default
    ):
        metrics_path = cfg.telemetry.metrics_path

    if shared_metrics_registry is None and cfg.metrics.enabled:
        shared_metrics_registry = build_metrics_registry(cfg)

    from mousedroid.hardware.protocols import RawFrameSourceProtocol
    from mousedroid.telemetry.server import TelemetryServer

    raw_frame_source: RawFrameSourceProtocol | None = None
    if camera is not None and isinstance(camera, RawFrameSourceProtocol):
        raw_frame_source = camera

    _log.info(
        "telemetry_server_built",
        host=cfg.telemetry.host,
        port=cfg.telemetry.port,
        raw_frame_source=raw_frame_source is not None,
    )
    failure_recorder = build_failure_recorder(cfg, shared_metrics_registry)
    # PR #4: wire the raw LiDAR queue when the publisher exposes one.
    # Older custom publishers without ``get_lidar_raw_queue`` keep
    # working — the server simply registers the raw route as 503.
    lidar_raw_queue = None
    get_raw_queue = getattr(publisher, "get_lidar_raw_queue", None)
    if callable(get_raw_queue):
        lidar_raw_queue = get_raw_queue()
    return TelemetryServer(
        cfg=cfg.telemetry,
        telemetry_queue=publisher.get_queue(),
        health_monitor=health_monitor,
        log_buffer=log_buffer,
        metrics_registry=shared_metrics_registry,
        metrics_path=metrics_path,
        publisher=publisher,
        lidar_max_range_m=cfg.lidar.max_range_m if cfg.lidar is not None else None,
        raw_frame_source=raw_frame_source,
        raw_frame_hz=cfg.telemetry.raw_frame_hz,
        cloud_enabled=cfg.gcp is not None,
        mission_dispatcher=mission_dispatcher,
        openclaw_cfg=cfg.openclaw,
        failure_recorder=failure_recorder,
        lidar_raw_queue=lidar_raw_queue,
    )


def build_mock_telemetry_source(
    cfg: Settings,
    publisher: TelemetryPublisherProtocol | None,
) -> Any:
    """Build a ``MockTelemetrySource`` when running in mock mode.

    Returns ``None`` when the source is disabled or when no publisher
    is available. The returned object exposes ``start()`` / ``stop()``
    coroutines so the orchestrator can manage its lifecycle alongside
    the telemetry server.

    Args:
        cfg: Root settings.
        publisher: Telemetry publisher to push synthetic payloads into.

    Returns:
        A ``MockTelemetrySource`` instance, or ``None`` if disabled.
    """
    if not cfg.mock_hardware:
        return None
    if not cfg.telemetry.enabled:
        return None
    if publisher is None:
        return None
    if not cfg.telemetry.mock_telemetry_source_enabled:
        return None

    from mousedroid.telemetry.mock_source import MockTelemetrySource

    source = MockTelemetrySource(cfg.telemetry, publisher)
    _log.info("mock_telemetry_source_built")
    return source
