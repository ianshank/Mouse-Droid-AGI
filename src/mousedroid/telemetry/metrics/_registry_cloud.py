"""Cloud Digital Twin publish/circuit/backlog metrics + Tier-C1 weight-update OTA metrics.

Every public method here is ``inc_cloud_*`` / ``observe_cloud_*`` /
``set_cloud_*`` / ``get_cloud_*`` — a single coherent "cloud" family, merging
what the pre-split code rendered as two adjacent-but-separate functions
(``_families_cloud`` for telemetry/experience publish + circuit-breaker state,
``_families_cloud_ota`` for the OTA weight-update download/swap counters).
``render_prometheus`` still calls both family renderers at their original
relative positions, so the merge is a file-organisation choice only — it does
not change the rendered byte order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.telemetry.metrics.primitives import (
    _classify_dropped_observation,
    _Gauge,
    _Histogram,
    _LabeledCounter,
    _LabeledGauge,
    _log,
    _prepare_bucket_boundaries,
    _render_gauge,
    _render_histogram,
    _render_labeled_counter,
    _render_labeled_gauge,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import MetricsConfig


class _CloudMetricsMixin:
    """Cloud Digital Twin publish/circuit/backlog + weight-update OTA metrics."""

    # Populated by ``_CoreMetricsMixin._init_core_metrics``, which always runs
    # first from ``MetricsRegistry.__init__``.
    _cfg: MetricsConfig

    def _init_cloud_metrics(self, cfg: MetricsConfig) -> None:
        """Initialise Cloud Digital Twin publish + weight-update OTA metrics.

        Args:
            cfg: Metrics configuration with namespace and toggle flags.
        """
        ns = cfg.namespace

        # Cloud Digital Twin — per-sink (telemetry/experience) result counters
        self._cloud_telemetry_publish = _LabeledCounter()
        self._cloud_experience_publish = _LabeledCounter()
        self._cloud_experience_export_records = _LabeledCounter()

        # Cloud Digital Twin gauges — breaker state per breaker, export backlog
        self._cloud_circuit_state = _LabeledGauge()
        self._cloud_experience_hwm_lag = _Gauge()
        self._cloud_experience_queue_depth = _Gauge()

        # Cloud publish latency histograms — share the LLM bucket layout by
        # default; both telemetry and experience publishes fall in the
        # 25 ms - 2 s envelope.
        llm_buckets = _prepare_bucket_boundaries(cfg.llm_latency_buckets_ms)
        self._cloud_telemetry_publish_latency_ms = _Histogram(llm_buckets)
        self._cloud_experience_publish_latency_ms = _Histogram(llm_buckets)

        # Tier C1 — Closed-loop cloud retraining + OTA weight updates.
        # Four pure-add metric families surface the OTA loop on /metrics:
        # downloads, SHA-256 mismatches, download latency, and engine swaps.
        # All four are emitted unconditionally (no track_* toggle) — operators
        # disable them by leaving cfg.cloud.weight_update.poll_interval_s = 0
        # so the poller never fires. The render path emits them only after
        # the first observation/increment lands, mirroring PR-A2.
        self._cloud_weight_update_downloads = _LabeledCounter()
        self._cloud_weight_update_sha256_mismatches = _LabeledCounter()
        # Bucket boundaries normalised via the shared helper (C3.1 Gemini #2).
        self._cloud_weight_update_download_seconds = _Histogram(
            _prepare_bucket_boundaries(cfg.cloud_weight_update_download_seconds_buckets)
        )
        self._cloud_weight_update_swaps = _LabeledCounter()

        # Cloud Digital Twin metric names — all derived from namespace
        self._name_cloud_telemetry_publish = f"{ns}_cloud_telemetry_publish"
        self._name_cloud_experience_publish = f"{ns}_cloud_experience_publish"
        self._name_cloud_telemetry_publish_latency = f"{ns}_cloud_telemetry_publish_latency_ms"
        self._name_cloud_experience_publish_latency = f"{ns}_cloud_experience_publish_latency_ms"
        self._name_cloud_circuit_state = f"{ns}_cloud_circuit_state"
        self._name_cloud_experience_export_records = f"{ns}_cloud_experience_export_records"
        self._name_cloud_experience_hwm_lag = f"{ns}_cloud_experience_hwm_lag"
        self._name_cloud_experience_queue_depth = f"{ns}_cloud_experience_queue_depth"

        # Tier C1 — cloud weight-update OTA metric names
        self._name_cloud_weight_update_downloads = f"{ns}_cloud_weight_update_downloads"
        self._name_cloud_weight_update_sha256_mismatches = (
            f"{ns}_cloud_weight_update_sha256_mismatches"
        )
        self._name_cloud_weight_update_download_seconds = (
            f"{ns}_cloud_weight_update_download_seconds"
        )
        self._name_cloud_weight_update_swaps = f"{ns}_cloud_weight_update_swaps"

    # ------------------------------------------------------------------
    # Cloud Digital Twin helpers
    # ------------------------------------------------------------------

    def inc_cloud_telemetry_publish(self, result: str, amount: int = 1) -> None:
        """Increment cloud telemetry publish counter for a result label.

        Args:
            result: Outcome label (e.g. ``"success"``, ``"error"``,
                ``"circuit_open"``, ``"retry_exhausted"``).
            amount: Increment amount (default 1).
        """
        if self._cfg.track_cloud:
            self._cloud_telemetry_publish.inc(result, amount)

    def inc_cloud_experience_publish(self, result: str, amount: int = 1) -> None:
        """Increment cloud experience publish counter for a result label."""
        if self._cfg.track_cloud:
            self._cloud_experience_publish.inc(result, amount)

    def observe_cloud_telemetry_publish_latency_ms(self, value: float) -> None:
        """Record cloud telemetry publish latency in milliseconds."""
        if self._cfg.track_cloud:
            self._cloud_telemetry_publish_latency_ms.observe(value)

    def observe_cloud_experience_publish_latency_ms(self, value: float) -> None:
        """Record cloud experience publish latency in milliseconds."""
        if self._cfg.track_cloud:
            self._cloud_experience_publish_latency_ms.observe(value)

    def set_cloud_circuit_state(self, breaker: str, state: str) -> None:
        """Record current circuit breaker state as a numeric gauge.

        Gauge encoding: ``0`` = CLOSED, ``1`` = HALF_OPEN, ``2`` = OPEN.
        Unknown states default to ``-1``. The mapping is intentionally
        not config-driven because Grafana dashboards rely on these
        numeric values.

        Args:
            breaker: Circuit breaker name (e.g. ``"cloud_telemetry"``).
            state: Lowercased state string from :class:`CircuitState`.
        """
        if not self._cfg.track_cloud:
            return
        encoded: dict[str, float] = {
            "closed": 0.0,
            "half_open": 1.0,
            "open": 2.0,
        }
        self._cloud_circuit_state.set(breaker, encoded.get(state, -1.0))

    def inc_cloud_experience_export_records(self, result: str, amount: int) -> None:
        """Increment experience-records-exported counter.

        Args:
            result: Outcome label (``"success"``, ``"error"``).
            amount: Number of records successfully/failed in this batch.
        """
        if self._cfg.track_cloud and amount > 0:
            self._cloud_experience_export_records.inc(result, amount)

    def set_cloud_experience_hwm_lag(self, lag_records: int) -> None:
        """Set how many records remain between current HWM and DB tip."""
        if self._cfg.track_cloud:
            self._cloud_experience_hwm_lag.set(float(lag_records))

    def set_cloud_experience_queue_depth(self, depth: int) -> None:
        """Set current in-memory experience queue depth."""
        if self._cfg.track_cloud:
            self._cloud_experience_queue_depth.set(float(depth))

    @staticmethod
    def _decode_cloud_circuit_state(value: float) -> str:
        """Map numeric breaker gauge values back to symbolic states."""
        if value == 0.0:
            return "closed"
        if value == 1.0:
            return "half_open"
        if value == 2.0:
            return "open"
        return "unknown"

    def get_cloud_health_snapshot(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot of cloud health metrics.

        The telemetry server uses this to expose ``/api/v1/health/cloud``
        without coupling to concrete cloud sink/exporter implementations.
        """
        if not self._cfg.track_cloud:
            return {"enabled": False, "status": "disabled"}

        breaker_states = {
            breaker: self._decode_cloud_circuit_state(encoded)
            for breaker, encoded in self._cloud_circuit_state.snapshot().items()
        }
        queue_depth = int(self._cloud_experience_queue_depth.value)
        hwm_lag = int(self._cloud_experience_hwm_lag.value)
        status = "ok"
        if any(state == "open" for state in breaker_states.values()):
            status = "degraded"
        elif queue_depth > 0 or hwm_lag > 0:
            status = "backlogged"

        return {
            "enabled": True,
            "status": status,
            "breaker_states": breaker_states,
            "queue_depth": queue_depth,
            "hwm_lag": hwm_lag,
            "telemetry_publish": self._cloud_telemetry_publish.snapshot(),
            "experience_publish": self._cloud_experience_publish.snapshot(),
            "experience_export_records": self._cloud_experience_export_records.snapshot(),
        }

    # ------------------------------------------------------------------
    # Tier C1 — cloud weight-update OTA observability helpers
    # ------------------------------------------------------------------

    def inc_cloud_weight_update_download(self, repo_id: str, amount: int = 1) -> None:
        """Increment the cloud weight-update download counter for ``repo_id``.

        Non-positive ``amount`` is a no-op (counter monotonicity guard).

        Args:
            repo_id: HuggingFace Hub repo ID the artifact was downloaded
                from (e.g. ``"ianshank/mousedroid-policy-v2"``). Used as
                the Prometheus label value.
            amount: Increment magnitude (default 1).
        """
        if amount > 0:
            self._cloud_weight_update_downloads.inc(repo_id, amount)

    def inc_cloud_weight_update_sha256_mismatch(self, repo_id: str, amount: int = 1) -> None:
        """Increment the cloud weight-update SHA-256 mismatch counter.

        SAFETY-CRITICAL: every increment of this counter corresponds to a
        refused artifact swap. Operator alert rules should page on any
        non-zero rate.

        Args:
            repo_id: HF Hub repo ID whose downloaded artifact failed the
                SHA-256 integrity check.
            amount: Increment magnitude (default 1).
        """
        if amount > 0:
            self._cloud_weight_update_sha256_mismatches.inc(repo_id, amount)

    def observe_cloud_weight_update_download_seconds(self, value: float) -> None:
        """Observe one OTA artifact download latency sample (seconds).

        Defensively drops samples that would corrupt the histogram sum
        via :func:`_classify_dropped_observation` (the shared C3.1 helper):

        * NaN — timer misuse / division-by-zero upstream
        * ``+Inf`` — severe hang / watchdog-flagged elapsed time
        * Negative — clock skew / wall-clock wrap

        Args:
            value: Wall-clock seconds spent inside ``hf_hub_download(...)``,
                measured by the caller wrapping the download with
                ``time.perf_counter()``.
        """
        reason = _classify_dropped_observation(value)
        if reason is not None:
            _log.debug(
                "cloud_weight_update_download_seconds_dropped",
                reason=reason,
                value=value,
            )
            return
        self._cloud_weight_update_download_seconds.observe(value)

    def inc_cloud_weight_update_swap(self, engine_type: str, amount: int = 1) -> None:
        """Increment the cloud weight-update swap counter for one engine swap.

        Args:
            engine_type: ``"policy"`` for VLA policy swaps or
                ``"world_model"`` for world-model swaps. Free-form string —
                kept un-Literal to allow future engine kinds (e.g.
                ``"vlm_progress"``) without churning the schema.
            amount: Increment magnitude (default 1).
        """
        if amount > 0:
            self._cloud_weight_update_swaps.inc(engine_type, amount)

    # ------------------------------------------------------------------
    # Prometheus text exposition — family renderers
    # ------------------------------------------------------------------

    def _families_cloud(self) -> list[list[str]]:
        """Cloud digital-twin publish / circuit / backlog families."""
        cfg = self._cfg
        out: list[list[str]] = []
        if cfg.track_cloud:
            telemetry_counts = self._cloud_telemetry_publish.snapshot()
            if telemetry_counts:
                out.append(
                    _render_labeled_counter(
                        self._name_cloud_telemetry_publish,
                        "Cloud telemetry publish outcomes (label: result)",
                        "result",
                        telemetry_counts,
                    )
                )
            experience_counts = self._cloud_experience_publish.snapshot()
            if experience_counts:
                out.append(
                    _render_labeled_counter(
                        self._name_cloud_experience_publish,
                        "Cloud experience publish outcomes (label: result)",
                        "result",
                        experience_counts,
                    )
                )
            tel_buckets, tel_sum, tel_count = self._cloud_telemetry_publish_latency_ms.snapshot()
            if tel_count > 0:
                out.append(
                    _render_histogram(
                        self._name_cloud_telemetry_publish_latency,
                        "Cloud telemetry publish latency (milliseconds)",
                        tel_buckets,
                        tel_sum,
                        tel_count,
                    )
                )
            exp_buckets, exp_sum, exp_count = self._cloud_experience_publish_latency_ms.snapshot()
            if exp_count > 0:
                out.append(
                    _render_histogram(
                        self._name_cloud_experience_publish_latency,
                        "Cloud experience publish latency (milliseconds)",
                        exp_buckets,
                        exp_sum,
                        exp_count,
                    )
                )
            circuit_snapshot = self._cloud_circuit_state.snapshot()
            if circuit_snapshot:
                out.append(
                    _render_labeled_gauge(
                        self._name_cloud_circuit_state,
                        ("Circuit breaker state (0=closed, 1=half_open, 2=open; label: breaker)"),
                        "breaker",
                        circuit_snapshot,
                    )
                )
            export_counts = self._cloud_experience_export_records.snapshot()
            if export_counts:
                out.append(
                    _render_labeled_counter(
                        self._name_cloud_experience_export_records,
                        "Cloud experience records exported (label: result)",
                        "result",
                        export_counts,
                    )
                )
            out.append(
                _render_gauge(
                    self._name_cloud_experience_hwm_lag,
                    "Experience records between LMDB HWM and tip",
                    self._cloud_experience_hwm_lag.value,
                )
            )
            out.append(
                _render_gauge(
                    self._name_cloud_experience_queue_depth,
                    "Pending experience records awaiting cloud publish",
                    self._cloud_experience_queue_depth.value,
                )
            )
        return out

    def _families_cloud_ota(self) -> list[list[str]]:
        """Tier-C1 cloud weight-update OTA families."""
        out: list[list[str]] = []
        # Tier C1 — cloud weight-update OTA metrics. Emitted only after the
        # first observation/increment lands so deployments with the OTA
        # poller disabled (default ``cloud.weight_update.poll_interval_s = 0``)
        # produce byte-identical Prometheus exposition output to pre-C1.
        cwu_downloads = self._cloud_weight_update_downloads.snapshot()
        if cwu_downloads:
            out.append(
                _render_labeled_counter(
                    self._name_cloud_weight_update_downloads,
                    "OTA weight-update downloads from HF Hub (label: repo_id)",
                    "repo_id",
                    cwu_downloads,
                )
            )
        cwu_mismatches = self._cloud_weight_update_sha256_mismatches.snapshot()
        if cwu_mismatches:
            out.append(
                _render_labeled_counter(
                    self._name_cloud_weight_update_sha256_mismatches,
                    "OTA weight-update SHA-256 integrity failures (label: repo_id)",
                    "repo_id",
                    cwu_mismatches,
                )
            )
        cwu_buckets, cwu_sum, cwu_count = self._cloud_weight_update_download_seconds.snapshot()
        if cwu_count > 0:
            out.append(
                _render_histogram(
                    self._name_cloud_weight_update_download_seconds,
                    "OTA weight-update download latency histogram (seconds)",
                    cwu_buckets,
                    cwu_sum,
                    cwu_count,
                )
            )
        cwu_swaps = self._cloud_weight_update_swaps.snapshot()
        if cwu_swaps:
            out.append(
                _render_labeled_counter(
                    self._name_cloud_weight_update_swaps,
                    "OTA atomic engine swaps applied by the orchestrator (label: engine_type)",
                    "engine_type",
                    cwu_swaps,
                )
            )
        return out
