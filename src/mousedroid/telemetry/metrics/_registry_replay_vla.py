"""PR-A2 replay / VLA / VLM / world-model observability metrics.

Naming follows the project convention:
  * ``inc_*``      — counter increments (any cardinality)
  * ``observe_*``  — histogram observations

All families here are pure-add: they have no config toggle (operators
disable them by simply not consuming them, since the writer-side guards
in the calling subsystems treat ``metrics is None`` as a no-op) — so this
mixin, unlike most others, never needs to read ``self._cfg``.

Label values use ``Literal`` aliases from :mod:`mousedroid.config.schema`
(``ReplayOutcomeLiteral``, ``VLAActiveBackendLiteral``) so a backend rename in
one place propagates to every caller via mypy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.telemetry.metrics.primitives import (
    _classify_dropped_observation,
    _Counter,
    _Histogram,
    _LabeledCounter,
    _log,
    _prepare_bucket_boundaries,
    _render_counter,
    _render_histogram,
    _render_labeled_counter,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import (
        MetricsConfig,
        ReplayOutcomeLiteral,
        VLAActiveBackendLiteral,
    )


class _ReplayVlaMetricsMixin:
    """Replay / VLA / VLM / world-model observe_step metric family."""

    def _init_replay_vla_metrics(self, cfg: MetricsConfig) -> None:
        """Initialise PR-A2 replay / VLA / VLM / world-model metrics.

        Args:
            cfg: Metrics configuration with namespace and toggle flags.
        """
        ns = cfg.namespace

        # PR-A2 — Phase 2 replay / Phase 3 VLA / Phase 4 VLM observability.
        # All four metric families are pure-add: their internal state is
        # constructed up-front, but they are **omitted from the rendered
        # /metrics output** until the first observation/increment lands
        # (see the conditional blocks in ``render_prometheus``). Default
        # deployments therefore produce byte-identical exposition output to
        # pre-PR-A2 — the new families surface only after a writer touches them.
        # No config toggle is required to disable them — writer-side guards
        # in the calling subsystems treat ``metrics is None`` as a no-op.
        self._replay_records = _LabeledCounter()
        self._vla_inference_seconds = _Histogram(
            _prepare_bucket_boundaries(cfg.vla_inference_seconds_buckets)
        )
        self._vla_timeouts = _LabeledCounter()
        self._vlm_progress_cache_hits = _Counter()
        self._vlm_progress_cache_misses = _Counter()
        # World-model observe_step latency histogram — Tier B2 helper, wired
        # by Tier C3.1 unconditionally. ``DualStreamRSSMOnnx`` now calls the
        # helper directly (no defensive ``getattr`` fallback).
        self._world_model_observe_step_seconds = _Histogram(
            _prepare_bucket_boundaries(cfg.world_model_observe_step_seconds_buckets)
        )

        # PR-A2 — replay / VLA / VLM metric names
        self._name_replay_records = f"{ns}_replay_records"
        self._name_vla_inference_seconds = f"{ns}_vla_inference_seconds"
        self._name_vla_timeouts = f"{ns}_vla_timeouts"
        self._name_vlm_progress_cache_hits = f"{ns}_vlm_progress_cache_hits"
        self._name_vlm_progress_cache_misses = f"{ns}_vlm_progress_cache_misses"
        # Tier B2 — world-model observe_step latency histogram
        self._name_world_model_observe_step_seconds = f"{ns}_world_model_observe_step_seconds"

    def inc_replay_record(
        self,
        outcome: ReplayOutcomeLiteral,
        amount: int = 1,
    ) -> None:
        """Increment the replay-record counter for one read outcome.

        Non-positive ``amount`` is a no-op so the Prometheus counter
        monotonicity invariant is preserved if a buggy caller passes
        zero or a negative delta.

        Args:
            outcome: ``"ok"`` for a successfully deserialised record;
                ``"schema_mismatch"`` for records dropped because their
                schema version did not match the runtime ``SCHEMA_VERSION``.
                Typed as :data:`mousedroid.config.schema.ReplayOutcomeLiteral`
                so mypy catches any drift.
            amount: Increment magnitude (default 1). Values ``<= 0`` are
                ignored.
        """
        if amount > 0:
            self._replay_records.inc(outcome, amount)

    def observe_vla_inference_seconds(self, value: float) -> None:
        """Observe one VLA policy inference latency sample (seconds).

        Defensively drops samples that would corrupt the histogram sum:

        * NaN — timer misuse / division-by-zero upstream
        * ``+Inf`` — a severe hang (e.g. the backend never returned and a
          watchdog flagged the elapsed time as infinity). ``_Histogram``
          would happily route ``+Inf`` into the ``le=+Inf`` bucket, but the
          ``_sum`` accumulator would then go to ``+Inf`` forever, breaking
          every rate / quantile computation downstream.
        * Negative — clock skew / wall-clock wrap

        Drops emit a DEBUG-level structured log so operators can correlate
        missing histogram observations with the upstream root cause.

        Args:
            value: Wall-clock seconds spent inside the VLA backend's
                ``predict()`` call, measured by the caller wrapping the
                inference site with ``time.perf_counter()``.
        """
        reason = _classify_dropped_observation(value)
        if reason is not None:
            _log.debug(
                "vla_inference_seconds_dropped",
                reason=reason,
                value=value,
            )
            return
        self._vla_inference_seconds.observe(value)

    def observe_world_model_observe_step_seconds(self, value: float) -> None:
        """Observe one ``DualStreamRSSM.observe_step`` latency sample (seconds).

        Tier B2 documented this helper in the export plan but the actual
        wiring was deferred — the ``DualStreamRSSMOnnx`` runtime class
        used a defensive ``getattr(..., None)`` lookup until Tier C3.1
        landed the helper unconditionally. The runtime now calls it directly.

        Defensively drops samples that would corrupt the histogram sum
        via :func:`_classify_dropped_observation`:

        * NaN — timer misuse / division-by-zero upstream
        * ``+Inf`` — severe hang / watchdog-flagged elapsed time
        * Negative — clock skew / wall-clock wrap

        Drops emit a DEBUG-level structured log so operators can correlate
        missing histogram observations with the upstream root cause.

        Args:
            value: Wall-clock seconds spent inside one ``observe_step``
                call, measured by the caller wrapping the inference site
                with ``time.perf_counter()``.
        """
        reason = _classify_dropped_observation(value)
        if reason is not None:
            _log.debug(
                "world_model_observe_step_seconds_dropped",
                reason=reason,
                value=value,
            )
            return
        self._world_model_observe_step_seconds.observe(value)

    def inc_vla_timeout(
        self,
        mode: VLAActiveBackendLiteral,
        amount: int = 1,
    ) -> None:
        """Increment the VLA timeout counter for one fallback event.

        The ``mode`` parameter is typed as
        :data:`mousedroid.config.schema.VLAActiveBackendLiteral` — the
        narrowed subset of :data:`VLABackendLiteral` that excludes
        ``"none"``. The disabled backend cannot run inference and therefore
        cannot fire a timeout, so the type system forbids that label value
        and prevents accidental cardinality growth from spurious
        ``{mode="none"}`` series.

        Args:
            mode: Active VLA backend mode that timed out (``"mock"`` or
                ``"distilled_onnx"``). Sourced from ``cfg.vla.backend``.
            amount: Increment magnitude (default 1). Values ``<= 0`` are
                ignored to preserve Prometheus counter monotonicity.
        """
        if amount > 0:
            self._vla_timeouts.inc(mode, amount)

    def inc_vlm_cache_hit(self, amount: int = 1) -> None:
        """Increment the VLM progress-reward cache-hit counter.

        Non-positive ``amount`` is a no-op (counter monotonicity guard).
        """
        if amount > 0:
            self._vlm_progress_cache_hits.inc(amount)

    def inc_vlm_cache_miss(self, amount: int = 1) -> None:
        """Increment the VLM progress-reward cache-miss counter.

        Non-positive ``amount`` is a no-op (counter monotonicity guard).
        """
        if amount > 0:
            self._vlm_progress_cache_misses.inc(amount)

    # ------------------------------------------------------------------
    # Prometheus text exposition — family renderer
    # ------------------------------------------------------------------

    def _families_replay_vla(self) -> list[list[str]]:
        """Replay / VLA / world-model / VLM observability families."""
        out: list[list[str]] = []
        # PR-A2 — replay / VLA / VLM observability metrics. Emit conditionally
        # so deployments that never exercise these paths don't ship zero-valued
        # series. The Prometheus exposition spec allows a metric to be absent
        # entirely when no observations exist; promtool tolerates that.
        replay_snapshot = self._replay_records.snapshot()
        if replay_snapshot:
            out.append(
                _render_labeled_counter(
                    self._name_replay_records,
                    "LMDB replay records read (labels: outcome=ok|schema_mismatch)",
                    "outcome",
                    replay_snapshot,
                )
            )
        vla_buckets, vla_sum, vla_count = self._vla_inference_seconds.snapshot()
        if vla_count > 0:
            out.append(
                _render_histogram(
                    self._name_vla_inference_seconds,
                    "VLA policy inference latency histogram (seconds)",
                    vla_buckets,
                    vla_sum,
                    vla_count,
                )
            )
        wm_buckets, wm_sum, wm_count = self._world_model_observe_step_seconds.snapshot()
        if wm_count > 0:
            out.append(
                _render_histogram(
                    self._name_world_model_observe_step_seconds,
                    "DualStreamRSSM.observe_step latency histogram (seconds)",
                    wm_buckets,
                    wm_sum,
                    wm_count,
                )
            )
        vla_timeout_snapshot = self._vla_timeouts.snapshot()
        if vla_timeout_snapshot:
            out.append(
                _render_labeled_counter(
                    self._name_vla_timeouts,
                    "VLA inference timeouts / fallbacks (label: mode)",
                    "mode",
                    vla_timeout_snapshot,
                )
            )
        if self._vlm_progress_cache_hits.value > 0:
            out.append(
                _render_counter(
                    self._name_vlm_progress_cache_hits,
                    "VLM progress-reward cache hits",
                    self._vlm_progress_cache_hits.value,
                )
            )
        if self._vlm_progress_cache_misses.value > 0:
            out.append(
                _render_counter(
                    self._name_vlm_progress_cache_misses,
                    "VLM progress-reward cache misses",
                    self._vlm_progress_cache_misses.value,
                )
            )
        return out
