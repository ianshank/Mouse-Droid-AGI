"""LLM-gateway observability (Anthropic Claude tier) metrics.

Token usage, round-trip latency, per-tier served outcomes, and
latency-budget-exceeded events for the cloud/local-fallback LLM gateway
(PR #107 / PR #115). Distinct from the legacy rule-based mission-translation
family in ``_registry_core.py`` and the Phase-7 LLM-parse-latency gauge in
``_registry_phase7.py`` — this is specifically the Anthropic-tier gateway
introduced later. All gated by ``cfg.track_llm_gateway``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.telemetry.metrics.primitives import (
    _LLM_SERVED_OUTCOMES,
    _LLM_SERVED_TIERS,
    _LLM_TOKEN_TYPES,
    _classify_dropped_observation,
    _DoubleLabeledCounter,
    _Histogram,
    _LabeledCounter,
    _log,
    _prepare_bucket_boundaries,
    _render_double_labeled_counter,
    _render_histogram,
    _render_labeled_counter,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import MetricsConfig


class _LLMGatewayMetricsMixin:
    """Anthropic-tier LLM-gateway token/latency/served/budget metric family."""

    # Populated by ``_CoreMetricsMixin._init_core_metrics``, which always runs
    # first from ``MetricsRegistry.__init__``.
    _cfg: MetricsConfig

    def _init_llm_gateway_metrics(self, cfg: MetricsConfig) -> None:
        """Initialise LLM-gateway (Anthropic Claude tier) observability metrics.

        Args:
            cfg: Metrics configuration with namespace and toggle flags.
        """
        ns = cfg.namespace

        # LLM-gateway observability (Anthropic Claude tier). Pure-add families:
        # omitted from /metrics until a writer touches them, and the writer-side
        # ``metrics is None`` guard makes them a no-op when no registry is wired.
        # Gated by cfg.track_llm_gateway.
        self._llm_tokens = _DoubleLabeledCounter()
        self._llm_gateway_latency_ms = _Histogram(
            _prepare_bucket_boundaries(cfg.llm_gateway_latency_buckets_ms)
        )
        self._llm_gateway_served = _DoubleLabeledCounter()
        self._llm_latency_budget_exceeded = _LabeledCounter()

        # LLM-gateway observability metric names. The counter helpers suffix
        # ``_total`` via the shared render helpers, so omit it here.
        self._name_llm_tokens = f"{ns}_llm_tokens"
        self._name_llm_gateway_latency = f"{ns}_llm_gateway_latency_ms"
        self._name_llm_gateway_served = f"{ns}_llm_gateway_served"
        self._name_llm_latency_budget_exceeded = f"{ns}_llm_latency_budget_exceeded"

    # ------------------------------------------------------------------
    # LLM-gateway observability (Anthropic Claude tier). All gated by
    # cfg.track_llm_gateway; the writer-side ``metrics is None`` guard in the
    # gateway makes these a no-op when no registry is wired.
    # ------------------------------------------------------------------

    def inc_llm_tokens(self, model: str, token_type: str, amount: int | None) -> None:
        """Increment the LLM token-usage counter (labels: model, token_type).

        ``amount`` is ``int | None`` so callers can forward a possibly-absent
        token count straight from the SDK response usage; ``None`` and values
        ``<= 0`` are no-ops (preserves Prometheus counter monotonicity).

        Args:
            model: Model identifier label (e.g. ``"claude-haiku-4-5"``).
            token_type: ``"input"`` or ``"output"``.
            amount: Token count, or ``None`` when the response carried no usage.
        """
        if not self._cfg.track_llm_gateway or amount is None or amount <= 0:
            return
        if token_type not in _LLM_TOKEN_TYPES:
            # An out-of-set value would open a fresh series and leak cardinality.
            _log.debug("llm_tokens_dropped_invalid_token_type", token_type=token_type)
            return
        self._llm_tokens.inc(model, token_type, amount)

    def observe_llm_gateway_latency_ms(self, value: float) -> None:
        """Observe one LLM-gateway round-trip latency sample (milliseconds).

        Defensively drops NaN/+Inf/negative via
        :func:`_classify_dropped_observation` so a misused timer never corrupts
        the histogram ``_sum``. Drops emit a DEBUG structured log.

        Args:
            value: Wall-clock milliseconds for one ``translate_mission`` call.
        """
        if not self._cfg.track_llm_gateway:
            return
        reason = _classify_dropped_observation(value)
        if reason is not None:
            _log.debug("llm_gateway_latency_ms_dropped", reason=reason, value=value)
            return
        self._llm_gateway_latency_ms.observe(value)

    def inc_llm_gateway_served(self, tier: str, outcome: str, amount: int = 1) -> None:
        """Increment the per-tier served counter (labels: tier, outcome).

        Args:
            tier: ``"primary"`` (cloud Claude) or ``"secondary"`` (local fallback).
            outcome: ``"ok"`` or ``"degraded"``.
            amount: Increment magnitude (default 1); ``<= 0`` is a no-op.
        """
        if not self._cfg.track_llm_gateway or amount <= 0:
            return
        if tier not in _LLM_SERVED_TIERS or outcome not in _LLM_SERVED_OUTCOMES:
            # Fixed 2x2 label grid — drop anything else so a typo never leaks
            # cardinality into the served counter.
            _log.debug("llm_gateway_served_dropped_invalid_labels", tier=tier, outcome=outcome)
            return
        self._llm_gateway_served.inc(tier, outcome, amount)

    def inc_llm_latency_budget_exceeded(self, model: str, amount: int = 1) -> None:
        """Increment the latency-budget-exceeded counter (label: model).

        Fired alongside the ``anthropic_gateway_slow`` warning when a cloud
        round-trip exceeds ``cfg.llm.latency_target_ms``.

        Args:
            model: Model identifier label (e.g. ``"claude-haiku-4-5"``).
            amount: Increment magnitude (default 1); ``<= 0`` is a no-op.
        """
        if self._cfg.track_llm_gateway and amount > 0:
            self._llm_latency_budget_exceeded.inc(model, amount)

    # ------------------------------------------------------------------
    # Prometheus text exposition — family renderer
    # ------------------------------------------------------------------

    def _families_llm_gateway(self) -> list[list[str]]:
        """Anthropic-tier LLM-gateway observability families."""
        cfg = self._cfg
        out: list[list[str]] = []
        # LLM-gateway observability (Anthropic Claude tier). Pure-add: each
        # family is emitted only after a writer touches it (snapshot non-empty /
        # count > 0), so default deployments produce byte-identical exposition.
        if cfg.track_llm_gateway:
            llm_tokens_snapshot = self._llm_tokens.snapshot()
            if llm_tokens_snapshot:
                out.append(
                    _render_double_labeled_counter(
                        self._name_llm_tokens,
                        "LLM gateway token usage (labels: model, token_type)",
                        "model",
                        "token_type",
                        llm_tokens_snapshot,
                    )
                )
            gw_buckets, gw_sum, gw_count = self._llm_gateway_latency_ms.snapshot()
            if gw_count > 0:
                out.append(
                    _render_histogram(
                        self._name_llm_gateway_latency,
                        "LLM gateway round-trip latency histogram (milliseconds)",
                        gw_buckets,
                        gw_sum,
                        gw_count,
                    )
                )
            llm_served_snapshot = self._llm_gateway_served.snapshot()
            if llm_served_snapshot:
                out.append(
                    _render_double_labeled_counter(
                        self._name_llm_gateway_served,
                        "LLM gateway translations served (labels: tier, outcome)",
                        "tier",
                        "outcome",
                        llm_served_snapshot,
                    )
                )
            budget_snapshot = self._llm_latency_budget_exceeded.snapshot()
            if budget_snapshot:
                out.append(
                    _render_labeled_counter(
                        self._name_llm_latency_budget_exceeded,
                        "LLM gateway latency-budget-exceeded events (label: model)",
                        "model",
                        budget_snapshot,
                    )
                )
        return out
