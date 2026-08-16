"""Growth-pillar VLA distillation-cycle metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.telemetry.metrics.primitives import (
    _GROWTH_DISTILL_OUTCOMES,
    _LabeledCounter,
    _log,
    _render_labeled_counter,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import MetricsConfig


class _GrowthMetricsMixin:
    """Growth-pillar VLA distillation-cycle metric family."""

    # Populated by ``_CoreMetricsMixin._init_core_metrics``, which always runs
    # first from ``MetricsRegistry.__init__``.
    _cfg: MetricsConfig

    def _init_growth_metrics(self, cfg: MetricsConfig) -> None:
        """Initialise the growth-pillar distillation counter.

        Args:
            cfg: Metrics configuration with namespace and toggle flags.
        """
        ns = cfg.namespace

        # Growth-pillar distillation counter. Pure-add: omitted from /metrics
        # until the first cycle; gated by cfg.track_growth_distillation.
        self._growth_distillation = _LabeledCounter()

        # Growth-pillar distillation counter name (render helper suffixes ``_total``).
        self._name_growth_distillation = f"{ns}_growth_distillations"

    def inc_growth_distilled(self, outcome: str, amount: int = 1) -> None:
        """Increment the growth-pillar distillation counter (label: outcome).

        Fired once per slow-cadence distillation cycle. Pure-add and gated by
        ``cfg.track_growth_distillation``.

        Args:
            outcome: One of ``"completed"`` (a distilled student was persisted) or
                ``"skipped_no_batch"`` (the trigger armed but no latent batch was
                available). Out-of-set values are dropped with a DEBUG log so a
                free-text string never leaks cardinality.
            amount: Increment magnitude (default 1); ``<= 0`` is a no-op.
        """
        if not self._cfg.track_growth_distillation or amount <= 0:
            return
        if outcome not in _GROWTH_DISTILL_OUTCOMES:
            _log.debug("growth_distilled_dropped_invalid_outcome", outcome=outcome)
            return
        self._growth_distillation.inc(outcome, amount)

    # ------------------------------------------------------------------
    # Prometheus text exposition — family renderer
    # ------------------------------------------------------------------

    def _families_growth_distillation(self) -> list[list[str]]:
        """Growth-pillar distillation counter (label: outcome). Pure-add."""
        out: list[list[str]] = []
        cfg = self._cfg
        if cfg.track_growth_distillation:
            distill_snapshot = self._growth_distillation.snapshot()
            if distill_snapshot:
                out.append(
                    _render_labeled_counter(
                        self._name_growth_distillation,
                        "Growth-pillar VLA distillation cycles (label: outcome)",
                        "outcome",
                        distill_snapshot,
                    )
                )
        return out
