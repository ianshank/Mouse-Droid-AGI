"""Phase-6 on-device-learning weight-update revert metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.telemetry.metrics.primitives import (
    _ON_DEVICE_REVERT_REASONS,
    _LabeledCounter,
    _log,
    _render_labeled_counter,
)

if TYPE_CHECKING:
    from mousedroid.config.schema import MetricsConfig


class _OnDeviceLearningMetricsMixin:
    """Phase-6 on-device-learning revert-counter metric family."""

    # Populated by ``_CoreMetricsMixin._init_core_metrics``, which always runs
    # first from ``MetricsRegistry.__init__``.
    _cfg: MetricsConfig

    def _init_on_device_learning_metrics(self, cfg: MetricsConfig) -> None:
        """Initialise the Phase-6 on-device-learning revert counter.

        Args:
            cfg: Metrics configuration with namespace and toggle flags.
        """
        ns = cfg.namespace

        # Phase-6 on-device-learning revert counter. Pure-add: omitted from
        # /metrics until the first revert; gated by cfg.track_on_device_learning.
        self._on_device_learning_reverted = _LabeledCounter()

        # Phase-6 on-device-learning revert counter name. The counter render
        # helper suffixes ``_total``, so omit it here.
        self._name_on_device_learning_reverted = f"{ns}_on_device_learning_reverted"

    def inc_on_device_learning_reverted(self, reason: str, amount: int = 1) -> None:
        """Increment the Phase-6 on-device-learning revert counter (label: reason).

        Fired when the safety-regression gate reverts an on-device weight update
        back to the cloud slot. Pure-add and gated by
        ``cfg.track_on_device_learning``.

        Args:
            reason: One of ``"regression_bound"`` (held-out recon+KL loss exceeded
                ``baseline_loss + regression_tolerance`` — LOWER loss is better),
                ``"integrity_mismatch"`` (SHA-256 checkpoint verification failed),
                or ``"exception"`` (the update path raised). Out-of-set values are
                dropped with a DEBUG log so a free-text mission string never leaks
                cardinality.
            amount: Increment magnitude (default 1); ``<= 0`` is a no-op.
        """
        if not self._cfg.track_on_device_learning or amount <= 0:
            return
        if reason not in _ON_DEVICE_REVERT_REASONS:
            _log.debug("on_device_learning_reverted_dropped_invalid_reason", reason=reason)
            return
        self._on_device_learning_reverted.inc(reason, amount)

    # ------------------------------------------------------------------
    # Prometheus text exposition — family renderer
    # ------------------------------------------------------------------

    def _families_on_device_learning(self) -> list[list[str]]:
        """Phase-6 on-device-learning revert counter."""
        cfg = self._cfg
        out: list[list[str]] = []
        # Phase-6 on-device-learning revert counter. Pure-add: emitted only
        # after a revert lands (snapshot non-empty), so default deployments
        # render byte-identically.
        if cfg.track_on_device_learning:
            revert_snapshot = self._on_device_learning_reverted.snapshot()
            if revert_snapshot:
                out.append(
                    _render_labeled_counter(
                        self._name_on_device_learning_reverted,
                        "On-device-learning weight-update reverts (label: reason)",
                        "reason",
                        revert_snapshot,
                    )
                )
        return out
