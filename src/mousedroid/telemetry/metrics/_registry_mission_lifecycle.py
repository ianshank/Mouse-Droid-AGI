"""Tier-C2 mission lifecycle + safety projection metrics.

All four families default-disabled by writer-side guards: the projector is
only built when ``cfg.safety.projector.enabled`` is True, and the mission
lifecycle is only built when ``cfg.mission.replan_enabled`` is True. Pre-C2
deployments therefore produce byte-identical /metrics output — these names
appear only after a writer first touches them. No ``track_*`` config toggle
gates this family (unlike most others), so this mixin never needs to read
``self._cfg``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mousedroid.telemetry.metrics.primitives import (
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


class _MissionLifecycleMetricsMixin:
    """Mission state-transition / replan / safety-clamp metric family."""

    def _init_mission_lifecycle_metrics(self, cfg: MetricsConfig) -> None:
        """Initialise Tier-C2 mission lifecycle + safety projection metrics.

        Args:
            cfg: Metrics configuration with namespace and toggle flags.
        """
        ns = cfg.namespace

        # Tier C2 (C2.3) — mission lifecycle + safety projection metrics.
        # All four families are pure-add: their internal state is constructed
        # up-front, but they are omitted from the rendered /metrics output
        # until the first observation / increment lands (see the conditional
        # blocks in ``render_prometheus``). Default deployments therefore
        # produce byte-identical exposition output to pre-C2 — the new
        # families surface only after a writer touches them.
        self._safety_action_clamps = _LabeledCounter()
        self._mission_state_transitions = _DoubleLabeledCounter()
        self._mission_replans = _LabeledCounter()
        # Tier C2.3 — LLM-backed replan attempts per outcome
        # (``ok`` | ``degraded`` | ``exception``).
        self._mission_replan_llm_calls = _LabeledCounter()
        # Bucket boundaries normalised via the shared helper (C3.1 Gemini #2).
        self._mission_active_duration_seconds = _Histogram(
            _prepare_bucket_boundaries(cfg.mission_duration_seconds_buckets)
        )

        # Tier C2 (C2.3) — mission lifecycle + safety projection metric names
        self._name_safety_action_clamps = f"{ns}_safety_action_clamps"
        self._name_mission_state_transitions = f"{ns}_mission_state_transitions"
        self._name_mission_replans = f"{ns}_mission_replans"
        # Tier C2.3 — LLM replan call counter (suffixed ``_total`` by the
        # shared ``_render_labeled_counter`` helper, so omit it here).
        self._name_mission_replan_llm_calls = f"{ns}_mission_replan_llm_calls"
        self._name_mission_active_duration_seconds = f"{ns}_mission_active_duration_seconds"

    def inc_safety_action_clamp(self, reason: str, amount: int = 1) -> None:
        """Increment the safety-action-clamp counter for one clamp reason.

        Args:
            reason: One of ``forward_velocity`` / ``human_proximity`` /
                ``tight_quarters``. Other values are accepted but
                discouraged — they would grow the label cardinality.
            amount: Increment magnitude (default 1). Values ``<= 0`` are
                ignored to preserve Prometheus counter monotonicity.
        """
        if amount > 0:
            self._safety_action_clamps.inc(reason, amount)

    def inc_mission_state_transition(
        self,
        from_state: str,
        to_state: str,
        amount: int = 1,
    ) -> None:
        """Increment the mission state-transition counter for one edge.

        Args:
            from_state: Lower-case ``TaskStatus`` value the lifecycle is
                leaving (e.g. ``"running"``).
            to_state: Lower-case ``TaskStatus`` value the lifecycle is
                entering (e.g. ``"replanning"``).
            amount: Increment magnitude (default 1). Values ``<= 0`` are
                ignored.
        """
        if amount > 0:
            self._mission_state_transitions.inc(from_state, to_state, amount)

    def inc_mission_replan(self, outcome: str, amount: int = 1) -> None:
        """Increment the mission-replan counter for one replan outcome.

        Args:
            outcome: ``"succeeded"`` when the LLM returned a fresh
                ``GoalVector`` and the lifecycle resumed RUNNING;
                ``"failed"`` when the LLM returned ``None`` and the
                lifecycle transitioned to FAILED.
            amount: Increment magnitude (default 1). Values ``<= 0`` are
                ignored.
        """
        if amount > 0:
            self._mission_replans.inc(outcome, amount)

    def inc_mission_replan_llm(self, outcome: str, amount: int = 1) -> None:
        """Increment the Tier C2.3 LLM-backed replan attempt counter.

        Args:
            outcome: One of ``"ok"`` (gateway returned a parsed
                ``GoalVector``), ``"degraded"`` (gateway's ``is_ready``
                property was False so the adapter short-circuited),
                or ``"exception"`` (gateway raised mid-call). Other
                strings are accepted but operators should reserve them
                for future expansion — Prometheus alerts can pin the
                allowed label set.
            amount: Increment magnitude (default 1). Values ``<= 0`` are
                ignored, mirroring :meth:`inc_mission_replan`.
        """
        if amount > 0:
            self._mission_replan_llm_calls.inc(outcome, amount)

    def observe_mission_active_duration_seconds(self, value: float) -> None:
        """Record one terminal mission's active duration (seconds).

        Defensively drops samples that would corrupt the histogram sum
        via :func:`_classify_dropped_observation` (the shared C3.1 helper):

        * NaN — timer misuse / division-by-zero upstream
        * ``+Inf`` — severe hang / watchdog-flagged elapsed time
        * Negative — clock skew / wall-clock wrap

        Drops emit a DEBUG-level structured log so operators can correlate
        missing observations with the upstream root cause.

        Args:
            value: Wall-clock seconds the mission spent in RUNNING /
                REPLANNING before terminating (SUCCEEDED or FAILED).
        """
        reason = _classify_dropped_observation(value)
        if reason is not None:
            _log.debug(
                "mission_active_duration_seconds_dropped",
                reason=reason,
                value=value,
            )
            return
        self._mission_active_duration_seconds.observe(value)

    # ------------------------------------------------------------------
    # Prometheus text exposition — family renderer
    # ------------------------------------------------------------------

    def _families_mission_lifecycle(self) -> list[list[str]]:
        """Tier-C2 mission-lifecycle + safety-projection families."""
        out: list[list[str]] = []
        # Tier C2 (C2.3) — mission lifecycle + safety projection.
        # Emit conditionally so deployments that never exercise these
        # paths don't ship zero-valued series (matches the PR-A2 pattern).
        safety_clamps_snapshot = self._safety_action_clamps.snapshot()
        if safety_clamps_snapshot:
            out.append(
                _render_labeled_counter(
                    self._name_safety_action_clamps,
                    "Safety action clamps applied by the projector (label: reason)",
                    "reason",
                    safety_clamps_snapshot,
                )
            )
        mission_transitions_snapshot = self._mission_state_transitions.snapshot()
        if mission_transitions_snapshot:
            out.append(
                _render_double_labeled_counter(
                    self._name_mission_state_transitions,
                    "Mission lifecycle state transitions (labels: from_state, to_state)",
                    "from_state",
                    "to_state",
                    mission_transitions_snapshot,
                )
            )
        mission_replans_snapshot = self._mission_replans.snapshot()
        if mission_replans_snapshot:
            out.append(
                _render_labeled_counter(
                    self._name_mission_replans,
                    "Mission replans by outcome (label: outcome)",
                    "outcome",
                    mission_replans_snapshot,
                )
            )
        # Tier C2.3 — LLM replan call counter (ok/degraded/exception).
        mission_replan_llm_snapshot = self._mission_replan_llm_calls.snapshot()
        if mission_replan_llm_snapshot:
            out.append(
                _render_labeled_counter(
                    self._name_mission_replan_llm_calls,
                    "LLM-backed replan attempts by outcome (label: outcome)",
                    "outcome",
                    mission_replan_llm_snapshot,
                )
            )
        (
            mission_buckets,
            mission_sum,
            mission_count,
        ) = self._mission_active_duration_seconds.snapshot()
        if mission_count > 0:
            out.append(
                _render_histogram(
                    self._name_mission_active_duration_seconds,
                    "Mission active duration histogram (seconds)",
                    mission_buckets,
                    mission_sum,
                    mission_count,
                )
            )
        return out
