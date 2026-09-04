"""``MetricsRegistry`` — the composed Prometheus metrics registry.

Usage::

    from mousedroid.telemetry.metrics import MetricsRegistry
    from mousedroid.config.schema import MetricsConfig

    cfg = MetricsConfig()
    registry = MetricsRegistry(cfg)

    registry.inc_frame_drops()
    registry.set_loop_time_ms(18.5)
    text = registry.render_prometheus()

``MetricsRegistry`` is assembled from one mixin per metric family (see the
sibling ``_registry_*.py`` modules) so the 80+ ``inc_*``/``set_*``/
``observe_*`` methods stay on a single flat namespace — every call site in
the codebase keeps calling ``registry.inc_x(...)`` directly, unaware of the
underlying multiple-inheritance split.

Only two methods live on the concrete class rather than on any one mixin:
``__init__`` and ``render_prometheus``. Both are genuinely cross-cutting —
``__init__`` must invoke every mixin's ``_init_*_metrics`` helper, and
``render_prometheus`` must invoke every mixin's ``_families_*`` renderer.
Defining them directly on ``MetricsRegistry`` (rather than on any one mixin)
lets mypy resolve ``self`` as the full composed class, so no cross-mixin
``Protocol`` boilerplate is needed for this orchestration alone.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from mousedroid.telemetry.metrics._registry_cloud import _CloudMetricsMixin
from mousedroid.telemetry.metrics._registry_core import _CoreMetricsMixin
from mousedroid.telemetry.metrics._registry_growth import _GrowthMetricsMixin
from mousedroid.telemetry.metrics._registry_lidar import _LidarMetricsMixin
from mousedroid.telemetry.metrics._registry_llm_gateway import _LLMGatewayMetricsMixin
from mousedroid.telemetry.metrics._registry_mcp import _McpMetricsMixin
from mousedroid.telemetry.metrics._registry_mission_lifecycle import (
    _MissionLifecycleMetricsMixin,
)
from mousedroid.telemetry.metrics._registry_on_device_learning import (
    _OnDeviceLearningMetricsMixin,
)
from mousedroid.telemetry.metrics._registry_phase7 import _Phase7MetricsMixin
from mousedroid.telemetry.metrics._registry_replay_vla import _ReplayVlaMetricsMixin
from mousedroid.telemetry.metrics._registry_streaming import _StreamingMetricsMixin
from mousedroid.telemetry.metrics._registry_voice import _VoiceMetricsMixin
from mousedroid.telemetry.metrics.primitives import _log, _render_gauge

if TYPE_CHECKING:
    from mousedroid.config.schema import MetricsConfig


class MetricsRegistry(
    _CoreMetricsMixin,
    _Phase7MetricsMixin,
    _LidarMetricsMixin,
    _StreamingMetricsMixin,
    _CloudMetricsMixin,
    _McpMetricsMixin,
    _LLMGatewayMetricsMixin,
    _OnDeviceLearningMetricsMixin,
    _GrowthMetricsMixin,
    _VoiceMetricsMixin,
    _ReplayVlaMetricsMixin,
    _MissionLifecycleMetricsMixin,
):
    """Central registry for all MouseDroid Prometheus metrics.

    All metric names are derived from ``cfg.namespace`` so they can be
    overridden without code changes.  Individual metrics can be disabled
    via toggle flags in :class:`~mousedroid.config.schema.MetricsConfig`.

    Instantiate once and pass to :class:`TelemetryServer`.  Call the
    ``set_*`` / ``inc_*`` / ``observe_*`` helpers from the broadcast loop
    or orchestrator; call :meth:`render_prometheus` from the HTTP handler.
    """

    def __init__(self, cfg: MetricsConfig) -> None:
        """Initialise registry and all metric families.

        Orchestrates every mixin's ``_init_*_metrics`` helper explicitly
        (rather than a cooperative ``super().__init__()`` chain) so
        construction order is obvious and each mixin's init signature stays
        a plain ``(self, cfg: MetricsConfig) -> None`` — no MRO/``**kwargs``
        forwarding subtleties. ``_CoreMetricsMixin`` runs first because it
        establishes ``self._cfg``, which every other mixin's public methods
        read (none of the other ``_init_*_metrics`` helpers read ``_cfg``
        during construction, so the remaining call order is unconstrained).

        Args:
            cfg: Metrics configuration with namespace and toggle flags.
        """
        self._init_core_metrics(cfg)
        self._init_phase7_metrics(cfg)
        self._init_lidar_metrics(cfg)
        self._init_streaming_metrics(cfg)
        self._init_cloud_metrics(cfg)
        self._init_mcp_metrics(cfg)
        self._init_llm_gateway_metrics(cfg)
        self._init_on_device_learning_metrics(cfg)
        self._init_growth_metrics(cfg)
        self._init_voice_metrics(cfg)
        self._init_replay_vla_metrics(cfg)
        self._init_mission_lifecycle_metrics(cfg)

        _log.debug("metrics_registry_initialised", namespace=cfg.namespace)

    # ------------------------------------------------------------------
    # Prometheus text exposition format
    # ------------------------------------------------------------------

    def render_prometheus(self) -> str:
        """Render all enabled metrics in Prometheus text format 0.0.4.

        Returns:
            Plain-text Prometheus scrape payload.  Each metric family is
            separated by a blank line.  The output ends with a trailing
            newline as required by the spec.
        """
        sections: list[list[str]] = []

        # Uptime (always emitted — useful for detecting restarts)
        uptime = time.monotonic() - self._start_time
        sections.append(
            _render_gauge(self._name_uptime, "Seconds since metrics registry start", uptime)
        )
        sections.extend(self._families_core_loop())
        sections.extend(self._families_llm_translation())
        sections.extend(self._families_llm_gateway())
        sections.extend(self._families_on_device_learning())
        sections.extend(self._families_growth_distillation())
        sections.extend(self._families_voice_degradation())
        sections.extend(self._families_lidar())
        sections.extend(self._families_phase7())
        sections.extend(self._families_cloud())
        sections.extend(self._families_mcp())
        sections.extend(self._families_replay_vla())
        sections.extend(self._families_cloud_ota())
        sections.extend(self._families_mission_lifecycle())
        sections.extend(self._families_subsystem_failures())

        # Telemetry publisher rate (Hz) — always emitted.
        sections.append(
            _render_gauge(
                self._name_publish_hz,
                "Telemetry publisher rate (Hz)",
                self._publish_hz.value,
            )
        )
        sections.extend(self._families_streaming())

        return "\n\n".join("\n".join(section) for section in sections) + "\n"


def generate_metrics_sample() -> str:
    """Generate a representative Prometheus metrics sample for CI validation.

    Creates a :class:`MetricsRegistry` with default config, populates every
    metric family with representative data, and returns the rendered
    Prometheus text exposition output.  Used by the CI ``promtool check
    metrics`` step to validate format compliance.

    Returns:
        Prometheus text exposition format string with all metric families.
    """
    from mousedroid.config.schema import MetricsConfig
    from mousedroid.config.schema._primitives import TickPhaseLiteral

    cfg = MetricsConfig.model_validate({})
    registry = MetricsRegistry(cfg)

    # Populate every metric family so all appear in the output.
    registry.set_loop_time_ms(15.0)
    registry.set_battery_voltage(11.8)
    registry.set_ws_client_count(2)
    registry.set_gpu_temp_celsius(52.0)
    registry.set_publish_hz(10.0)
    registry.inc_frame_drops(3)
    registry.inc_safety_violation("law1")
    registry.inc_llm_translation("translated")
    registry.observe_llm_translation_latency_ms(42.0)
    registry.set_lidar_sectors([0.9, 0.4, 1.0, 1.0, 0.7, 1.0, 1.0, 0.2], max_range_m=12.0)
    registry.set_lidar_min_distance_m(2.4)
    registry.set_lidar_scan_points(456)
    registry.inc_subsystem_failure("voice", "device_disconnected", "error")
    registry.inc_subsystem_failure("telemetry", "bind_exhausted", "warning")
    # Per-phase tick latency: seed every phase so promtool sees the full label
    # set, not just whichever one a sample tick happened to record. The tuple
    # is annotated rather than inferred — a bare tuple widens to ``str`` and
    # mypy then (correctly) rejects the call, which is the compile-time half
    # of the label guard doing its job.
    phases: tuple[TickPhaseLiteral, ...] = (
        "sense",
        "safety",
        "world_model",
        "plan",
        "act",
        "learn",
        "telemetry",
        "post",
    )
    for phase_index, phase in enumerate(phases):
        registry.observe_tick_phase_ms(phase, 1.0 + phase_index)
    registry.inc_tick_overrun()

    # PR-A2 — exercise the new replay / VLA / VLM observability metrics so
    # ``promtool check metrics`` sees them in the CI rendered output.
    registry.inc_replay_record("ok")
    registry.inc_replay_record("schema_mismatch")
    registry.observe_vla_inference_seconds(0.012)
    registry.inc_vla_timeout("distilled_onnx")
    registry.inc_vlm_cache_hit()
    registry.inc_vlm_cache_miss()
    # Tier B2 — exercise the world-model observe_step histogram. 8 ms is
    # representative of the Orin Nano <10 ms target with TensorRT EP; lands
    # in the (0.005, 0.01] bucket of the default schema configuration.
    registry.observe_world_model_observe_step_seconds(0.008)

    # Tier C1 — exercise the cloud weight-update OTA metric families so
    # promtool / Grafana / alert evaluation all see non-empty series from
    # the first scrape (REQUIRED, not optional — alert rules reference
    # these series and would fail without seed values).
    registry.inc_cloud_weight_update_download("ianshank/mousedroid-policy-v2")
    registry.inc_cloud_weight_update_sha256_mismatch("ianshank/mousedroid-policy-v2")
    registry.observe_cloud_weight_update_download_seconds(2.5)
    registry.inc_cloud_weight_update_swap("policy")
    registry.inc_cloud_weight_update_swap("world_model")

    # Tier C2 (C2.3) — exercise mission lifecycle + safety projection
    # families so ``promtool check metrics`` sees them in CI.
    registry.inc_safety_action_clamp("forward_velocity")
    registry.inc_safety_action_clamp("human_proximity")
    registry.inc_safety_action_clamp("tight_quarters")
    registry.inc_mission_state_transition("pending", "running")
    registry.inc_mission_state_transition("running", "replanning")
    registry.inc_mission_state_transition("running", "succeeded")
    registry.inc_mission_replan("succeeded")
    registry.inc_mission_replan("failed")
    registry.observe_mission_active_duration_seconds(45.0)

    # LLM-gateway observability — seed all four families so promtool / Grafana /
    # alert evaluation see non-empty series from the first scrape.
    registry.inc_llm_tokens("claude-haiku-4-5", "input", 120)
    registry.inc_llm_tokens("claude-haiku-4-5", "output", 40)
    registry.observe_llm_gateway_latency_ms(180.0)
    registry.inc_llm_gateway_served("primary", "ok")
    registry.inc_llm_gateway_served("secondary", "degraded")
    registry.inc_llm_latency_budget_exceeded("claude-haiku-4-5")

    # Phase-6 — seed the on-device-learning revert counter (one per reason) so
    # promtool / Grafana / alert evaluation see non-empty series from the first
    # scrape.
    registry.inc_on_device_learning_reverted("regression_bound")
    registry.inc_on_device_learning_reverted("integrity_mismatch")
    registry.inc_on_device_learning_reverted("exception")

    # Growth-pillar distillation counter — seed one series per valid outcome so
    # promtool / Grafana / alert evaluation see non-empty series from the first
    # scrape.
    registry.inc_growth_distilled("completed")
    registry.inc_growth_distilled("skipped_no_batch")

    # Voice-degradation counters — seed one series per valid label value so
    # promtool / Grafana / alert evaluation see non-empty series from the first
    # scrape.
    registry.inc_voice_speaker_degraded("usb_speaker")
    registry.inc_voice_speaker_degraded("rocky_fallback")
    registry.inc_voice_tts_synthesize_failures("synthesize")
    registry.inc_voice_tts_synthesize_failures("synthesize_wav")
    registry.inc_voice_tts_synthesize_failures("synthesize_wav_file")

    return registry.render_prometheus()
