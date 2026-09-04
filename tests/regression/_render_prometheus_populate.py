"""Shared maximal-populate helper for the ``render_prometheus`` golden test.

Builds a :class:`MetricsRegistry` with a default (all-``track_*``-enabled)
config and exercises *every* public populate method so that every metric
family appears in the rendered exposition. Used by
``test_render_prometheus_golden.py`` to pin the full Prometheus output
byte-for-byte across the ``render_prometheus`` decomposition (Item 2 of the
enterprise-hardening sprint).

The only non-deterministic line in the output is the always-emitted uptime
gauge (``time.monotonic()``-derived); callers mask it via
:func:`mask_uptime` before comparison.
"""

from __future__ import annotations

import re

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import MetricsRegistry

_UPTIME_VALUE_RE = re.compile(r"^(\w*uptime\w* )\S+$", re.MULTILINE)


def mask_uptime(rendered: str) -> str:
    """Replace the non-deterministic uptime gauge value with a placeholder.

    The uptime family is the only clock-derived line in the exposition; every
    other value is a function of the deterministic populate calls below.

    Args:
        rendered: Raw ``render_prometheus()`` output.

    Returns:
        The same text with the uptime metric value replaced by ``<MASKED>``.
    """
    return _UPTIME_VALUE_RE.sub(r"\1<MASKED>", rendered)


def build_maximal_registry() -> MetricsRegistry:
    """Build a registry with every metric family populated.

    Returns:
        A :class:`MetricsRegistry` whose ``render_prometheus`` emits every
        family the registry knows how to expose.
    """
    registry = MetricsRegistry(MetricsConfig.model_validate({}))

    # Core loop / safety / power.
    registry.set_loop_time_ms(15.0)
    registry.set_battery_voltage(11.8)
    registry.set_ws_client_count(2)
    registry.set_gpu_temp_celsius(52.0)
    registry.set_publish_hz(10.0)
    registry.inc_frame_drops(3)
    registry.inc_safety_violation("law1")

    # LLM mission translation.
    registry.observe_mcp_memory_query_latency_ms(15.0)
    registry.inc_llm_translation("translated")
    registry.observe_llm_translation_latency_ms(42.0)

    # LLM gateway observability (Anthropic tier).
    registry.inc_llm_tokens("claude-haiku-4-5", "input", 120)
    registry.inc_llm_tokens("claude-haiku-4-5", "output", 40)
    registry.observe_llm_gateway_latency_ms(180.0)
    registry.inc_llm_gateway_served("primary", "ok")
    registry.inc_llm_gateway_served("secondary", "degraded")
    registry.inc_llm_latency_budget_exceeded("claude-haiku-4-5")

    # On-device learning.
    registry.inc_on_device_learning_reverted("regression_bound")
    registry.inc_on_device_learning_reverted("integrity_mismatch")
    registry.inc_on_device_learning_reverted("exception")

    # Voice degradation (speaker + TTS).
    registry.inc_voice_speaker_degraded("usb_speaker")
    registry.inc_voice_speaker_degraded("rocky_fallback")
    registry.inc_voice_tts_synthesize_failures("synthesize")
    registry.inc_voice_tts_synthesize_failures("synthesize_wav")
    registry.inc_voice_tts_synthesize_failures("synthesize_wav_file")

    # LiDAR sector gauges.
    registry.set_lidar_sectors([0.9, 0.4, 1.0, 1.0, 0.7, 1.0, 1.0, 0.2], max_range_m=12.0)
    registry.set_lidar_min_distance_m(2.4)
    registry.set_lidar_scan_points(456)

    # Phase 7 — memory / voice / llm-latency / curiosity / sensor-recovery.
    registry.set_episodic_size(128)
    registry.set_semantic_size(64)
    registry.set_working_size(16)
    registry.inc_voice_event("wake_word")
    registry.set_llm_latency_ms(210.0)
    registry.set_curiosity_reward(0.37)
    registry.inc_sensor_recoveries(2)
    registry.inc_sensor_recovery_failures(1)

    # Cloud digital twin.
    registry.inc_cloud_telemetry_publish("ok")
    registry.inc_cloud_experience_publish("ok")
    registry.observe_cloud_telemetry_publish_latency_ms(12.0)
    registry.observe_cloud_experience_publish_latency_ms(18.0)
    registry.set_cloud_circuit_state("telemetry", "closed")
    registry.inc_cloud_experience_export_records("ok", 5)
    registry.set_cloud_experience_hwm_lag(3)
    registry.set_cloud_experience_queue_depth(7)

    # MCP server.
    registry.inc_mcp_request()
    registry.inc_mcp_tool_call("calibrate", "ok")
    registry.observe_mcp_request_latency_ms(9.0)

    # PR-A2 — replay / VLA / VLM.
    registry.inc_replay_record("ok")
    registry.inc_replay_record("schema_mismatch")
    registry.observe_vla_inference_seconds(0.012)
    registry.observe_world_model_observe_step_seconds(0.008)
    registry.inc_vla_timeout("distilled_onnx")
    registry.inc_vlm_cache_hit()
    registry.inc_vlm_cache_miss()

    # Tier C1 — cloud weight-update OTA.
    registry.inc_cloud_weight_update_download("ianshank/mousedroid-policy-v2")
    registry.inc_cloud_weight_update_sha256_mismatch("ianshank/mousedroid-policy-v2")
    registry.observe_cloud_weight_update_download_seconds(2.5)
    registry.inc_cloud_weight_update_swap("policy")
    registry.inc_cloud_weight_update_swap("world_model")

    # Tier C2 — mission lifecycle + safety projection.
    registry.inc_safety_action_clamp("forward_velocity")
    registry.inc_safety_action_clamp("human_proximity")
    registry.inc_mission_state_transition("pending", "running")
    registry.inc_mission_state_transition("running", "succeeded")
    registry.inc_mission_replan("succeeded")
    registry.inc_mission_replan("failed")
    registry.inc_mission_replan_llm("ok")
    registry.observe_mission_active_duration_seconds(45.0)

    # Subsystem failures.
    registry.inc_subsystem_failure("voice", "device_disconnected", "error")
    registry.inc_subsystem_failure("telemetry", "bind_exhausted", "warning")

    # PR #4 — streaming / dashboard liveness.
    registry.set_sensor_liveness({"vision": "live", "lidar": "stale"})
    registry.set_mdns_registered("mousedroid._http._tcp", ok=True)
    registry.set_bound_port(8080)
    registry.inc_lidar_raw_published(11)
    registry.inc_lidar_raw_dropped(2)

    # Tick instrumentation — every phase, so the golden fixture pins the full
    # label set and a dropped/renamed phase shows up as a diff.
    for phase_index, phase in enumerate(
        ("sense", "safety", "world_model", "plan", "act", "learn", "telemetry", "post")
    ):
        registry.observe_tick_phase_ms(phase, 1.0 + phase_index)
    registry.inc_tick_overrun()

    return registry
