"""Reward and safety configuration models.

Pillar 6 (multi-objective reward, including the VLM-derived dense progress
signal) and the safety monitor / geometric action projector / Three Laws
enforcement.
"""

from __future__ import annotations

from itertools import pairwise

from pydantic import Field, model_validator

from mousedroid.config.schema._primitives import (
    Self,
    StrictBaseModel,
    _settings_default_factory,
)


class VLMProgressConfig(StrictBaseModel):
    """VLM-derived dense progress reward configuration (Phase 4).

    The VLM progress head produces a scalar in ``[0, 1]`` that estimates how
    much closer the current observation is to satisfying ``instruction``
    compared to the previous observation. The score is gated by the Three
    Laws Law-1 sigmoid in :class:`MultiObjectiveRewardModel`, so a contrived
    high progress value cannot override a harm violation.
    """

    enabled: bool = Field(False, description="Toggle VLM progress head")
    cache_size: int = Field(
        4096,
        ge=1,
        description="Max entries in the (prev,curr,instruction) LRU cache",
    )
    instruction: str = Field(
        "complete the task safely",
        description="Default natural-language instruction passed to the VLM",
    )
    mock_progress_value: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Constant value returned by MockVLMProgress backend (tests/default-off)",
    )
    hash_decimals: int = Field(
        4,
        ge=0,
        le=12,
        description="Decimal places used when hashing obs tensors for cache key stability",
    )


class RewardConfig(StrictBaseModel):
    """Multi-objective reward configuration (Pillar 6)."""

    weight_truthfulness: float = Field(0.4, ge=0, le=1, description="Truth reward weight")
    weight_helpfulness: float = Field(0.3, ge=0, le=1, description="Help reward weight")
    weight_safety: float = Field(0.2, ge=0, le=1, description="Safety reward weight")
    weight_engagement: float = Field(0.1, ge=0, le=1, description="Engagement reward weight")
    weight_vlm_progress: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="VLM progress reward weight (off by default for safety)",
    )
    vlm_progress: VLMProgressConfig = Field(
        default_factory=VLMProgressConfig,
        description="VLM-derived progress reward head settings",
    )


class SafetyProjectorConfig(StrictBaseModel):
    """Geometric safety action projector configuration (Tier C2 / C2.1).

    Geometric constraint projection is the right fit for continuous action
    spaces: it is a pure function of the frozen :class:`SafetyContext` plus
    the proposed action — no Lagrangian variable, no state across ticks.
    Clamping is deterministic and stateless. When ``enabled=False`` (the
    default), the orchestrator never builds a projector and the tick body
    short-circuits the projection seam, so existing deployments produce
    byte-identical actions.
    """

    enabled: bool = Field(
        False,
        description=(
            "Enable the soft-constraint safety projector. Default ``False`` "
            "preserves byte-identical pre-PR behaviour."
        ),
    )
    lidar_brake_distance_m: float = Field(
        0.30,
        gt=0,
        description=(
            "Forward-velocity clamp kicks in when ``lidar_min_dist_m`` falls "
            "below this threshold (m)."
        ),
    )
    crawl_velocity_mps: float = Field(
        0.10,
        ge=0,
        description=(
            "Maximum forward velocity (m/s) the projector permits when LiDAR "
            "clearance is low or ``forward_clearance_ok`` is ``False``."
        ),
    )
    human_keepout_m: float = Field(
        1.0,
        gt=0,
        description=(
            "Human-proximity clamp activates when ``human_detected`` is True "
            "AND ``human_dist_m`` is below this distance (m)."
        ),
    )
    human_proximity_speed_mps: float = Field(
        0.05,
        ge=0,
        description=(
            "Per-component magnitude cap (m/s) applied to every action "
            "dimension when a human is inside the keepout radius."
        ),
    )
    tight_quarters_dist_m: float = Field(
        0.50,
        gt=0,
        description=(
            "Rotational clamp activates when ``lidar_min_dist_m`` is below "
            "this distance (m) — operating in tight corridors."
        ),
    )
    tight_quarters_omega_max_rads: float = Field(
        0.50,
        ge=0,
        description=("Maximum angular velocity magnitude (rad/s) permitted in tight quarters."),
    )


class SafetyConfig(StrictBaseModel):
    """Safety monitor thresholds."""

    min_forward_clearance_m: float = Field(0.20, gt=0, description="Min obstacle clearance (m)")
    max_velocity_mps: float = Field(0.5, gt=0, description="Max allowed velocity (m/s)")
    sensor_stale_s: float = Field(0.5, gt=0, description="Sensor staleness threshold (s)")
    max_loop_time_ms: float = Field(200.0, gt=0, description="Max loop time before emergency (ms)")
    max_loop_time_factor: float | None = Field(
        None,
        gt=1.0,
        description=(
            "Multiplier on the control period (1 / loop.control_hz) used to "
            "DERIVE max_loop_time_ms. When set, Settings' root validator "
            "overwrites max_loop_time_ms with factor / control_hz * 1000, so "
            "the overrun threshold tracks the tick rate instead of drifting "
            "when control_hz changes. None (default) keeps the literal "
            "max_loop_time_ms, so every existing YAML resolves identically. "
            "The shipped 30 Hz loop with the historical 200 ms threshold "
            "corresponds to factor=6.0 — that 6x relationship was previously "
            "undocumented folklore rather than a stated rule."
        ),
    )
    loop_overrun_consecutive_ticks: int = Field(
        1,
        ge=1,
        description=(
            "Consecutive ticks whose measured duration must exceed "
            "max_loop_time_ms before the monitor raises an emergency stop. "
            "1 (default) preserves the pre-feature single-sample trip. Raise "
            "it to debounce isolated GC or page-fault spikes on a loaded "
            "Jetson without widening the threshold itself."
        ),
    )
    loop_overrun_warmup_ticks: int = Field(
        0,
        ge=0,
        description=(
            "Ticks from loop start during which a loop overrun is logged and "
            "counted but never raises an emergency stop. Covers lazy CUDA "
            "context creation and TensorRT/ONNX kernel warm-up, where a slow "
            "first tick is expected rather than a fault. 0 (default) "
            "preserves pre-feature behaviour."
        ),
    )
    loop_soft_budget_factor: float = Field(
        1.0,
        gt=0,
        description=(
            "Multiplier on the control period defining the SOFT per-tick "
            "budget. Exceeding it increments mousedroid_tick_overruns_total "
            "and logs loop_budget_exceeded; it never raises an emergency "
            "stop. This is the signal that catches a loop degrading from "
            "30 Hz toward 5 Hz — a band that sits below max_loop_time_ms and "
            "so trips nothing today."
        ),
    )
    min_valid_sensors: int = Field(2, ge=0, description="Min valid sensors for operation")
    gpu_warn_temp_c: float = Field(75.0, gt=0, description="GPU warning temperature (C)")
    gpu_critical_temp_c: float = Field(90.0, gt=0, description="GPU critical temperature (C)")
    distance_fallback_m: float = Field(
        999.0,
        gt=0,
        description="Distance value used when the ultrasonic sensor is unavailable",
    )
    battery_warn_v: float = Field(
        10.5,
        ge=0,
        description="Battery warning voltage (V); 0 disables",
    )
    battery_critical_v: float = Field(
        9.5,
        ge=0,
        description="Battery critical voltage (V); 0 disables",
    )
    battery_implausible_below_v: float = Field(
        1.0,
        ge=0,
        description=(
            "Readings strictly below this are treated as MISSING TELEMETRY, "
            "not a flat pack — a rover whose Jetson is powered enough to run "
            "this code cannot genuinely read ~0 V, so such a value means the "
            "comms layer had nothing to report (timeout, degraded-mode skip, "
            "or a firmware/command-set mismatch returning an unparseable "
            "frame). Without this floor a comms fault masquerades as "
            "``battery_critical`` on every tick and latches a permanent "
            "emergency stop, while the operator is pointed at the battery. "
            "Set to 0 to disable the plausibility check and restore the "
            "pre-F-025 behaviour of trusting every reading."
        ),
    )
    default_battery_v: float = Field(
        12.6,
        gt=0,
        description="Default battery voltage when sensor data is unavailable (V)",
    )
    reverse_velocity: float = Field(
        -0.5, le=0, description="Reverse velocity for obstacle avoidance"
    )
    action_min: list[float] | None = Field(
        None,
        description=(
            "Per-dimension lower bounds for normalized actions. "
            "None expands to -1.0 for each action dimension."
        ),
    )
    action_max: list[float] | None = Field(
        None,
        description=(
            "Per-dimension upper bounds for normalized actions. "
            "None expands to 1.0 for each action dimension."
        ),
    )
    lidar_max_range_m: float = Field(
        12.0, gt=0, description="LiDAR max range for clearance conversion (m)"
    )
    sensor_recovery_attempts: int = Field(
        1,
        ge=0,
        description="Max sensor recovery attempts before emergency stop",
    )
    sensor_recovery_delay_s: float = Field(
        0.5,
        gt=0,
        description="Delay between sensor recovery attempts (s)",
    )
    projector: SafetyProjectorConfig = Field(
        default_factory=_settings_default_factory(SafetyProjectorConfig),
        description=(
            "Geometric safety action projection block (Tier C2). "
            "Default ``projector.enabled=false`` preserves byte-identical "
            "pre-C2 behaviour — the orchestrator skips the projection seam "
            "entirely when disabled."
        ),
    )

    @model_validator(mode="after")
    def thresholds_ordered(self) -> Self:
        """Reject threshold sets that are individually valid but jointly absurd.

        Every field here carries its own bound, so a swapped pair passes field
        validation and only misbehaves at runtime: with ``gpu_warn_temp_c``
        above ``gpu_critical_temp_c`` the monitor warns at the higher
        temperature and criticals at the lower one, and the operator is given
        no signal that the config is inverted.

        Battery thresholds honour their documented ``0 disables`` semantics —
        a disabled threshold is skipped rather than forced into the ordering,
        so existing YAML that switches one off keeps loading unchanged.
        """
        if self.gpu_warn_temp_c >= self.gpu_critical_temp_c:
            msg = (
                f"gpu_warn_temp_c ({self.gpu_warn_temp_c}) must be below "
                f"gpu_critical_temp_c ({self.gpu_critical_temp_c})"
            )
            raise ValueError(msg)

        # Ascending severity: implausible-floor < critical < warn. Compare only
        # adjacent *enabled* pairs (0 disables), so switching one threshold off
        # never makes the remaining two unsatisfiable.
        enabled = [
            (name, value)
            for name, value in (
                ("battery_implausible_below_v", self.battery_implausible_below_v),
                ("battery_critical_v", self.battery_critical_v),
                ("battery_warn_v", self.battery_warn_v),
            )
            if value > 0
        ]
        for (lower_name, lower), (upper_name, upper) in pairwise(enabled):
            if lower >= upper:
                msg = (
                    f"{lower_name} ({lower}) must be below {upper_name} ({upper}); "
                    "battery thresholds ascend implausible < critical < warn "
                    "(set a threshold to 0 to disable it)"
                )
                raise ValueError(msg)
        return self


class ThreeLawsConfig(StrictBaseModel):
    """Three Laws of Robotics configuration.

    Enforces Asimov's Three Laws with hierarchical priority:
    Law 1 (No Harm) > Law 2 (Obedience) > Law 3 (Self-Preservation).
    """

    enabled: bool = Field(True, description="Enable Three Laws enforcement")
    human_safety_radius_m: float = Field(
        0.5,
        gt=0,
        description="Law 1: min distance to humans (m)",
    )
    emergency_stop_dist_m: float = Field(
        0.15,
        gt=0,
        description="Law 1: emergency stop distance (m)",
    )
    max_safe_acceleration_mps2: float = Field(
        1.0,
        gt=0,
        description="Law 1: max safe acceleration (m/s²)",
    )
    idle_speed_threshold: float = Field(
        0.05,
        gt=0,
        description="Speed below which robot is considered idle (m/s)",
    )
    alert_signal_speed: float = Field(
        0.1,
        gt=0,
        description="Alert nudge speed for inaction harm (m/s)",
    )
    command_blend_weight: float = Field(
        0.8,
        gt=0,
        le=1,
        description="Law 2: human command blend weight",
    )
    battery_preservation_v: float = Field(
        10.5,
        gt=0,
        description="Law 3: battery preservation threshold (V)",
    )
    thermal_critical_c: float = Field(
        85.0,
        gt=0,
        description="Law 3: thermal preservation threshold (°C)",
    )
    smoothing_factor: float = Field(
        0.5,
        gt=0,
        le=1,
        description="Law 3: direction reversal smoothing factor",
    )
    law1_reward_weight: float = Field(
        0.5,
        gt=0,
        le=1,
        description="Law 1 reward penalty weight",
    )
    law2_reward_weight: float = Field(
        0.3,
        gt=0,
        le=1,
        description="Law 2 compliance reward weight",
    )
    law3_reward_weight: float = Field(
        0.2,
        gt=0,
        le=1,
        description="Law 3 preservation reward weight",
    )
    command_diff_threshold: float = Field(
        0.01,
        gt=0,
        description="Min action diff to log a violation",
    )
    thermal_severity_range_c: float = Field(
        15.0,
        gt=0,
        description="Temp range over critical for severity scaling (°C)",
    )
    rapid_reversal_threshold: float = Field(
        1.0,
        gt=0,
        description="Magnitude change triggering reversal smoothing",
    )
    inaction_harm_severity: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Law 1 inaction violation severity",
    )
    law1_override_severity: float = Field(
        0.3,
        ge=0,
        le=1,
        description="Law 2 command-override-by-Law-1 severity",
    )
    zone_boundary_severity: float = Field(
        0.4,
        ge=0,
        le=1,
        description="Law 2 zone boundary clip severity",
    )
    mechanical_stress_severity: float = Field(
        0.3,
        ge=0,
        le=1,
        description="Law 3 reversal smoothing severity",
    )
    battery_damping_factor: float = Field(
        0.5,
        gt=0,
        le=1,
        description="Action scale factor when battery low",
    )
    thermal_damping_factor: float = Field(
        0.5,
        gt=0,
        le=1,
        description="Action scale factor when GPU overheating",
    )
