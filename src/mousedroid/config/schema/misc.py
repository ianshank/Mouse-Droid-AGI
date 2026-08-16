"""Miscellaneous configuration models with no single natural pillar home.

Sim-to-real domain randomization, LMDB experience storage, resilience
(circuit breaker + retry) policies, physical chassis parameters, structured
logging, main-loop timing, the operator-tools spoken greeting, and hardware
baselines.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from mousedroid.config.schema._primitives import RangeF


class DomainRandomizationConfig(BaseModel):
    """Per-episode randomization for sim-to-real RSSM pretraining (Phase 1).

    All ranges are configurable so production / mock / mission-specific YAMLs
    can widen or narrow the noise envelope without code changes. Setting
    ``enabled=False`` produces empty :class:`EpisodeParams` and the data
    generator path is byte-identical to the pre-feature output.
    """

    enabled: bool = Field(
        True,
        description="Master switch — when False every sample yields empty EpisodeParams",
    )

    # --- Visual / camera ---
    brightness: RangeF = Field(default_factory=lambda: RangeF(low=0.6, high=1.4))
    contrast: RangeF = Field(default_factory=lambda: RangeF(low=0.7, high=1.3))
    hue_shift_deg: RangeF = Field(default_factory=lambda: RangeF(low=-15.0, high=15.0))
    gaussian_noise_std: RangeF = Field(default_factory=lambda: RangeF(low=0.0, high=0.04))
    motion_blur_px: RangeF = Field(default_factory=lambda: RangeF(low=0.0, high=2.5))

    # --- Camera intrinsics / extrinsics jitter ---
    fov_deg: RangeF = Field(default_factory=lambda: RangeF(low=58.0, high=72.0))
    cam_pitch_deg: RangeF = Field(default_factory=lambda: RangeF(low=-3.0, high=3.0))
    cam_yaw_deg: RangeF = Field(default_factory=lambda: RangeF(low=-2.0, high=2.0))
    cam_height_m: RangeF = Field(default_factory=lambda: RangeF(low=0.085, high=0.115))

    # --- Range sensor (HC-SR04) ---
    ultrasonic_noise_m: RangeF = Field(default_factory=lambda: RangeF(low=0.0, high=0.03))
    ultrasonic_dropout_prob: RangeF = Field(default_factory=lambda: RangeF(low=0.0, high=0.05))

    # --- Mecanum chassis dynamics ---
    wheel_friction: RangeF = Field(default_factory=lambda: RangeF(low=0.7, high=1.3))
    wheel_slip: RangeF = Field(default_factory=lambda: RangeF(low=0.0, high=0.15))
    chassis_mass_kg: RangeF = Field(default_factory=lambda: RangeF(low=2.4, high=3.0))
    motor_gain: RangeF = Field(default_factory=lambda: RangeF(low=0.85, high=1.15))

    # --- Comms latency (ESP32 <-> Jetson) ---
    uart_latency_ms: RangeF = Field(default_factory=lambda: RangeF(low=2.0, high=18.0))
    encoder_dropout_prob: RangeF = Field(default_factory=lambda: RangeF(low=0.0, high=0.02))

    # --- External disturbance ---
    push_force_n: RangeF = Field(default_factory=lambda: RangeF(low=0.0, high=1.5))
    push_event_prob: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Per-episode probability of an external push disturbance occurring",
    )

    # --- Feature-space (post-CNN) noise applied during data generation ---
    feature_noise_std: RangeF = Field(default_factory=lambda: RangeF(low=0.0, high=0.02))


class ExperienceConfig(BaseModel):
    """LMDB experience storage configuration."""

    path: str = Field("/home/jetson/mousedroid_experience", description="LMDB storage path")
    map_size_gb: float = Field(
        20.0,
        gt=0,
        description="LMDB map size (GB; fractional values allowed)",
    )
    flush_every_n: int = Field(30, gt=0, description="Flush after N records")
    export_path: str = Field(
        "/tmp/export",  # noqa: S108 — operator-overridable default path, not a temp-file write
        description="Default experience export path",
    )
    nvme_device: str = Field(
        "/dev/nvme0n1",
        description=(
            "NVMe block device path the PCIe SSD smoke probes via "
            "``smartctl -H``. Operators with secondary NVMe slots or "
            "USB-NVMe enclosures override to point at the correct device. "
            "Lives on ``ExperienceConfig`` because the SSD layout is "
            "primarily about hosting the experience LMDB."
        ),
    )
    nvme_partition: str = Field(
        "/dev/nvme0n1p1",
        description=(
            "NVMe partition path the PCIe SSD smoke probes via "
            "``findmnt -no TARGET``. Operators with non-standard "
            "partition tables (e.g. an ESP first, ext4 second) override "
            "to point at the data partition."
        ),
    )
    diagnostics_subprocess_timeout_s: float = Field(
        10.0,
        gt=0,
        description=(
            "Per-subprocess timeout (seconds) for the diagnostics probes "
            "in ``mousedroid.validation.runtime`` (lspci / lsblk / "
            "smartctl / findmnt). 10 s is generous for healthy tools; "
            "operators on slow USB-NVMe enclosures may bump higher."
        ),
    )


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration for fault tolerance."""

    failure_threshold: int = Field(5, gt=0, description="Failures before opening circuit")
    recovery_timeout_s: float = Field(30.0, gt=0, description="Recovery timeout (s)")
    half_open_max_calls: int = Field(3, gt=0, description="Max calls in half-open state")


class RetryConfig(BaseModel):
    """Retry policy configuration."""

    max_attempts: int = Field(3, gt=0, description="Maximum retry attempts")
    base_delay_s: float = Field(1.0, gt=0, description="Base delay between retries (s)")
    max_delay_s: float = Field(30.0, gt=0, description="Maximum delay between retries (s)")
    exponential_base: float = Field(2.0, gt=0, description="Exponential backoff base")
    jitter_fraction: float = Field(
        0.1,
        ge=0,
        le=1,
        description="Jitter as fraction of delay for retry backoff",
    )


class RobotConfig(BaseModel):
    """Physical robot chassis parameters (Wave Rover)."""

    wheel_base_m: float = Field(0.20, gt=0, description="Wheelbase length (m)")
    track_width_m: float = Field(0.20, gt=0, description="Track width (m)")
    max_speed_mps: float = Field(0.50, gt=0, description="Max speed at full power (m/s)")
    wheel_radius_m: float = Field(0.042, gt=0, description="Wheel radius (m)")
    wheel_type: Literal["mecanum", "standard"] = Field(
        "mecanum",
        description="Wheel type for kinematics",
    )


class LoggingConfig(BaseModel):
    """Structured logging configuration."""

    level: str = Field("INFO", description="Log level")
    format: Literal["json", "console"] = Field("json", description="Output format")


class LoopConfig(BaseModel):
    """Main loop timing configuration."""

    perception_hz: float = Field(30.0, gt=0, description="Vision capture rate (Hz)")
    ultrasonic_hz: float = Field(20.0, gt=0, description="Ultrasonic read rate (Hz)")
    control_hz: float = Field(30.0, gt=0, description="Motor command rate (Hz)")
    planning_hz: float = Field(10.0, gt=0, description="MCTS planning rate (Hz)")
    audio_hz: float = Field(16.0, gt=0, description="Microphone capture rate (Hz)")
    lidar_hz: float = Field(10.0, gt=0, description="LiDAR scan rate (Hz)")
    tick_timeout_s: float = Field(
        1.0,
        gt=0,
        description="Max seconds per tick before triggering emergency stop",
    )
    watchdog_enabled: bool = Field(
        False,
        description="Enable watchdog notifications (systemd or file heartbeat)",
    )
    watchdog_mode: Literal["auto", "systemd", "file", "none"] = Field(
        "auto",
        description=(
            "Watchdog mode: 'auto' (systemd if NOTIFY_SOCKET set, else file), "
            "'systemd', 'file', 'none'"
        ),
    )
    watchdog_interval_s: float = Field(
        10.0,
        gt=0,
        description="Maximum interval between watchdog heartbeats (seconds)",
    )
    watchdog_heartbeat_path: str = Field(
        "/tmp/mousedroid_heartbeat",  # noqa: S108 — operator-overridable default, not a temp-file write
        # watchdog_mode 'file' or 'auto' fallback
        description="Path for file-based watchdog heartbeat",
    )
    watchdog_tolerance_factor: float = Field(
        3.0,
        gt=0,
        description=(
            "Multiplier on watchdog_interval_s used to derive the Docker "
            "HEALTHCHECK staleness threshold. A heartbeat older than "
            "(interval * tolerance_factor) seconds flips the container to "
            "unhealthy. Default 3.0 tolerates three missed beats before "
            "alarming."
        ),
    )
    start_grace_s: float = Field(
        60.0,
        ge=0,
        description=(
            "Grace window (seconds) after container start during which the "
            "heartbeat healthcheck returns success even if the heartbeat "
            "file is absent. Covers the gap between container start and the "
            "first orchestrator tick."
        ),
    )
    start_grace_file: str = Field(
        "/run/mousedroid.start",
        description=(
            "Path the container entrypoint touches at startup; the "
            "healthcheck script reads its mtime as the grace-window anchor. "
            "Configurable for deployments that cannot write to /run."
        ),
    )

    @field_validator("watchdog_heartbeat_path", "start_grace_file")
    @classmethod
    def _validate_shell_safe_path(cls, v: str) -> str:
        """Reject paths with characters unsafe for shell-source env files.

        The healthcheck script dot-sources an env file derived from these
        values; any single quote, backtick, dollar sign, or whitespace
        would be a code-execution path. Whitelist matches what real path
        configurations need: alphanumerics, dot, dash, underscore, slash,
        colon. Validated at YAML load so malicious config fails fast with
        a clear error.
        """
        import re

        if not re.fullmatch(r"^[A-Za-z0-9._/\-:]+$", v):
            msg = f"path {v!r} contains shell-unsafe characters; allowed: [A-Za-z0-9._/-:]"
            raise ValueError(msg)
        return v

    policy_selector: Literal["nav_agent", "vla", "auto"] = Field(
        "nav_agent",
        description=(
            "Action policy selector. 'nav_agent' (default) preserves legacy "
            "behavior. 'vla' routes through the VLA policy and falls back "
            "to nav_agent only on timeout. 'auto' prefers VLA when one is "
            "wired and silently falls back otherwise."
        ),
    )
    inference_timeout_s: float | None = Field(
        None,
        description=(
            "Per-tick VLA inference timeout (seconds). When None the "
            "orchestrator uses 1.0 / control_hz."
        ),
    )


class GreetingConfig(BaseModel):
    """Operator-tools: MSE-6 spoken greeting subsystem (``scripts/greet_intro.py``).

    Drives a one-shot named greeting through the existing
    :class:`RockyVoiceEngine` — a pre-flourish phrase-bank event (default
    ``greeting_excited``) followed by the operator-configured message
    template with names interpolated. Designed to opt-in via a dedicated
    YAML overlay; ``Settings.greeting`` defaults to ``None`` so existing
    YAML files load byte-identical.

    The OLED face controller is NOT wired here — the operator's current
    dev rover has no SSD1306 attached. The :class:`Greeter` class
    exposes an extension point so the face can be added later without
    touching this config.
    """

    enabled: bool = Field(
        False,
        description=(
            "Master switch. ``False`` (default) keeps the greeting subsystem "
            "inert so default YAML files load unchanged. Operators flip to "
            "``True`` on an overlay (see ``config/greeting_pilot.yaml.example``)."
        ),
    )
    names: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of names to greet. Empty by default so the schema "
            "can default-construct in any context; the ``@model_validator`` "
            "below rejects ``enabled=True`` with an empty ``names`` list so a "
            "misconfigured overlay is caught at YAML-load time rather than "
            "surfacing a confusing empty-greeting at runtime. Loaded from "
            "YAML only (no CLI override) per the PR design decision: "
            "operator-edited config is the single source of truth for who "
            "the rover knows about."
        ),
    )
    message_template: str = Field(
        "Hello {names}! I have been waiting to meet you for some time",
        min_length=4,
        description=(
            "Template string with a single ``{names}`` placeholder. The "
            "placeholder is filled by an Oxford-comma list (``A, B, C and D``). "
            "Edit on the overlay to change the wording without code changes."
        ),
    )
    pre_chirp_event: str = Field(
        "greeting_excited",
        description=(
            "Phrase-bank event name to fire as an MSE-6-style audible "
            "flourish before the custom message. Defaults to "
            "``greeting_excited`` (existing entry in "
            "``src/mousedroid/voice/phrase_bank.py``). Set to empty "
            "string to skip the pre-flourish entirely."
        ),
    )
    excitement_intensity: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description=(
            "Intensity passed to ``rocky_transform`` for the custom message "
            "— pushes the phrase past the personality engine's intensity "
            "threshold so names get the excited repetition + exclamation. "
            "Range-gated [0, 1]. Default 0.9 exceeds the GLOBAL "
            "``VoiceConfig.intensity_threshold`` default of 0.7, so the "
            "excited path fires unless an operator has set a per-event "
            "override above 0.9 in ``VoiceConfig.event_intensity_thresholds`` "
            "(note: ``rocky_transform`` is invoked with the message text "
            "directly, not an event name — only the global threshold is "
            "consulted by the greeter today, but raising this comparison "
            "above the configured value is the supported way to suppress "
            "personality effects without disabling the greeter)."
        ),
    )
    inter_chirp_delay_s: float = Field(
        0.25,
        ge=0.0,
        le=5.0,
        description=(
            "Pause (seconds) between the pre-flourish phrase finishing and "
            "the custom message starting. Avoids run-on audio that masks "
            "the chirp's tail. Range-gated [0, 5]."
        ),
    )
    fire_on_startup: bool = Field(
        False,
        description=(
            "Issue #109 lifecycle wiring. When ``True`` (and ``enabled`` is "
            "``True``) the orchestrator fires the greeting ONCE during "
            "``start()`` — before entering the 30 Hz control loop — through "
            "the same voice engine it already manages. Defaults ``False`` so "
            "existing YAML files load unchanged and the hot loop stays "
            "byte-identical (the startup greeting is a one-shot OUTSIDE the "
            "loop). A greeting failure is logged and swallowed; it never "
            "blocks orchestrator startup."
        ),
    )
    startup_timeout_s: float = Field(
        10.0,
        gt=0,
        description=(
            "Issue #109. Upper bound (seconds) on the one-shot startup greeting "
            "fired in ``start()``. The greeting is wrapped in "
            "``asyncio.wait_for`` so a hung TTS engine / blocked ALSA device can "
            "never wedge orchestrator bring-up — on timeout the greeting is "
            "abandoned (and logged) and the control loop starts anyway. Default "
            "10.0s keeps pre-#109 YAML loading unchanged."
        ),
    )

    @model_validator(mode="after")
    def _require_names_when_enabled(self) -> GreetingConfig:
        # All three guards gate on ``enabled`` so a disabled overlay can
        # carry an in-progress / placeholder template without failing
        # YAML-load (code-reviewer round-1 finding #1: an operator setting
        # ``enabled: false`` with a custom template should not be
        # rejected — the template is never read while disabled).
        if not self.enabled:
            return self
        if not self.names:
            msg = "greeting.enabled=true requires a non-empty greeting.names list"
            raise ValueError(msg)
        if "{names}" not in self.message_template:
            msg = "greeting.message_template must contain the '{names}' placeholder"
            raise ValueError(msg)
        # Round-3 review (Gemini): ``.format(names=...)`` at runtime can
        # raise ``KeyError`` / ``ValueError`` / ``IndexError`` if the
        # operator's template also references foreign placeholders (e.g.
        # ``{wrong_key}``, positional ``{0}``, or unbalanced braces).
        # Validate at YAML-load with a probe so the error surfaces where
        # the operator can fix it, not in the live greeter call.
        try:
            self.message_template.format(names="__probe__")
        except (KeyError, ValueError, IndexError) as exc:
            msg = (
                "greeting.message_template formatting failed at config "
                f"load — only the '{{names}}' placeholder is supported "
                f"({type(exc).__name__}: {exc})"
            )
            raise ValueError(msg) from exc
        return self


class BaselinesConfig(BaseModel):
    """Hardware baselines configuration for NemoClaw/OpenShell integration (F-027)."""

    max_memory_query_latency_ms: float = Field(
        150.0,
        gt=0,
        description=(
            "Maximum allowed latency (ms) for memory queries "
            "(episodic/semantic) before triggering a degradation warning."
        ),
    )
