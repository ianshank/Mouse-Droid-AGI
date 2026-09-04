"""Root configuration — single source of truth for all settings.

All values read from YAML config files or environment variables. Nothing
hardcoded elsewhere. New fields MUST have defaults (backwards compatibility
guarantee).

``Settings`` composes every domain module in this package as typed fields;
splitting the domain models out of this file (see the package ``__init__``)
keeps this module a thin composition root.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mousedroid.config.migration import (
    apply_aliases as apply_aliases,
)
from mousedroid.config.migration import (
    migrate_group_sections as migrate_group_sections,
)
from mousedroid.config.migration import (
    migrate_section_aliases as migrate_section_aliases,
)
from mousedroid.config.migration import (
    migrate_section_transforms as migrate_section_transforms,
)
from mousedroid.config.migration import (
    milliseconds_to_seconds as milliseconds_to_seconds,
)
from mousedroid.config.migration import (
    seconds_to_hz as seconds_to_hz,
)
from mousedroid.config.migration import (
    seconds_to_milliseconds as seconds_to_milliseconds,
)
from mousedroid.config.schema._primitives import PlatformType, Self, _settings_default_factory
from mousedroid.config.schema.arm import (
    ArmConfig,
    ArmCurriculumConfig,
    ArmPerceptionConfig,
    ArmPlanningConfig,
    ArmSimConfig,
    ArmTaskConfig,
    ArmTrainingConfig,
    PPOConfig,
)
from mousedroid.config.schema.cognitive import (
    CognitiveConfig,
    CuriosityConfig,
    MemoryConfig,
    MetacognitiveConfig,
    SurpriseConfig,
)
from mousedroid.config.schema.gcp_cloud import CloudConfig, GCPConfig
from mousedroid.config.schema.hardware import (
    CameraConfig,
    ESP32Config,
    HailoConfig,
    HealthConfig,
    HostEnvConfig,
    JetsonConfig,
    LidarConfig,
    MotorControllerConfig,
    UltrasonicConfig,
    USBCDiscoveryConfig,
)
from mousedroid.config.schema.harness_mcp import HarnessConfig, MCPConfig, OpenClawConfig
from mousedroid.config.schema.learning import (
    GrowthConfig,
    LearningConfig,
    OfflineRLConfig,
    OnDeviceLearningConfig,
)
from mousedroid.config.schema.llm import LLMConfig, MissionConfig, MissionParserConfig, VLAConfig
from mousedroid.config.schema.misc import (
    BaselinesConfig,
    CircuitBreakerConfig,
    DomainRandomizationConfig,
    ExperienceConfig,
    GreetingConfig,
    LoggingConfig,
    LoopConfig,
    RetryConfig,
    RobotConfig,
)
from mousedroid.config.schema.reward_safety import RewardConfig, SafetyConfig, ThreeLawsConfig
from mousedroid.config.schema.sim import RoverConfig
from mousedroid.config.schema.telemetry import MetricsConfig, ObservabilityConfig, TelemetryConfig
from mousedroid.config.schema.training import TrainingConfig, TrainingPipelineConfig
from mousedroid.config.schema.voice import (
    FaceDisplayConfig,
    MicrophoneConfig,
    SpeakerConfig,
    VoiceConfig,
)
from mousedroid.config.schema.world_model import (
    DualStreamTrainingConfig,
    MCTSConfig,
    ModelConfig,
    WorldModelConfig,
    WorldModelMemoryConfig,
)
from mousedroid.constants import MILLISECONDS_PER_SECOND

_TOP_LEVEL_SECTION_ALIASES: dict[str, str] = {
    "arm_hardware": "arm",
    "arm_simulation": "arm_sim",
    "arm_vision": "arm_perception",
    "arm_symbolic_planning": "arm_planning",
    "arm_rl_training": "arm_training",
    "arm_curriculum_learning": "arm_curriculum",
    "arm_tasks": "arm_task",
}

_ROBOT_ARM_GROUP_SECTION_ALIASES: dict[str, str] = {
    "arm": "arm",
    "hardware": "arm",
    "arm_sim": "arm_sim",
    "sim": "arm_sim",
    "arm_perception": "arm_perception",
    "perception": "arm_perception",
    "arm_planning": "arm_planning",
    "planning": "arm_planning",
    "arm_training": "arm_training",
    "training": "arm_training",
    "arm_curriculum": "arm_curriculum",
    "curriculum": "arm_curriculum",
    "arm_task": "arm_task",
    "task": "arm_task",
}

_SECTION_FIELD_ALIASES: dict[str, dict[str, str]] = {
    "safety": {
        "max_loop_time": "max_loop_time_ms",
        "min_clearance_m": "min_forward_clearance_m",
    },
    "health": {
        "gpu_warn_temp_c": "gpu_temp_warn_c",
        "gpu_critical_temp_c": "gpu_temp_critical_c",
    },
    "camera": {
        "width": "resolution_width",
        "height": "resolution_height",
    },
    "esp32": {
        "baud_rate": "serial_baud",
        "timeout_s": "command_timeout_s",
    },
    "telemetry": {
        "ws_endpoint": "ws_path",
        "websocket_path": "ws_path",
        "api_base": "api_prefix",
        "api_base_path": "api_prefix",
        "publish_rate_hz": "publish_hz",
        "telemetry_rate_hz": "publish_hz",
        "max_ws_clients": "max_clients",
        "websocket_queue_size": "queue_size",
        "bind_host": "host",
        "bind_port": "port",
    },
}

_SECTION_FIELD_TRANSFORMS = {
    "loop": {
        "tick_timeout_ms": ("tick_timeout_s", milliseconds_to_seconds),
        "watchdog_interval_ms": ("watchdog_interval_s", milliseconds_to_seconds),
    },
    "safety": {
        "max_loop_time_s": ("max_loop_time_ms", seconds_to_milliseconds),
    },
    "telemetry": {
        "publish_interval_s": ("publish_hz", seconds_to_hz),
    },
}


class Settings(BaseSettings):
    """Root configuration — single source of truth for all settings.

    All values read from YAML config files. Nothing hardcoded elsewhere.
    New fields MUST have defaults (backwards compatibility guarantee).
    """

    model_config = SettingsConfigDict(
        env_prefix="MOUSEDROID_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    platform: PlatformType = Field(
        PlatformType.MOUSE_DROID,
        description="Hardware platform type",
    )
    mock_hardware: bool = Field(False, description="Use mock drivers")
    debug: bool = Field(False, description="Enable debug logging + assertions")

    loop: LoopConfig = Field(default_factory=_settings_default_factory(LoopConfig))
    model: ModelConfig = Field(default_factory=_settings_default_factory(ModelConfig))
    world_model: WorldModelConfig = Field(
        default_factory=_settings_default_factory(WorldModelConfig)
    )
    mcts: MCTSConfig = Field(default_factory=_settings_default_factory(MCTSConfig))
    surprise: SurpriseConfig = Field(default_factory=_settings_default_factory(SurpriseConfig))
    safety: SafetyConfig = Field(default_factory=_settings_default_factory(SafetyConfig))
    esp32: ESP32Config = Field(default_factory=_settings_default_factory(ESP32Config))
    motor: MotorControllerConfig = Field(
        default_factory=_settings_default_factory(MotorControllerConfig),
        description="Async motor controller driver configuration.",
    )
    ultrasonic: UltrasonicConfig | None = Field(
        None,
        description="Required if mock_hardware=false",
    )
    microphone: MicrophoneConfig | None = Field(
        None,
        description="USB microphone config (None=disabled)",
    )
    lidar: LidarConfig | None = Field(
        None,
        description="FHL-LD19 2D LiDAR config (None=disabled)",
    )
    speaker: SpeakerConfig | None = Field(
        None,
        description="USB speaker config (None=disabled)",
    )
    face_display: FaceDisplayConfig | None = Field(
        None,
        description="Optional SSD1306 OLED face display (None=disabled)",
    )
    voice: VoiceConfig = Field(default_factory=_settings_default_factory(VoiceConfig))
    camera: CameraConfig = Field(default_factory=_settings_default_factory(CameraConfig))
    jetson: JetsonConfig = Field(default_factory=_settings_default_factory(JetsonConfig))
    robot: RobotConfig = Field(default_factory=_settings_default_factory(RobotConfig))
    rover: RoverConfig | None = Field(
        None,
        description=(
            "4WD rover sim-to-real configuration (None=disabled, "
            "backwards compatible). Required only when building "
            "``build_rover_env`` for Isaac Lab / MuJoCo training."
        ),
    )
    experience: ExperienceConfig = Field(
        default_factory=_settings_default_factory(ExperienceConfig)
    )
    logging: LoggingConfig = Field(default_factory=_settings_default_factory(LoggingConfig))
    training: TrainingConfig = Field(default_factory=_settings_default_factory(TrainingConfig))
    health: HealthConfig = Field(default_factory=_settings_default_factory(HealthConfig))
    retry: RetryConfig = Field(default_factory=_settings_default_factory(RetryConfig))
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=_settings_default_factory(CircuitBreakerConfig)
    )
    cognitive: CognitiveConfig = Field(default_factory=_settings_default_factory(CognitiveConfig))
    metrics: MetricsConfig = Field(default_factory=_settings_default_factory(MetricsConfig))
    memory: MemoryConfig = Field(default_factory=_settings_default_factory(MemoryConfig))
    learning: LearningConfig = Field(default_factory=_settings_default_factory(LearningConfig))
    llm: LLMConfig = Field(default_factory=_settings_default_factory(LLMConfig))
    vla: VLAConfig = Field(
        default_factory=_settings_default_factory(VLAConfig),
        description=(
            "VLA policy block (Phase 3a). Default backend='none' preserves "
            "legacy nav-agent behaviour."
        ),
    )
    reward: RewardConfig = Field(default_factory=_settings_default_factory(RewardConfig))
    curiosity: CuriosityConfig = Field(default_factory=_settings_default_factory(CuriosityConfig))
    domain_randomization: DomainRandomizationConfig = Field(
        default_factory=_settings_default_factory(DomainRandomizationConfig),
        description=(
            "Per-episode sim-to-real randomization for RSSM data generation; "
            "set ``enabled: false`` for byte-identical legacy behaviour."
        ),
    )
    metacognitive: MetacognitiveConfig = Field(
        default_factory=_settings_default_factory(MetacognitiveConfig)
    )
    mission_parser: MissionParserConfig = Field(
        default_factory=_settings_default_factory(MissionParserConfig)
    )
    mission: MissionConfig = Field(
        default_factory=_settings_default_factory(MissionConfig),
        description=(
            "Mission lifecycle state-machine block (Tier C2). Default "
            "``mission.replan_enabled=false`` preserves byte-identical "
            "pre-C2 behaviour — the orchestrator skips wiring the "
            "MissionLifecycle entirely when disabled."
        ),
    )
    offline_rl: OfflineRLConfig = Field(default_factory=_settings_default_factory(OfflineRLConfig))
    on_device_learning: OnDeviceLearningConfig | None = Field(
        None,
        description=(
            "Phase-6 on-device incremental-learning block. ``None`` (default) "
            "disables — existing YAML loads byte-identical. Populate with "
            "``enabled: true`` to let the rover update its own weights between "
            "cloud retraining cycles, gated by a safety-regression auto-revert."
        ),
    )
    growth: GrowthConfig | None = Field(
        None,
        description=(
            "Growth-pillar knowledge-distillation block. ``None`` (default) "
            "disables — existing YAML loads byte-identical. Populate with "
            "``enabled: true`` to distil the wired VLA teacher policy into a "
            "compact student on a slow-cadence background task; the distilled "
            "student is persisted to a SHA-256 slot, never hot-swapped."
        ),
    )
    observability: ObservabilityConfig | None = Field(
        None,
        description=(
            "Top-level observability config (experiment logger). None (default) "
            "preserves byte-identical pre-feature behavior. Set to enable "
            "MLflow-backed metric logging for training runs."
        ),
    )
    world_model_memory: WorldModelMemoryConfig | None = Field(
        None,
        description=(
            "Bounded-context latent memory block (F-023). ``None`` (default) "
            "disables — existing YAML loads byte-identical and the tick path is "
            "unchanged. Populate with ``enabled: true`` to blend a persistent "
            "sink anchor + compressed rolling history into the carried (h, z) "
            "at the observe seam."
        ),
    )
    ppo: PPOConfig = Field(default_factory=_settings_default_factory(PPOConfig))
    telemetry: TelemetryConfig = Field(default_factory=_settings_default_factory(TelemetryConfig))
    mcp: MCPConfig | None = Field(
        None,
        description="MCP server config (None=disabled, backwards compatible)",
    )
    usbc_discovery: USBCDiscoveryConfig | None = Field(
        None,
        description=(
            "Optional USB-C enumeration gate used by the Jetson smoke "
            "scripts. None disables (backwards compatible)."
        ),
    )
    greeting: GreetingConfig | None = Field(
        None,
        description=(
            "Optional MSE-6 spoken-greeting subsystem "
            "(``scripts/greet_intro.py``). ``None`` (default) disables — "
            "existing YAML loads byte-identical. Populate on an "
            "operator-tools overlay (``config/greeting_pilot.yaml.example``) "
            "to enable the named greeting flow. Pure speech surface today "
            "(no OLED face animation — the operator's dev rover has no "
            "display attached); the ``Greeter`` class exposes a documented "
            "extension point for the face when one is reconnected."
        ),
    )
    three_laws: ThreeLawsConfig = Field(default_factory=_settings_default_factory(ThreeLawsConfig))
    dual_stream_training: DualStreamTrainingConfig = Field(
        default_factory=_settings_default_factory(DualStreamTrainingConfig),
        description=(
            "Dual-stream RSSM training hyper-parameters (reserved; consumed by the "
            "training pipeline when cfc_hidden_dim > 0)"
        ),
    )

    training_pipeline: TrainingPipelineConfig | None = Field(
        None,
        description="GPU pre-training pipeline orchestrator config (ADR-005)",
    )

    # Hardware accelerator configs (optional)
    hailo: HailoConfig | None = Field(
        None,
        description="Hailo-8 neural accelerator config (None=disabled)",
    )

    # GCP Digital Twin config (optional — all cloud features disabled when None)
    gcp: GCPConfig | None = Field(
        None,
        description="GCP Digital Twin config (None=fully offline autonomous mode)",
    )

    # Tier C1 — Closed-loop cloud retraining + OTA weight-update poller block.
    # Default-on with ``weight_update.poll_interval_s = 0.0`` so existing YAML
    # files load with byte-identical pre-Tier-C1 behavior (poller disabled).
    cloud: CloudConfig = Field(
        default_factory=_settings_default_factory(CloudConfig),
        description=(
            "Tier C1 cloud retraining loop config. Default-on with the OTA "
            "poller disabled (``cloud.weight_update.poll_interval_s = 0.0``)."
        ),
    )

    # Robot arm platform configs (optional — only used when platform=robot_arm)
    arm: ArmConfig | None = Field(
        None,
        description="Robot arm hardware config (required when platform=robot_arm)",
    )
    arm_sim: ArmSimConfig | None = Field(
        None,
        description="MuJoCo simulation config for arm training",
    )
    arm_perception: ArmPerceptionConfig | None = Field(
        None,
        description="Arm perception stack config (depth camera, YOLO, pose)",
    )
    arm_planning: ArmPlanningConfig | None = Field(
        None,
        description="Arm symbolic planning config (PDDL, replanner)",
    )
    arm_training: ArmTrainingConfig | None = Field(
        None,
        description="Arm RL training config (SAC+HER hyperparameters)",
    )
    arm_curriculum: ArmCurriculumConfig | None = Field(
        None,
        description="Arm curriculum learning config (progressive difficulty)",
    )
    arm_task: ArmTaskConfig | None = Field(
        None,
        description="Arm task config (Tower of Hanoi / laundry sorting params)",
    )

    # Agent harness — opt-in deterministic exoskeleton around the orchestrator
    # (task tracker, hooks, journal, approval gate, skills). None=disabled.
    harness: HarnessConfig | None = Field(
        None,
        description="Agent harness config (None=disabled, backwards compatible)",
    )

    # OpenClaw integration — multi-channel NL control plane on a dedicated
    # Mac mini host. None=disabled; existing YAML files load unchanged.
    openclaw: OpenClawConfig | None = Field(
        None,
        description="OpenClaw integration config (None=disabled, backwards compatible)",
    )

    # Host env-file durability check (F-017) — WARN-only preflight probe that
    # the deployed docker.env carries the template's key-set. None=disabled;
    # existing YAML files load unchanged.
    host_env: HostEnvConfig | None = Field(
        None,
        description="Host env-file key-set check config (None=disabled, backwards compatible)",
    )

    # Baselines for hardware constraints (F-027)
    baselines: BaselinesConfig | None = Field(
        None,
        description="Hardware constraints baseline config (None=disabled, backwards compatible)",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, data: Any) -> Any:
        """Migrate legacy config keys to current schema keys.

        This keeps older YAML files loadable while allowing internal field
        names to evolve. Canonical keys always take precedence when both are
        provided.
        """
        if not isinstance(data, dict):
            return data

        # Deep copy so the migration helpers (which mutate nested section dicts
        # in place) never leak mutations back to the caller-provided input.
        migrated = copy.deepcopy(data)

        apply_aliases(migrated, _TOP_LEVEL_SECTION_ALIASES)
        migrate_group_sections(migrated, "robot_arm", _ROBOT_ARM_GROUP_SECTION_ALIASES)
        migrate_section_aliases(migrated, _SECTION_FIELD_ALIASES)
        migrate_section_transforms(migrated, _SECTION_FIELD_TRANSFORMS)

        # The ``robot_arm`` container is not a real Settings field; drop it once
        # its nested sections have been lifted to top-level canonical keys.
        # Legacy top-level and per-section aliases are popped by the helpers.
        if isinstance(migrated.get("robot_arm"), dict):
            migrated.pop("robot_arm", None)

        return migrated

    @model_validator(mode="before")
    @classmethod
    def hardware_requires_pins(cls, data: Any) -> Any:
        """Validate that real hardware mode has required sensor configs."""
        if not isinstance(data, dict):
            return data

        raw_mock_hardware = data.get("mock_hardware", False)
        if isinstance(raw_mock_hardware, str):
            mock_hardware = raw_mock_hardware.strip().lower() in {"1", "true", "yes", "on"}
        else:
            mock_hardware = bool(raw_mock_hardware)

        if not mock_hardware and data.get("ultrasonic") is None and data.get("lidar") is None:
            msg = (
                "at least one distance sensor (ultrasonic or lidar) required"
                " when mock_hardware=false"
            )
            raise ValueError(msg)

        return data

    @model_validator(mode="after")
    def action_bounds_match_action_dim(self) -> Self:
        """Populate and validate normalized action bounds against ``model.action_dim``."""
        action_dim = self.model.action_dim
        if self.safety.action_min is None:
            self.safety.action_min = [-1.0] * action_dim
        if self.safety.action_max is None:
            self.safety.action_max = [1.0] * action_dim

        action_min = self.safety.action_min
        action_max = self.safety.action_max
        if len(action_min) != action_dim:
            msg = f"safety.action_min length must equal model.action_dim ({action_dim})"
            raise ValueError(msg)
        if len(action_max) != action_dim:
            msg = f"safety.action_max length must equal model.action_dim ({action_dim})"
            raise ValueError(msg)

        for idx, (lower, upper) in enumerate(zip(action_min, action_max, strict=False)):
            if lower < -1.0 or upper > 1.0:
                msg = f"normalized action bounds must stay within [-1, 1] (index {idx})"
                raise ValueError(msg)
            if lower >= upper:
                msg = f"safety.action_min[{idx}] must be < safety.action_max[{idx}]"
                raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def derive_max_loop_time_from_control_hz(self) -> Self:
        """Derive ``safety.max_loop_time_ms`` from ``loop.control_hz`` when asked.

        ``SafetyConfig`` and ``LoopConfig`` are siblings, and the factory hands
        the monitor only ``cfg.safety``, so the relationship between the tick
        rate and the overrun threshold cannot live on either model alone. It
        belongs here, next to :meth:`action_bounds_match_action_dim`, which
        already populates a safety field from a model field.

        Opt-in: ``max_loop_time_factor`` defaults to ``None``, in which case the
        literal ``max_loop_time_ms`` is used unchanged and every existing YAML
        resolves byte-identically (CLAUDE.md invariant 6). Setting it replaces
        the previously-undocumented 6x relationship between the 30 Hz period
        and the 200 ms threshold with a named, described, validated rule that
        tracks ``control_hz`` instead of silently drifting from it.
        """
        factor = self.safety.max_loop_time_factor
        if factor is None:
            return self
        self.safety.max_loop_time_ms = factor / self.loop.control_hz * MILLISECONDS_PER_SECOND
        return self

    @model_validator(mode="after")
    def openclaw_requires_telemetry_auth(self) -> Self:
        """Refuse to enable OpenClaw's actuation endpoint without telemetry auth.

        ``POST /api/v1/mission`` dispatches real ``GoalVector`` actuation
        commands once ``openclaw.enabled=True`` wires it up
        (``telemetry/server/_lifecycle.py``). That route is gated purely on
        ``openclaw.enabled`` today, independent of ``telemetry.auth`` — so
        without either bearer-token auth (``telemetry.auth.auth_enabled``) or
        the legacy X-API-Key (``telemetry.api_key``) configured, the endpoint
        is reachable unauthenticated to anyone who can reach the telemetry
        server. Mirrors the same class of guard MCP's ``bind_external``
        already enforces via ``_require_token_for_remote``
        (``config/schema/harness_mcp.py``).

        Returns:
            The validated config instance.

        Raises:
            ValueError: If OpenClaw is enabled but neither telemetry auth
                mechanism is configured.
        """
        if self.openclaw is None or not self.openclaw.enabled:
            return self
        auth_cfg = self.telemetry.auth
        bearer_auth_enabled = auth_cfg is not None and auth_cfg.auth_enabled
        legacy_api_key_set = self.telemetry.api_key is not None
        if not bearer_auth_enabled and not legacy_api_key_set:
            msg = (
                "openclaw.enabled=true requires telemetry auth to be configured "
                "(telemetry.auth.auth_enabled=true or telemetry.api_key set) — "
                "the mission-dispatch endpoint must never be reachable "
                "unauthenticated"
            )
            raise ValueError(msg)
        return self
