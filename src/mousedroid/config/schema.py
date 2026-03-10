"""Root configuration schema — single source of truth for all settings.

All values read from YAML config files or environment variables.
Nothing hardcoded elsewhere. New fields MUST have defaults (backwards
compatibility guarantee).
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PlatformType(enum.StrEnum):
    """Supported hardware platform types."""

    MOUSE_DROID = "mouse_droid"


# ---------------------------------------------------------------------------
# Nested config models (alphabetical)
# ---------------------------------------------------------------------------


class CameraConfig(BaseModel):
    """Raspberry Pi AI Camera (IMX500) configuration."""

    resolution_width: int = Field(640, gt=0, description="Capture width (px)")
    resolution_height: int = Field(480, gt=0, description="Capture height (px)")
    fps: int = Field(30, gt=0, le=120, description="Capture frame rate")
    model_path: Path | None = Field(
        None, description="IMX500 onboard model path (None = use default)"
    )
    feature_dim: int = Field(256, gt=0, description="Vision feature vector dimension")
    use_onboard_inference: bool = Field(
        True,
        description="Use IMX500 onboard AI vs Jetson GPU",
    )
    backend: Literal["auto", "picamera2", "jetson_csi"] = Field(
        "auto",
        description="Camera backend: auto-detect, picamera2, or Jetson CSI",
    )


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration for fault tolerance."""

    failure_threshold: int = Field(5, gt=0, description="Failures before opening circuit")
    recovery_timeout_s: float = Field(30.0, gt=0, description="Recovery timeout (s)")
    half_open_max_calls: int = Field(3, gt=0, description="Max calls in half-open state")


class CuriosityConfig(BaseModel):
    """Curiosity-driven exploration configuration (Pillar 8)."""

    intrinsic_reward_scale: float = Field(
        0.1,
        gt=0,
        description="Intrinsic reward scaling factor",
    )
    forward_model_hidden: int = Field(256, gt=0, description="Forward model hidden dim")
    inverse_model_hidden: int = Field(256, gt=0, description="Inverse model hidden dim")


class ESP32Config(BaseModel):
    """ESP32 communication configuration for Wave Rover motor control."""

    protocol: Literal["serial", "wifi"] = Field(
        "serial",
        description="Communication protocol: serial (UART) or wifi (HTTP)",
    )
    serial_port: str = Field("/dev/ttyUSB0", description="Serial port path")
    serial_baud: int = Field(
        1_000_000,
        gt=0,
        description="Serial baud rate (Wave Rover default)",
    )
    wifi_host: str = Field("192.168.4.1", description="ESP32 WiFi AP IP address")
    wifi_port: int = Field(80, gt=0, le=65535, description="ESP32 HTTP port")
    command_timeout_s: float = Field(0.5, gt=0, description="Command ACK timeout (s)")
    keepalive_hz: float = Field(10.0, gt=0, description="Motor command keepalive rate (Hz)")
    max_velocity_mps: float = Field(0.5, gt=0, description="Max velocity magnitude (m/s)")
    max_omega_rads: float = Field(2.0, gt=0, description="Max angular velocity (rad/s)")


class ExperienceConfig(BaseModel):
    """LMDB experience storage configuration."""

    path: str = Field("/home/jetson/mousedroid_experience", description="LMDB storage path")
    map_size_gb: int = Field(20, gt=0, description="LMDB map size (GB)")
    flush_every_n: int = Field(30, gt=0, description="Flush after N records")


class HealthConfig(BaseModel):
    """Health monitoring configuration."""

    check_interval_s: float = Field(5.0, gt=0, description="Health check interval (s)")
    gpu_temp_warn_c: float = Field(75.0, gt=0, description="GPU temp warning threshold (C)")
    gpu_temp_critical_c: float = Field(90.0, gt=0, description="GPU temp critical threshold (C)")
    memory_warn_pct: float = Field(85.0, gt=0, le=100, description="Memory warning threshold (%)")


class JetsonConfig(BaseModel):
    """Nvidia Jetson Orin Nano hardware configuration."""

    tensorrt_enabled: bool = Field(True, description="Enable TensorRT optimization")
    gpu_memory_fraction: float = Field(
        0.5,
        gt=0,
        le=1.0,
        description="GPU memory fraction for PyTorch",
    )
    power_mode: Literal["15W", "7W"] = Field("15W", description="Jetson power mode")
    dla_enabled: bool = Field(False, description="Enable Deep Learning Accelerator")
    thermal_zone_path: Path = Field(
        Path("/sys/devices/virtual/thermal/thermal_zone0/temp"),
        description="Jetson GPU thermal zone sysfs path",
    )
    gpu_load_path: Path = Field(
        Path("/sys/devices/platform/gpu.0/load"),
        description="Jetson GPU load sysfs path",
    )
    precision: Literal["fp32", "fp16", "int8"] = Field(
        "fp16",
        description="TensorRT inference precision",
    )
    workspace_gb: float = Field(1.0, gt=0, description="TensorRT builder workspace (GB)")


class LearningConfig(BaseModel):
    """Continual learning configuration (Pillar 3)."""

    ewc_lambda: float = Field(5000.0, gt=0, description="EWC regularization strength")
    ewc_fisher_samples: int = Field(200, gt=0, description="Samples for Fisher estimation")
    progressive_enabled: bool = Field(False, description="Enable progressive column growth")


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


class MCTSConfig(BaseModel):
    """Monte Carlo Tree Search configuration."""

    n_simulations_base: int = Field(50, gt=0, description="Base MCTS simulations")
    n_simulations_max: int = Field(200, gt=0, description="Max MCTS simulations")
    rollout_depth: int = Field(5, gt=0, description="Rollout depth")
    gamma: float = Field(0.97, gt=0, le=1, description="Discount factor")
    n_action_candidates: int = Field(9, gt=0, description="Action candidates per node")
    ucb_c: float = Field(1.41, gt=0, description="UCB exploration constant")


class MemoryConfig(BaseModel):
    """Layered memory system configuration (Pillar 4)."""

    working_context_size: int = Field(8192, gt=0, description="Working memory context tokens")
    episodic_capacity: int = Field(50_000, gt=0, description="Episodic replay buffer size")
    semantic_dim: int = Field(256, gt=0, description="Semantic embedding dimension")
    consolidation_batch_size: int = Field(32, gt=0, description="Offline consolidation batch")
    consolidation_interval_s: float = Field(60.0, gt=0, description="Consolidation period (s)")


class MetricsConfig(BaseModel):
    """Metrics export configuration."""

    enabled: bool = Field(True, description="Enable metrics collection")
    export_interval_s: float = Field(10.0, gt=0, description="Export interval (s)")


class ModelConfig(BaseModel):
    """Neural network model dimensions."""

    vision_dim: int = Field(256, gt=0, description="Vision feature input dim")
    ultrasonic_dim: int = Field(1, gt=0, description="Ultrasonic input dim")
    motor_state_dim: int = Field(4, gt=0, description="Motor state dim [vx, vy, omega, battery]")
    hidden_dim: int = Field(256, gt=0, description="RNN hidden dim")
    latent_dim: int = Field(64, gt=0, description="Latent state dim")
    action_dim: int = Field(3, gt=0, description="Action dim [vx, vy, omega]")
    obs_dim: int = Field(256, gt=0, description="Fused observation embedding dim")
    vision_proj_dim: int = Field(128, gt=0, description="Vision projection dim")
    ultrasonic_proj_dim: int = Field(32, gt=0, description="Ultrasonic projection dim")
    motor_proj_dim: int = Field(32, gt=0, description="Motor state projection dim")
    belief_dim: int = Field(128, gt=0, description="BDI belief latent dim")
    desire_dim: int = Field(64, gt=0, description="BDI desire latent dim")
    intention_classes: int = Field(10, gt=0, description="BDI intention classes")
    affect_dim: int = Field(2, gt=0, description="BDI affect dim (valence, arousal)")


class RetryConfig(BaseModel):
    """Retry policy configuration."""

    max_attempts: int = Field(3, gt=0, description="Maximum retry attempts")
    base_delay_s: float = Field(1.0, gt=0, description="Base delay between retries (s)")
    max_delay_s: float = Field(30.0, gt=0, description="Maximum delay between retries (s)")
    exponential_base: float = Field(2.0, gt=0, description="Exponential backoff base")


class RewardConfig(BaseModel):
    """Multi-objective reward configuration (Pillar 6)."""

    weight_truthfulness: float = Field(0.4, ge=0, le=1, description="Truth reward weight")
    weight_helpfulness: float = Field(0.3, ge=0, le=1, description="Help reward weight")
    weight_safety: float = Field(0.2, ge=0, le=1, description="Safety reward weight")
    weight_engagement: float = Field(0.1, ge=0, le=1, description="Engagement reward weight")


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


class SafetyConfig(BaseModel):
    """Safety monitor thresholds."""

    min_forward_clearance_m: float = Field(0.20, gt=0, description="Min obstacle clearance (m)")
    max_velocity_mps: float = Field(0.5, gt=0, description="Max allowed velocity (m/s)")
    sensor_stale_s: float = Field(0.5, gt=0, description="Sensor staleness threshold (s)")
    min_valid_sensors: int = Field(2, ge=0, description="Min valid sensors for operation")
    gpu_warn_temp_c: float = Field(75.0, gt=0, description="GPU warning temperature (C)")
    gpu_critical_temp_c: float = Field(90.0, gt=0, description="GPU critical temperature (C)")
    battery_warn_v: float = Field(10.5, gt=0, description="Battery warning voltage (V)")
    battery_critical_v: float = Field(9.5, gt=0, description="Battery critical voltage (V)")
    reverse_velocity: float = Field(
        -0.5, le=0, description="Reverse velocity for obstacle avoidance"
    )


class SurpriseConfig(BaseModel):
    """Surprise / anomaly detection configuration."""

    ema_alpha: float = Field(0.1, gt=0, le=1, description="EMA smoothing factor")
    high_threshold: float = Field(2.0, gt=0, description="High surprise threshold")
    critical_threshold: float = Field(5.0, gt=0, description="Critical surprise threshold")


class ThreeLawsConfig(BaseModel):
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


class PPOConfig(BaseModel):
    """Proximal Policy Optimization configuration for constitutional RL."""

    clip_epsilon: float = Field(0.2, gt=0, le=1, description="PPO clipping epsilon")
    gae_lambda: float = Field(0.95, gt=0, le=1, description="GAE lambda")
    ppo_epochs: int = Field(4, gt=0, description="PPO update epochs per rollout")
    n_rollout_steps: int = Field(128, gt=0, description="Steps per rollout segment")
    n_training_episodes: int = Field(5000, gt=0, description="Total training episodes")
    n_validation_episodes: int = Field(1000, gt=0, description="Held-out validation episodes")


class TrainingConfig(BaseModel):
    """Offline training configuration."""

    batch_size: int = Field(32, gt=0, description="Training batch size")
    learning_rate: float = Field(3e-4, gt=0, description="Learning rate")
    epochs: int = Field(100, gt=0, description="Training epochs")
    checkpoint_every_n: int = Field(10, gt=0, description="Checkpoint frequency")
    kl_beta: float = Field(1.0, gt=0, description="KL loss weight for RSSM training")
    sequence_length: int = Field(50, gt=0, description="Training sequence length")
    n_episodes: int = Field(1000, gt=0, description="Synthetic episodes to generate")
    data_dir: str = Field("training/data", description="Generated data directory")
    weights_dir: str = Field("weights", description="Checkpoint output directory")


class MicrophoneConfig(BaseModel):
    """SuziePi USB 2.0 Mini Microphone configuration."""

    device_index: int | None = Field(None, description="ALSA device index (None=auto-detect)")
    device_name: str = Field("SuziePi", description="USB device name substring for auto-detect")
    sample_rate: int = Field(16000, gt=0, description="Audio sample rate (Hz)")
    channels: int = Field(1, gt=0, le=2, description="Audio channels (1=mono, 2=stereo)")
    chunk_size: int = Field(1024, gt=0, description="Samples per read chunk")
    format: Literal["float32", "int16"] = Field("float32", description="Audio sample format")


class UltrasonicConfig(BaseModel):
    """HC-SR04 ultrasonic distance sensor configuration."""

    trigger_pin: int = Field(..., description="GPIO trigger pin (BCM numbering)")
    echo_pin: int = Field(..., description="GPIO echo pin (BCM numbering)")
    max_range_m: float = Field(4.0, gt=0, description="Maximum detection range (m)")
    min_range_m: float = Field(0.02, gt=0, description="Minimum detection range (m)")
    timeout_s: float = Field(0.1, gt=0, description="Echo timeout (s)")
    speed_of_sound_mps: float = Field(343.0, gt=0, description="Speed of sound (m/s, ~20C)")

    @model_validator(mode="after")
    def range_ordering(self) -> Self:
        """Validate max_range_m > min_range_m."""
        if self.max_range_m <= self.min_range_m:
            msg = "max_range_m must be > min_range_m"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Root Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Root configuration — single source of truth for all settings.

    All values read from YAML config files. Nothing hardcoded elsewhere.
    New fields MUST have defaults (backwards compatibility guarantee).
    """

    model_config = SettingsConfigDict(
        env_prefix="MOUSEDROID_",
        env_nested_delimiter="__",
    )

    platform: PlatformType = Field(
        PlatformType.MOUSE_DROID,
        description="Hardware platform type",
    )
    mock_hardware: bool = Field(False, description="Use mock drivers")
    debug: bool = Field(False, description="Enable debug logging + assertions")

    loop: LoopConfig = Field(default_factory=LoopConfig)  # type: ignore[arg-type]
    model: ModelConfig = Field(default_factory=ModelConfig)  # type: ignore[arg-type]
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)  # type: ignore[arg-type]
    surprise: SurpriseConfig = Field(default_factory=SurpriseConfig)  # type: ignore[arg-type]
    safety: SafetyConfig = Field(default_factory=SafetyConfig)  # type: ignore[arg-type]
    esp32: ESP32Config = Field(default_factory=ESP32Config)  # type: ignore[arg-type]
    ultrasonic: UltrasonicConfig | None = Field(
        None,
        description="Required if mock_hardware=false",
    )
    microphone: MicrophoneConfig | None = Field(
        None,
        description="USB microphone config (None=disabled)",
    )
    camera: CameraConfig = Field(default_factory=CameraConfig)  # type: ignore[arg-type]
    jetson: JetsonConfig = Field(default_factory=JetsonConfig)  # type: ignore[arg-type]
    robot: RobotConfig = Field(default_factory=RobotConfig)  # type: ignore[arg-type]
    experience: ExperienceConfig = Field(default_factory=ExperienceConfig)  # type: ignore[arg-type]
    logging: LoggingConfig = Field(default_factory=LoggingConfig)  # type: ignore[arg-type]
    training: TrainingConfig = Field(default_factory=TrainingConfig)  # type: ignore[arg-type]
    health: HealthConfig = Field(default_factory=HealthConfig)  # type: ignore[arg-type]
    retry: RetryConfig = Field(default_factory=RetryConfig)  # type: ignore[arg-type]
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)  # type: ignore[arg-type]
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)  # type: ignore[arg-type]
    memory: MemoryConfig = Field(default_factory=MemoryConfig)  # type: ignore[arg-type]
    learning: LearningConfig = Field(default_factory=LearningConfig)  # type: ignore[arg-type]
    reward: RewardConfig = Field(default_factory=RewardConfig)  # type: ignore[arg-type]
    curiosity: CuriosityConfig = Field(default_factory=CuriosityConfig)  # type: ignore[arg-type]
    ppo: PPOConfig = Field(default_factory=PPOConfig)  # type: ignore[arg-type]
    three_laws: ThreeLawsConfig = Field(default_factory=ThreeLawsConfig)  # type: ignore[arg-type]

    @model_validator(mode="after")
    def hardware_requires_pins(self) -> Self:
        """Validate that real hardware mode has required sensor configs."""
        if not self.mock_hardware and self.ultrasonic is None:
            msg = "ultrasonic config required when mock_hardware=false"
            raise ValueError(msg)
        return self
