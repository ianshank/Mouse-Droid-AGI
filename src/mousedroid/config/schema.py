"""Root configuration schema — single source of truth for all settings.

All values read from YAML config files or environment variables.
Nothing hardcoded elsewhere. New fields MUST have defaults (backwards
compatibility guarantee).
"""

from __future__ import annotations

import enum
import sys
from pathlib import Path
from typing import Any, Literal

if sys.version_info >= (3, 11):
    from enum import StrEnum
    from typing import Self
else:
    from typing_extensions import Self

    class StrEnum(str, enum.Enum):
        """Backport of enum.StrEnum for Python 3.10."""


from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mousedroid.constants import DEFAULT_UCB_CANDIDATES, DEFAULT_UCB_TARGET_MS


def _settings_default_factory(factory: Any) -> Any:
    """Return nested settings factories unchanged.

    Pydantic accepts model classes directly as ``default_factory`` callables,
    while the current mypy stubs are stricter about the callable signature.
    This helper preserves runtime behaviour and keeps the workaround local.
    """
    return factory


class PlatformType(StrEnum):
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


class CognitiveConfig(BaseModel):
    """Cognitive core configuration (Pillar 2 — dual-cadence BDI + constitutional RL)."""

    enabled: bool = Field(
        False,
        description="Enable cognitive core injection into orchestrator (requires trained weights)",
    )

    weights_dir: Path = Field(
        Path("weights/bdi/"),
        description="Directory containing BDI model weights (.npz files)",
    )
    huggingface_repo: str = Field(
        "ianshank/mousedroid-weights",
        pattern=r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$",
        description="HuggingFace repository ID (format: owner/repo-name)",
    )
    huggingface_subfolder: str = Field(
        "bdi",
        pattern=r"^$|^[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*$",
        description="Subfolder within the HF repo containing BDI .npz files (no '..' allowed)",
    )
    auto_download: bool = Field(
        True,
        description="Auto-download weights from HuggingFace if local weights missing",
    )
    fallback_to_mcts: bool = Field(
        True,
        description="Fall back to MCTS agent if cognitive core initialization fails",
    )
    download_max_retries: int = Field(
        3,
        gt=0,
        description="Max retry attempts for HuggingFace weight downloads",
    )
    download_backoff_base: float = Field(
        2.0,
        gt=0,
        description="Exponential backoff base for download retries (wait = base ^ attempt)",
    )


class CuriosityConfig(BaseModel):
    """Curiosity-driven exploration configuration (Pillar 8)."""

    intrinsic_reward_scale: float = Field(
        0.1,
        gt=0,
        description="Intrinsic reward scaling factor",
    )
    forward_model_hidden: int = Field(256, gt=0, description="Forward model hidden dim")
    inverse_model_hidden: int = Field(256, gt=0, description="Inverse model hidden dim")
    novelty_decay_enabled: bool = Field(
        False,
        description="Enable novelty decay for familiar state regions",
    )
    novelty_decay_rate: float = Field(
        0.01,
        gt=0,
        description="Exponential decay rate per state visitation",
    )
    novelty_min_scale: float = Field(
        0.01,
        gt=0,
        lt=1.0,
        description="Minimum novelty scale floor (prevents total suppression)",
    )


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


class OfflineRLConfig(BaseModel):
    """Offline RL training configuration (CQL / IQL)."""

    algorithm: Literal["cql", "iql"] = Field("cql", description="Offline RL algorithm")
    hidden_dim: int = Field(256, gt=0, description="Hidden layer dimension")
    gamma: float = Field(0.99, gt=0, le=1, description="Discount factor")
    tau: float = Field(0.005, gt=0, le=1, description="Soft target update coefficient")
    learning_rate: float = Field(3e-4, gt=0, description="Learning rate")
    epochs: int = Field(100, gt=0, description="Training epochs")
    batch_size: int = Field(64, gt=0, description="Training batch size")
    terminal_gap_s: float = Field(
        5.0,
        gt=0,
        description="Timestamp gap to mark episode boundaries (s)",
    )
    log_every_n_epochs: int = Field(10, gt=0, description="Log summary every N epochs")
    checkpoint_every_n_epochs: int = Field(20, gt=0, description="Save checkpoint every N epochs")
    cql_alpha: float = Field(1.0, gt=0, description="CQL regularization weight")
    cql_n_random_actions: int = Field(10, gt=0, description="Random actions for CQL logsumexp")
    iql_expectile: float = Field(0.7, gt=0, lt=1, description="IQL expectile for asymmetric loss")
    iql_beta: float = Field(
        3.0,
        gt=0,
        description="IQL inverse temperature for advantage weighting",
    )


class MCTSConfig(BaseModel):
    """Monte Carlo Tree Search configuration."""

    n_simulations_base: int = Field(50, gt=0, description="Base MCTS simulations")
    n_simulations_max: int = Field(200, gt=0, description="Max MCTS simulations")
    rollout_depth: int = Field(5, gt=0, description="Rollout depth")
    gamma: float = Field(0.97, gt=0, le=1, description="Discount factor")
    n_action_candidates: int = Field(9, gt=0, description="Action candidates per node")
    ucb_c: float = Field(1.41, gt=0, description="UCB exploration constant")
    ucb_candidates: list[float] = Field(
        default_factory=lambda: list(DEFAULT_UCB_CANDIDATES),
        description="Candidate UCB exploration constants evaluated during warm-start tuning",
    )
    ucb_target_ms: float = Field(
        DEFAULT_UCB_TARGET_MS,
        gt=0,
        description="Target median planning latency used when selecting a tuned UCB value",
    )


class MemoryConfig(BaseModel):
    """Layered memory system configuration (Pillar 4)."""

    working_context_size: int = Field(8192, gt=0, description="Working memory context tokens")
    episodic_capacity: int = Field(50_000, gt=0, description="Episodic replay buffer size")
    semantic_dim: int = Field(256, gt=0, description="Semantic embedding dimension")
    consolidation_batch_size: int = Field(32, gt=0, description="Offline consolidation batch")
    consolidation_interval_s: float = Field(60.0, gt=0, description="Consolidation period (s)")


class MetricsConfig(BaseModel):
    """Prometheus-compatible metrics export configuration.

    Controls metrics endpoint enablement, naming, and scrape path.  All
    metric names are derived from ``namespace`` so nothing is hardcoded
    outside this class.
    """

    enabled: bool = Field(True, description="Enable /metrics endpoint")
    path: str = Field("/metrics", description="HTTP path for Prometheus scrape endpoint")
    namespace: str = Field(
        "mousedroid",
        description="Prefix applied to all metric names (e.g. mousedroid_loop_time_ms)",
    )
    export_interval_s: float = Field(
        10.0, gt=0, description="[Reserved] Background export interval (s) — not wired to runtime"
    )
    # Individual metric enable/disable toggles (all default-on)
    track_loop_time: bool = Field(True, description="Expose loop_time_ms gauge")
    track_battery: bool = Field(True, description="Expose battery_voltage_v gauge")
    track_ws_clients: bool = Field(True, description="Expose ws_client_count gauge")
    track_frame_drops: bool = Field(True, description="Expose frame_drop_total counter")
    track_safety_violations: bool = Field(
        True, description="Expose safety_violations_total counter"
    )
    track_gpu_temp: bool = Field(True, description="Expose gpu_temp_celsius gauge")


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
    max_loop_time_ms: float = Field(200.0, gt=0, description="Max loop time before emergency (ms)")
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


class TelemetryConfig(BaseModel):
    """WiFi/Ethernet telemetry server configuration for remote monitoring.

    When enabled, exposes REST and WebSocket endpoints for real-time
    sensor data, log streaming, and health metrics. Binds to all
    network interfaces by default (WiFi + Ethernet + localhost).
    """

    enabled: bool = Field(False, description="Enable telemetry server")
    host: str = Field(
        "0.0.0.0",  # noqa: S104
        description="Server bind address (0.0.0.0 = all interfaces)",
    )
    port: int = Field(8080, gt=0, le=65535, description="Server port")
    preferred_interface: str | None = Field(
        None,
        description=(
            "[Reserved] Preferred network interface for mDNS (e.g. wlan0, eth0) — "
            "not wired to runtime"
        ),
    )
    ws_path: str = Field("/ws", description="WebSocket endpoint path")
    api_prefix: str = Field("/api/v1", description="REST API prefix")
    publish_hz: float = Field(
        10.0,
        gt=0,
        le=60,
        description="Telemetry publish rate (Hz)",
    )
    max_clients: int = Field(10, gt=0, description="Maximum concurrent WebSocket clients")
    queue_size: int = Field(64, gt=0, description="Internal publish queue depth")
    serialization: Literal["json", "msgpack"] = Field(
        "json",
        description="WebSocket serialization format",
    )
    api_key: str | None = Field(None, description="Optional API key (None=disabled)")
    mdns_enabled: bool = Field(True, description="Enable mDNS/Zeroconf discovery")
    mdns_service_name: str = Field(
        "MouseDroid Telemetry",
        description="mDNS service display name",
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins",
    )
    log_stream_buffer: int = Field(200, gt=0, description="Ring buffer size for log entries")
    metrics_path: str = Field(
        "/metrics",
        description=(
            "Legacy scrape endpoint path for direct TelemetryServer construction. "
            "Settings.metrics.path is the canonical configuration source."
        ),
    )


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


class GPUConfig(BaseModel):
    """GPU training configuration for Jetson Orin Nano."""

    device: str | None = Field(
        None,
        description="Force torch device (e.g. 'cuda:0', 'cpu'). None = auto-detect",
    )
    enable_amp: bool = Field(
        True,
        description="Enable Automatic Mixed Precision for PyTorch training phases",
    )
    memory_limit_gb: float = Field(
        6.0,
        gt=0,
        description="Max GPU memory budget in GB (Jetson: 8 GB shared, leave 2 GB headroom)",
    )


class TrainingConfig(BaseModel):
    """Offline training configuration."""

    batch_size: int = Field(32, gt=0, description="Training batch size")
    learning_rate: float = Field(3e-4, gt=0, description="Learning rate")
    epochs: int = Field(100, gt=0, description="Training epochs")
    checkpoint_every_n: int = Field(10, gt=0, description="Checkpoint frequency")
    gradient_scale: float = Field(2.0, gt=0, description="Gradient scale for numpy MSE losses")
    kl_beta: float = Field(1.0, gt=0, description="KL loss weight for RSSM training")
    sequence_length: int = Field(50, gt=0, description="Training sequence length")
    n_episodes: int = Field(1000, gt=0, description="Synthetic episodes to generate")
    data_dir: str = Field("training/data", description="Generated data directory")
    weights_dir: str = Field("weights", description="Checkpoint output directory")
    resume_from: str | None = Field(
        None,
        description="Path to checkpoint for resuming interrupted training",
    )
    gpu: GPUConfig = Field(
        default_factory=lambda: GPUConfig(
            device=None,
            enable_amp=True,
            memory_limit_gb=6.0,
        ),
    )


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

    loop: LoopConfig = Field(default_factory=_settings_default_factory(LoopConfig))
    model: ModelConfig = Field(default_factory=_settings_default_factory(ModelConfig))
    mcts: MCTSConfig = Field(default_factory=_settings_default_factory(MCTSConfig))
    surprise: SurpriseConfig = Field(default_factory=_settings_default_factory(SurpriseConfig))
    safety: SafetyConfig = Field(default_factory=_settings_default_factory(SafetyConfig))
    esp32: ESP32Config = Field(default_factory=_settings_default_factory(ESP32Config))
    ultrasonic: UltrasonicConfig | None = Field(
        None,
        description="Required if mock_hardware=false",
    )
    microphone: MicrophoneConfig | None = Field(
        None,
        description="USB microphone config (None=disabled)",
    )
    camera: CameraConfig = Field(default_factory=_settings_default_factory(CameraConfig))
    jetson: JetsonConfig = Field(default_factory=_settings_default_factory(JetsonConfig))
    robot: RobotConfig = Field(default_factory=_settings_default_factory(RobotConfig))
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
    reward: RewardConfig = Field(default_factory=_settings_default_factory(RewardConfig))
    curiosity: CuriosityConfig = Field(default_factory=_settings_default_factory(CuriosityConfig))
    offline_rl: OfflineRLConfig = Field(default_factory=_settings_default_factory(OfflineRLConfig))
    ppo: PPOConfig = Field(default_factory=_settings_default_factory(PPOConfig))
    telemetry: TelemetryConfig = Field(default_factory=_settings_default_factory(TelemetryConfig))
    three_laws: ThreeLawsConfig = Field(default_factory=_settings_default_factory(ThreeLawsConfig))

    @model_validator(mode="after")
    def hardware_requires_pins(self) -> Self:
        """Validate that real hardware mode has required sensor configs."""
        if not self.mock_hardware and self.ultrasonic is None:
            msg = "ultrasonic config required when mock_hardware=false"
            raise ValueError(msg)
        return self
