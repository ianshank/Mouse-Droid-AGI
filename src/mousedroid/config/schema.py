"""Root configuration schema — single source of truth for all settings.

All values read from YAML config files or environment variables.
Nothing hardcoded elsewhere. New fields MUST have defaults (backwards
compatibility guarantee).
"""

from __future__ import annotations

import copy
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

from mousedroid.config.migration import (
    apply_aliases,
    migrate_group_sections,
    migrate_section_aliases,
    migrate_section_transforms,
    milliseconds_to_seconds,
    seconds_to_hz,
    seconds_to_milliseconds,
)
from mousedroid.constants import DEFAULT_UCB_CANDIDATES, DEFAULT_UCB_TARGET_MS

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
    ROBOT_ARM = "robot_arm"


# ---------------------------------------------------------------------------
# Nested config models (alphabetical)
# ---------------------------------------------------------------------------


class CameraConfig(BaseModel):
    """Raspberry Pi AI Camera (IMX500) configuration."""

    resolution_width: int = Field(640, gt=0, description="Capture width (px)")
    resolution_height: int = Field(480, gt=0, description="Capture height (px)")
    fps: int = Field(30, gt=0, le=120, description="Capture frame rate")
    device_path: str = Field(
        "/dev/video0",
        description="Video device path for OpenCV V4L2 fallback capture",
    )
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
    mock_source: Literal["procedural", "screen_capture"] = Field(
        "procedural",
        description=(
            "Source used by the mock driver: 'procedural' synthesises an "
            "animated test pattern; 'screen_capture' streams the host "
            "desktop via PIL.ImageGrab (real photographic content)."
        ),
    )
    feature_extractor: Literal["mean_pool", "tensorrt", "hailo", "auto"] = Field(
        "mean_pool",
        description="Feature extraction backend: mean_pool (fallback), tensorrt, hailo, or auto",
    )
    l2_normalize: bool = Field(True, description="Apply L2 normalization to feature vectors")


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
        description="Enable novelty decay to reduce curiosity for familiar states",
    )
    novelty_decay_rate: float = Field(
        0.01,
        gt=0,
        description="Exponential decay rate per state visit",
    )
    novelty_min_scale: float = Field(
        0.01,
        gt=0,
        le=1,
        description="Minimum curiosity scale after decay",
    )
    novelty_n_bins: int = Field(
        32,
        gt=0,
        description="Discretisation bins per dimension for novelty decay",
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
    mock_battery_v: float = Field(
        12.0,
        gt=0,
        description="Mock driver default battery voltage (V)",
    )
    degraded_timeout_s: float = Field(
        0.05,
        gt=0,
        description="Serial timeout when ESP32 is unresponsive (s)",
    )
    max_consecutive_timeouts: int = Field(
        5,
        gt=0,
        description="Consecutive timeout failures before entering degraded mode",
    )
    degraded_poll_interval_s: float = Field(
        1.0,
        gt=0,
        description="Probe interval while degraded — poll once per N seconds instead of every tick",
    )


class ExperienceConfig(BaseModel):
    """LMDB experience storage configuration."""

    path: str = Field("/home/jetson/mousedroid_experience", description="LMDB storage path")
    map_size_gb: float = Field(
        20.0,
        gt=0,
        description="LMDB map size (GB; fractional values allowed)",
    )
    flush_every_n: int = Field(30, gt=0, description="Flush after N records")
    export_path: str = Field("/tmp/export", description="Default experience export path")  # noqa: S108


class HealthConfig(BaseModel):
    """Health monitoring configuration."""

    check_interval_s: float = Field(5.0, gt=0, description="Health check interval (s)")
    gpu_temp_warn_c: float = Field(75.0, gt=0, description="GPU temp warning threshold (C)")
    gpu_temp_critical_c: float = Field(90.0, gt=0, description="GPU temp critical threshold (C)")
    memory_warn_pct: float = Field(85.0, gt=0, le=100, description="Memory warning threshold (%)")


class HailoConfig(BaseModel):
    """Hailo-8 neural accelerator configuration.

    The Hailo-8 is a 26 TOPS INT8 M.2 accelerator that offloads perception
    workloads (YOLO detection, feature extraction) from the Jetson GPU to
    dedicated silicon, freeing GPU for reasoning (RSSM, MCTS, SAC, LLM).
    """

    enabled: bool = Field(False, description="Enable Hailo-8 accelerator for perception offload")
    device_path: str = Field(
        "/dev/hailo0",
        description="Hailo PCIe device path",
    )
    yolo_hef_path: Path = Field(
        Path("models/hailo/yolo11_disk_detector.hef"),
        description="Path to compiled YOLO HEF model",
    )
    feature_extractor_hef_path: Path = Field(
        Path("models/hailo/feature_extractor.hef"),
        description="Path to compiled feature extractor HEF model",
    )
    batch_size: int = Field(1, gt=0, le=8, description="Inference batch size")
    power_mode: Literal["performance", "balanced", "power_save"] = Field(
        "performance",
        description="Hailo-8 power mode (reserved: future hardware integration)",
    )
    input_format: Literal["uint8", "float32"] = Field(
        "uint8",
        description="Model input data format (reserved: future hardware integration)",
    )
    timeout_ms: float = Field(
        100.0,
        gt=0,
        description="Inference timeout in milliseconds",
    )
    fallback_on_failure: bool = Field(
        True,
        description="Fall back to GPU/CPU pipeline if Hailo inference fails",
    )


class JetsonConfig(BaseModel):
    """Nvidia Jetson Orin Nano hardware configuration."""

    tensorrt_enabled: bool = Field(True, description="Enable TensorRT optimization")
    gpu_memory_fraction: float = Field(
        0.5,
        gt=0,
        le=1.0,
        description="GPU memory fraction for PyTorch (reserved: future CUDA allocator)",
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
    tensorrt_cache_dir: Path = Field(
        Path("/opt/mousedroid/tensorrt_cache"),
        description="Directory for cached TensorRT compiled engines",
    )


class LidarConfig(BaseModel):
    """FHL-LD19 2D LiDAR configuration."""

    enabled: bool = Field(True, description="Enable LiDAR sensor")
    serial_port: str = Field("/dev/ttyUSB1", description="Serial port for LD19")
    baud_rate: int = Field(230400, gt=0, description="UART baud rate")
    max_range_m: float = Field(12.0, gt=0, description="Maximum detection range (m)")
    min_range_m: float = Field(0.15, gt=0, description="Minimum detection range (m)")
    scan_frequency_hz: float = Field(10.0, gt=0, description="Scan frequency (Hz)")
    min_confidence: int = Field(0, ge=0, le=255, description="Minimum point confidence [0-255]")
    read_timeout_s: float = Field(0.2, gt=0, description="Serial read timeout (s)")
    scan_acquisition_timeout_s: float = Field(
        1.0,
        gt=0,
        description="Maximum time to accumulate one LiDAR scan before returning partial data",
    )
    min_scan_coverage_deg: float = Field(
        270.0,
        gt=0,
        le=360.0,
        description="Minimum angular coverage to treat one LiDAR scan as complete",
    )
    scan_timeout_multiplier: float = Field(
        2.0,
        gt=0,
        description=(
            "Multiplier applied to the nominal scan period (1 / scan_frequency_hz) when "
            "computing the acquisition deadline. Increase for slow-spinning or "
            "high-interference environments."
        ),
    )
    n_sectors: int = Field(36, gt=0, description="Number of angular sectors for binning")
    feature_dim: int = Field(36, gt=0, description="Output feature vector dimension")
    mock_pattern: str = Field(
        "uniform",
        description=(
            "Mock LiDAR scan pattern. 'uniform' (default) fills scan at midrange; "
            "'rotating_wedge' rotates a narrow near-obstacle wedge for dashboard "
            "visual validation."
        ),
    )
    mock_rotation_hz: float = Field(
        0.5,
        gt=0,
        description="Rotation frequency (Hz) of the mock rotating-wedge pattern.",
    )

    @model_validator(mode="after")
    def _range_order(self) -> Self:
        """Validate max_range_m > min_range_m."""
        if self.max_range_m <= self.min_range_m:
            msg = "max_range_m must be > min_range_m"
            raise ValueError(msg)
        return self


class LearningConfig(BaseModel):
    """Continual learning configuration (Pillar 3)."""

    ewc_lambda: float = Field(5000.0, gt=0, description="EWC regularization strength")
    ewc_fisher_samples: int = Field(200, gt=0, description="Samples for Fisher estimation")
    progressive_enabled: bool = Field(False, description="Enable progressive column growth")


class LLMConfig(BaseModel):
    """LLM Gateway configuration for NL command interface."""

    enabled: bool = Field(True, description="Enable LLM gateway")
    model_path: Path = Field(
        Path("/opt/mousedroid/models/llama-3-8b-instruct.Q4_K_M.gguf"),
        description="Path to GGUF model file",
    )
    model_url: str = Field(
        "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
        "/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        description="URL to download model from",
    )
    model_checksum: str = Field(
        "",
        description="SHA-256 checksum for model file verification (empty=skip)",
    )
    context_length: int = Field(2048, gt=0, description="Model context window in tokens")
    n_threads: int = Field(4, gt=0, description="CPU threads for inference")
    n_gpu_layers: int = Field(-1, description="GPU layers to offload (-1 = all)")
    max_tokens: int = Field(256, gt=0, description="Max generation tokens")
    temperature: float = Field(0.1, ge=0, le=2, description="Sampling temperature")
    latency_target_ms: float = Field(
        500.0, gt=0, description="Target inference latency in milliseconds"
    )
    stop_tokens: list[str] = Field(
        default_factory=lambda: ["<|end|>", "<|endoftext|>"],
        description="Stop sequences",
    )
    max_command_len: int = Field(512, gt=0, description="Max NL command length in chars")
    max_vx_norm_mps: float = Field(0.5, gt=0, description="Max forward velocity norm (m/s)")
    max_vy_norm_mps: float = Field(0.3, gt=0, description="Max lateral velocity norm (m/s)")
    max_omega_norm_rads: float = Field(
        2.0,
        gt=0,
        description="Max angular velocity norm (rad/s)",
    )
    system_prompt: str = Field(
        "You are a Star Wars MSE-6 Mouse Droid navigation controller. "
        "Given a natural language mission, output a JSON object with keys "
        '"vx" (forward, -1 to 1), "vy" (lateral, -1 to 1), "omega" (rotation, -1 to 1). '
        "Respond with ONLY the JSON object.",
        description="System prompt for LLM mission translation",
    )
    injection_patterns: list[str] = Field(
        default_factory=lambda: [
            r"ignore (previous|above|all) instructions?",
            r"system prompt",
            r"you are now",
        ],
        description="Regex patterns to detect prompt injection attempts",
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
        "/tmp/mousedroid_heartbeat",  # noqa: S108
        # watchdog_mode 'file' or 'auto' fallback
        description="Path for file-based watchdog heartbeat",
    )


class MetacognitiveConfig(BaseModel):
    """Metacognitive loop configuration (self-monitoring capabilities)."""

    n_capabilities: int = Field(
        8,
        gt=0,
        description="Number of self-assessed capability dimensions",
    )
    loop_score_scale: float = Field(
        100.0,
        gt=0,
        description="Scaling factor for metacognitive loop scores",
    )


class MissionParserConfig(BaseModel):
    """NL mission parser configuration for speed and confidence mappings."""

    speed_map: dict[str, float] = Field(
        default_factory=lambda: {
            "slow": 0.3,
            "slowly": 0.3,
            "half speed": 0.5,
            "fast": 0.8,
            "quickly": 0.8,
            "full speed": 1.0,
        },
        description="Mapping of speed modifier keywords to normalised speed values",
    )
    default_speed: float = Field(0.5, gt=0, le=1, description="Default speed when no modifier")
    patrol_speed: float = Field(0.5, gt=0, le=1, description="Default patrol velocity (m/s)")
    avoid_speed: float = Field(0.3, gt=0, le=1, description="Default obstacle avoidance velocity")
    stop_confidence: float = Field(1.0, ge=0, le=1, description="Confidence for stop commands")
    direction_confidence: float = Field(
        0.9,
        ge=0,
        le=1,
        description="Confidence for directional movement commands",
    )
    patrol_confidence: float = Field(0.8, ge=0, le=1, description="Confidence for patrol commands")
    avoid_confidence: float = Field(
        0.7,
        ge=0,
        le=1,
        description="Confidence for obstacle avoidance commands",
    )
    llm_fallback_confidence: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Minimum parser confidence to skip LLM fallback",
    )


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

    enabled: bool = Field(False, description="Enable memory tier (episodic, semantic, working)")
    working_context_size: int = Field(8192, gt=0, description="Working memory context tokens")
    episodic_capacity: int = Field(50_000, gt=0, description="Episodic replay buffer size")
    semantic_dim: int = Field(256, gt=0, description="Semantic embedding dimension")
    consolidation_batch_size: int = Field(32, gt=0, description="Offline consolidation batch")
    consolidation_interval_s: float = Field(60.0, gt=0, description="Consolidation period (s)")
    semantic_retrieve_k: int = Field(
        1,
        gt=0,
        description="Number of nearest neighbours to retrieve from semantic index per tick",
    )
    min_episodic_priority: float = Field(
        1e-6,
        gt=0,
        description="Minimum episodic replay priority (floor above zero for FAISS)",
    )


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
    track_llm_translations: bool = Field(
        True,
        description="Expose llm_translation counters and latency histogram",
    )
    track_lidar: bool = Field(
        True,
        description=(
            "Expose lidar_sector_distance_m (labeled), lidar_min_distance_m, "
            "and lidar_scan_points gauges"
        ),
    )
    track_memory_tier: bool = Field(True, description="Expose memory tier gauges")
    track_voice_events: bool = Field(True, description="Expose voice event counter")
    track_llm_latency: bool = Field(True, description="Expose LLM mission parse latency")
    track_curiosity: bool = Field(True, description="Expose curiosity intrinsic reward gauge")
    track_sensor_recovery: bool = Field(True, description="Expose sensor recovery counter")
    loop_latency_buckets_ms: tuple[float, ...] = Field(
        (1.0, 2.5, 5.0, 10.0, 20.0, 33.0, 50.0, 100.0, 200.0, float("inf")),
        description="Histogram bucket boundaries for control-loop latency (ms)",
    )
    llm_latency_buckets_ms: tuple[float, ...] = Field(
        (25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, float("inf")),
        description="Histogram bucket boundaries for LLM translation latency (ms)",
    )


class ModelConfig(BaseModel):
    """Neural network model dimensions."""

    vision_dim: int = Field(256, gt=0, description="Vision feature input dim")
    ultrasonic_dim: int = Field(1, ge=0, description="Ultrasonic input dim (0=disabled)")
    motor_state_dim: int = Field(4, gt=0, description="Motor state dim [vx, vy, omega, battery]")
    hidden_dim: int = Field(256, gt=0, description="RNN hidden dim")
    latent_dim: int = Field(64, gt=0, description="Latent state dim")
    action_dim: int = Field(3, gt=0, description="Action dim [vx, vy, omega]")
    obs_dim: int = Field(256, gt=0, description="Fused observation embedding dim")
    vision_proj_dim: int = Field(128, gt=0, description="Vision projection dim")
    ultrasonic_proj_dim: int = Field(32, ge=0, description="Ultrasonic projection dim (0=disabled)")
    motor_proj_dim: int = Field(32, gt=0, description="Motor state projection dim")
    audio_dim: int = Field(0, ge=0, description="Audio feature input dim (0=disabled)")
    audio_proj_dim: int = Field(32, ge=0, description="Audio projection dim (0=disabled)")
    lidar_dim: int = Field(0, ge=0, description="LiDAR feature input dim (0=disabled)")
    lidar_proj_dim: int = Field(32, ge=0, description="LiDAR projection dim (0=disabled)")
    belief_dim: int = Field(128, gt=0, description="BDI belief latent dim")
    desire_dim: int = Field(64, gt=0, description="BDI desire latent dim")
    intention_classes: int = Field(10, gt=0, description="BDI intention classes")
    affect_dim: int = Field(2, gt=0, description="BDI affect dim (valence, arousal)")

    # CfC liquid neural network stream (Dual-Stream RSSM)
    cfc_hidden_dim: int = Field(0, ge=0, description="CfC stream hidden dim (0=disabled, pure GRU)")
    cfc_backbone_units: int = Field(64, gt=0, description="CfC backbone MLP hidden units")
    cfc_backbone_layers: int = Field(1, gt=0, description="CfC backbone MLP layer count")
    cfc_mode: Literal["default", "pure", "no_gate"] = Field(
        "default", description="CfC cell mode: default, pure, no_gate"
    )
    cfc_sparsity_level: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Reserved for future AutoNCP/CfC wiring sparsity support; currently unused",
    )

    @model_validator(mode="after")
    def _validate_optional_modalities(self) -> Self:
        """Validate optional modality dimension pairs."""
        if (self.ultrasonic_dim == 0) != (self.ultrasonic_proj_dim == 0):
            msg = (
                "ultrasonic_dim and ultrasonic_proj_dim must both be zero when disabling ultrasonic"
            )
            raise ValueError(msg)

        modality_dims = {
            "ultrasonic": (self.ultrasonic_dim, self.ultrasonic_proj_dim),
            "audio": (self.audio_dim, self.audio_proj_dim),
            "lidar": (self.lidar_dim, self.lidar_proj_dim),
        }
        for modality_name, (input_dim, proj_dim) in modality_dims.items():
            if input_dim > 0 and proj_dim == 0:
                msg = f"{modality_name}_proj_dim must be > 0 when {modality_name}_dim is enabled"
                raise ValueError(msg)

        if self.ultrasonic_dim == 0 and self.lidar_dim == 0:
            msg = "at least one distance modality must be enabled: ultrasonic_dim or lidar_dim"
            raise ValueError(msg)

        return self


class DualStreamTrainingConfig(BaseModel):
    """Dual-stream RSSM training configuration."""

    gru_lr: float = Field(3e-4, gt=0, description="GRU stream learning rate")
    cfc_lr: float = Field(1e-4, gt=0, description="CfC stream learning rate")
    gru_grad_clip: float = Field(10.0, gt=0, description="GRU gradient clip norm")
    cfc_grad_clip: float = Field(1.0, gt=0, description="CfC gradient clip norm")
    cfc_loss_weight_initial: float = Field(
        0.1, ge=0, le=1, description="CfC loss weight at start of training"
    )
    cfc_loss_weight_final: float = Field(
        1.0, ge=0, le=1, description="CfC loss weight at end of warmup"
    )
    cfc_loss_warmup_steps: int = Field(
        10000, ge=0, description="Steps to ramp CfC loss weight from initial to final"
    )
    fallback_check_interval: int = Field(
        1000, gt=0, description="Steps between CfC fallback quality checks"
    )
    fallback_degradation_threshold: float = Field(
        0.05, gt=0, description="Max allowed planning quality drop before CfC fallback"
    )


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


class SurpriseConfig(BaseModel):
    """Surprise / anomaly detection configuration."""

    ema_alpha: float = Field(0.1, gt=0, le=1, description="EMA smoothing factor")
    high_threshold: float = Field(2.0, gt=0, description="High surprise threshold")
    critical_threshold: float = Field(5.0, gt=0, description="Critical surprise threshold")


class TelemetryAuthConfig(BaseModel):
    """Bearer token authentication configuration for the telemetry server.

    When enabled, requires a valid ``Authorization: Bearer <token>`` header
    on all requests except those matching ``exempt_paths``. The token value
    is read from the environment variable named by ``token_env_var``.
    """

    auth_enabled: bool = Field(False, description="Enable bearer token authentication")
    token_env_var: str = Field(
        "MOUSEDROID_TELEMETRY_TOKEN",
        description="Environment variable name containing the bearer token",
    )
    allowed_origins: list[str] = Field(
        default_factory=list,
        description="CORS allowed origins for auth middleware (empty=unrestricted)",
    )
    exempt_paths: list[str] = Field(
        default_factory=lambda: ["/health", "/metrics"],
        description="Paths that bypass authentication",
    )


class TelemetryConfig(BaseModel):
    """WiFi/Ethernet telemetry server configuration for remote monitoring.

    When enabled, exposes REST and WebSocket endpoints for real-time
    sensor data, log streaming, and health metrics. Binds to all
    network interfaces by default (WiFi + Ethernet + localhost).
    """

    enabled: bool = Field(False, description="Enable telemetry server")
    force_real_server: bool = Field(
        False,
        description=(
            "When True, use the real aiohttp TelemetryServer even with "
            "mock_hardware=True. Useful for local dashboard validation."
        ),
    )
    raw_frame_hz: float = Field(
        10.0,
        gt=0,
        le=60,
        description=(
            "Target frame rate (Hz) for the /camera/stream MJPEG endpoint "
            "when the camera driver supports raw-frame capture."
        ),
    )
    vision_feature_max_samples: int = Field(
        256,
        gt=0,
        le=4096,
        description=(
            "Maximum number of vision-feature samples encoded into each "
            "TelemetryFrame.vision_features payload. Larger feature "
            "vectors are uniformly strided down to this size before "
            "serialisation, keeping dashboard bandwidth bounded."
        ),
    )
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
    auth: TelemetryAuthConfig | None = Field(
        None,
        description="Bearer token authentication config (None=disabled)",
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


# ---------------------------------------------------------------------------
# GCP Digital Twin config models
# ---------------------------------------------------------------------------


class GCPPubSubConfig(BaseModel):
    """Google Cloud Pub/Sub configuration for telemetry and experience export."""

    telemetry_topic: str = Field(
        "mousedroid-telemetry",
        description="Pub/Sub topic for telemetry frames",
    )
    experience_topic: str = Field(
        "mousedroid-experience",
        description="Pub/Sub topic for experience records",
    )
    batch_max_messages: int = Field(
        100,
        gt=0,
        description="Max messages per Pub/Sub publish batch",
    )
    batch_max_bytes: int = Field(
        1_048_576,
        gt=0,
        description="Max bytes per publish batch (default 1 MB)",
    )
    batch_max_latency_s: float = Field(
        1.0,
        gt=0,
        description="Max batch latency before flush (s)",
    )
    publish_timeout_s: float = Field(
        10.0,
        gt=0,
        description="Timeout in seconds for individual publish futures",
    )
    ordering_key: str = Field(
        "mousedroid-0",
        description="Message ordering key for ordered delivery",
    )


class GCPStorageConfig(BaseModel):
    """Google Cloud Storage configuration for experience archival."""

    bucket: str = Field(
        "mousedroid-experience",
        description="GCS bucket name for experience shards",
    )
    prefix: str = Field(
        "experience/v1",
        description="Object key prefix for experience data",
    )
    upload_batch_size: int = Field(
        1000,
        gt=0,
        description="Number of experience records per GCS shard file",
    )
    upload_interval_s: float = Field(
        300.0,
        gt=0,
        description="Seconds between GCS shard uploads",
    )
    compression: Literal["none", "gzip", "zstd"] = Field(
        "gzip",
        description="Shard file compression algorithm",
    )


class GCPLoggingConfig(BaseModel):
    """Google Cloud Logging sink configuration."""

    enabled: bool = Field(True, description="Forward structlog events to Cloud Logging")
    log_name: str = Field("mousedroid", description="Cloud Logging log name")
    min_level: str = Field("INFO", description="Minimum log level to forward to cloud")


class GCPMonitoringConfig(BaseModel):
    """Google Cloud Monitoring configuration for metrics export."""

    enabled: bool = Field(True, description="Export metrics to Cloud Monitoring")
    export_interval_s: float = Field(
        60.0,
        gt=0,
        description="Seconds between metric export batches",
    )
    metric_prefix: str = Field(
        "custom.googleapis.com/mousedroid",
        description="Cloud Monitoring custom metric type prefix",
    )


class GCPFirestoreConfig(BaseModel):
    """Firestore configuration for episodic memory synchronisation."""

    enabled: bool = Field(False, description="Sync episodic memory to Firestore")
    collection: str = Field(
        "mousedroid_episodes",
        description="Firestore collection for episode documents",
    )
    sync_interval_s: float = Field(
        120.0,
        gt=0,
        description="Seconds between episodic memory sync batches",
    )
    sync_batch_size: int = Field(
        10,
        gt=0,
        description="Max episodes to sync per batch",
    )


class GCPTrainingConfig(BaseModel):
    """Vertex AI cloud training pipeline configuration."""

    training_bucket: str = Field(
        "mousedroid-training",
        description="GCS bucket for training datasets and checkpoints",
    )
    pipeline_region: str = Field(
        "us-central1",
        description="Vertex AI pipeline region",
    )
    machine_type: str = Field(
        "a2-highgpu-1g",
        description="Training VM machine type (A100 GPU)",
    )
    accelerator_type: str = Field(
        "NVIDIA_TESLA_A100",
        description="GPU accelerator type for training",
    )
    accelerator_count: int = Field(1, gt=0, description="Number of GPUs per training job")
    max_run_hours: float = Field(
        4.0,
        gt=0,
        description="Maximum pipeline runtime in hours",
    )
    schedule_cron: str = Field(
        "0 2 * * *",
        description="Cloud Scheduler cron expression (UTC) for nightly retraining",
    )
    huggingface_repo: str = Field(
        "ianshank/mousedroid-weights",
        pattern=r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$",
        description="HuggingFace Hub repo for weight push after training",
    )
    ewc_enabled: bool = Field(
        True,
        description="Enable EWC Fisher matrix update step in pipeline",
    )


class GCPSimulationConfig(BaseModel):
    """GKE parallel simulation configuration for safety validation."""

    gke_cluster: str = Field(
        "mousedroid-sim",
        description="GKE Autopilot cluster name for sim pods",
    )
    region: str = Field("us-central1", description="GKE cluster region")
    max_parallel_pods: int = Field(
        50,
        gt=0,
        description="Maximum concurrent simulation pods",
    )
    sim_ticks_per_scenario: int = Field(
        300,
        gt=0,
        description="Orchestrator ticks per scenario (300 = 10 s at 30 Hz)",
    )
    results_bucket: str = Field(
        "mousedroid-sim-results",
        description="GCS bucket for simulation campaign results",
    )
    image: str = Field(
        "gcr.io/mousedroid-twin/mousedroid:sim",
        description="Container image for simulation pods",
    )


class GCPConfig(BaseModel):
    """GCP Digital Twin umbrella configuration.

    When ``None`` in ``Settings``, all GCP features are disabled and the droid
    operates in fully autonomous offline mode with zero cloud dependency.
    """

    project_id: str = Field(..., description="GCP project ID (required)")
    credentials_path: Path | None = Field(
        None,
        description="Service account key path (None = use ADC / metadata server)",
    )
    robot_id: str = Field(
        "droid-001",
        description="Unique identifier for this robot instance",
    )
    pubsub: GCPPubSubConfig = Field(
        default_factory=_settings_default_factory(GCPPubSubConfig),
    )
    storage: GCPStorageConfig = Field(
        default_factory=_settings_default_factory(GCPStorageConfig),
    )
    logging: GCPLoggingConfig = Field(
        default_factory=_settings_default_factory(GCPLoggingConfig),
    )
    monitoring: GCPMonitoringConfig = Field(
        default_factory=_settings_default_factory(GCPMonitoringConfig),
    )
    firestore: GCPFirestoreConfig = Field(
        default_factory=_settings_default_factory(GCPFirestoreConfig),
    )
    training: GCPTrainingConfig | None = Field(
        None,
        description="Cloud training pipeline config (None = no cloud training)",
    )
    simulation: GCPSimulationConfig | None = Field(
        None,
        description="GKE simulation config (None = no cloud simulation)",
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=lambda: CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_s=60.0,
            half_open_max_calls=1,
        ),
        description="Circuit breaker for cloud API calls (tuned for higher latency)",
    )
    retry: RetryConfig = Field(
        default_factory=lambda: RetryConfig(
            max_attempts=3,
            base_delay_s=2.0,
            max_delay_s=60.0,
            exponential_base=2.0,
            jitter_fraction=0.1,
        ),
        description="Retry config for cloud API calls",
    )


# ---------------------------------------------------------------------------
# Robot Arm platform config models
# ---------------------------------------------------------------------------


class ArmConfig(BaseModel):
    """Robot arm hardware configuration (SO-ARM100, myCobot, UR5e)."""

    urdf_path: Path = Field(
        Path("urdf/so_arm100.urdf"),
        description="Path to robot arm URDF file",
    )
    dof: int = Field(6, gt=0, le=12, description="Degrees of freedom")
    gripper_type: Literal["parallel", "suction", "soft"] = Field(
        "parallel",
        description="End-effector gripper type",
    )
    max_joint_velocity_rads: float = Field(2.0, gt=0, description="Max joint velocity (rad/s)")
    max_joint_torque_nm: float = Field(5.0, gt=0, description="Max joint torque (Nm)")
    home_position: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        description="Home joint angles (rad) — length must match dof",
    )
    serial_port: str = Field("/dev/ttyUSB1", description="Serial port for arm controller")
    serial_baud: int = Field(115200, gt=0, description="Serial baud rate")
    command_timeout_s: float = Field(1.0, gt=0, description="Command response timeout (s)")

    @model_validator(mode="after")
    def home_matches_dof(self) -> Self:
        """Validate home position length matches DOF."""
        if len(self.home_position) != self.dof:
            msg = f"home_position length ({len(self.home_position)}) must match dof ({self.dof})"
            raise ValueError(msg)
        return self


class ArmSimConfig(BaseModel):
    """MuJoCo simulation configuration for robot arm training."""

    scene_path: Path = Field(
        Path("sim/tower_of_hanoi.xml"),
        description="MuJoCo scene XML path",
    )
    timestep_s: float = Field(0.002, gt=0, description="Physics timestep (s)")
    n_substeps: int = Field(5, gt=0, description="Physics substeps per step")
    render_width: int = Field(640, gt=0, description="Render width (px)")
    render_height: int = Field(480, gt=0, description="Render height (px)")
    domain_randomization: bool = Field(True, description="Enable domain randomization")
    mass_range_pct: float = Field(20.0, ge=0, le=100, description="Mass variation range (%)")
    friction_range: float = Field(0.3, ge=0, description="Friction coefficient variation")
    position_noise_m: float = Field(0.005, ge=0, description="Object position noise (m)")
    lighting_variation: float = Field(0.2, ge=0, le=1, description="Lighting intensity variation")
    camera_pose_noise_deg: float = Field(10.0, ge=0, description="Camera pose noise (degrees)")


class ArmPerceptionConfig(BaseModel):
    """Perception stack configuration for robot arm platform."""

    depth_camera_type: Literal["realsense_d435i", "oak_d", "zed2i", "mock"] = Field(
        "realsense_d435i",
        description="Depth camera hardware type",
    )
    yolo_model_path: Path = Field(
        Path("models/yolo11_disk_detector.pt"),
        description="YOLO model weights path",
    )
    yolo_confidence_threshold: float = Field(
        0.5, gt=0, le=1, description="YOLO detection confidence threshold"
    )
    yolo_nms_iou_threshold: float = Field(
        0.45,
        gt=0,
        le=1,
        description="YOLO NMS IoU threshold for non-maximum suppression",
    )
    yolo_backend: Literal["ultralytics", "hailo", "auto"] = Field(
        "ultralytics",
        description="YOLO inference backend: ultralytics (GPU), hailo (accelerator), or auto",
    )
    pose_estimator: Literal["pnp", "learned"] = Field(
        "pnp",
        description="Pose estimation method",
    )
    pose_tolerance_m: float = Field(0.005, gt=0, description="Pose estimation tolerance (m)")
    detection_fps: float = Field(30.0, gt=0, description="Detection rate (Hz)")
    depth_min_m: float = Field(0.01, gt=0, description="Minimum valid depth (m)")
    depth_max_m: float = Field(10.0, gt=0, description="Maximum valid depth (m)")
    depth_hole_threshold_m: float = Field(
        0.02, gt=0, description="Depth below which pixels are treated as holes (m)"
    )
    depth_filter_kernel_size: int = Field(
        3, gt=0, description="Median filter kernel size for depth noise reduction"
    )
    fallback_depth_m: float = Field(
        0.3, gt=0, description="Fallback depth when centre pixel is invalid (m)"
    )
    invalid_depth_threshold_m: float = Field(
        0.01, ge=0, description="Depth values below this are considered invalid (m)"
    )
    white_brightness_threshold: float = Field(
        200.0, ge=0, le=255, description="Brightness above which garment is classified white"
    )
    white_saturation_threshold: float = Field(
        0.15, ge=0, le=1, description="Saturation below which bright garment is white"
    )
    dark_brightness_threshold: float = Field(
        80.0, ge=0, le=255, description="Brightness below which garment is classified dark"
    )
    # NOTE: yolo_nms_iou_threshold is defined once above (near the YOLO
    # confidence threshold); a second duplicate definition here has been
    # removed to keep a single authoritative field + default.
    default_focal_length: float = Field(500.0, gt=0, description="Default camera focal length (px)")
    default_principal_x: float = Field(320.0, gt=0, description="Default principal point X (px)")
    default_principal_y: float = Field(240.0, gt=0, description="Default principal point Y (px)")


class ArmPlanningConfig(BaseModel):
    """Symbolic planning configuration for robot arm tasks."""

    pddl_domain_path: Path = Field(
        Path("planning/pddl/hanoi_domain.pddl"),
        description="PDDL domain file path",
    )
    planner_backend: Literal["pyperplan", "fast_downward"] = Field(
        "pyperplan",
        description="PDDL solver backend",
    )
    llm_replanner_enabled: bool = Field(
        False,
        description="Enable LLM-based adaptive replanning on execution failure",
    )
    max_replan_attempts: int = Field(3, gt=0, description="Max replanning attempts before abort")
    planning_timeout_s: float = Field(5.0, gt=0, description="Maximum planning time (s)")


class ArmTrainingConfig(BaseModel):
    """RL training configuration for robot arm policies."""

    algorithm: Literal["sac", "ppo", "sac_her"] = Field(
        "sac_her",
        description="RL algorithm (SAC, PPO, or SAC+HER)",
    )
    learning_rate: float = Field(3e-4, gt=0, description="Policy learning rate")
    batch_size: int = Field(256, gt=0, description="Training batch size")
    buffer_size: int = Field(1_000_000, gt=0, description="Replay buffer capacity")
    gamma: float = Field(0.99, gt=0, le=1, description="Discount factor")
    tau: float = Field(0.005, gt=0, le=1, description="Soft target update coefficient")
    total_timesteps: int = Field(1_000_000, gt=0, description="Total training timesteps")
    eval_frequency: int = Field(10_000, gt=0, description="Evaluation frequency (steps)")
    checkpoint_frequency: int = Field(50_000, gt=0, description="Checkpoint save frequency (steps)")
    n_eval_episodes: int = Field(20, gt=0, description="Episodes per evaluation")
    video_frequency: int = Field(50_000, gt=0, description="Video rollout frequency (steps)")
    her_n_sampled_goal: int = Field(4, gt=0, description="HER goal relabeling ratio")
    her_goal_selection: Literal["future", "final", "episode"] = Field(
        "future",
        description="HER goal selection strategy",
    )
    reward_grasp: float = Field(0.1, description="Reward for successful grasp")
    reward_place: float = Field(0.2, description="Reward for correct placement")
    reward_complete: float = Field(1.0, description="Reward for task completion")
    penalty_collision: float = Field(-0.5, description="Penalty for collision")
    penalty_wrong_disk: float = Field(-0.1, description="Penalty for grasping wrong disk")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    weights_dir: str = Field("weights/arm", description="Checkpoint output directory")
    action_delta_min: float = Field(-0.1, description="Minimum action delta per step (rad)")
    action_delta_max: float = Field(0.1, gt=0, description="Maximum action delta per step (rad)")
    distance_penalty_coeff: float = Field(
        0.01, ge=0, description="Dense distance-based reward penalty coefficient"
    )


class ArmCurriculumConfig(BaseModel):
    """Curriculum learning configuration for progressive task difficulty."""

    enabled: bool = Field(True, description="Enable curriculum learning")
    stages: list[int] = Field(
        default_factory=lambda: [1, 2, 3, 5],
        description="Curriculum stages (number of disks per stage)",
    )
    promotion_threshold: float = Field(
        0.8, gt=0, le=1, description="Success rate threshold to advance stage"
    )
    promotion_eval_episodes: int = Field(
        50, gt=0, description="Episodes to evaluate before stage promotion"
    )
    warm_start: bool = Field(True, description="Warm-start from previous stage weights")


class ArmTaskConfig(BaseModel):
    """Task-specific configuration for robot arm manipulation tasks."""

    task_type: Literal["tower_of_hanoi", "laundry_sorting", "pick_place"] = Field(
        "tower_of_hanoi",
        description="Manipulation task type",
    )
    num_disks: int = Field(3, gt=0, le=10, description="Number of disks (Tower of Hanoi)")
    num_pegs: int = Field(3, gt=1, le=5, description="Number of pegs (Tower of Hanoi)")
    peg_positions: list[list[float]] = Field(
        default_factory=lambda: [[0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
        description="Peg XYZ positions (m) — length must match num_pegs",
    )
    num_baskets: int = Field(3, gt=0, le=5, description="Number of sorting baskets (laundry)")
    basket_positions: list[list[float]] = Field(
        default_factory=lambda: [[0.2, -0.2, 0.0], [0.3, -0.2, 0.0], [0.4, -0.2, 0.0]],
        description="Basket XYZ positions (m) — length must match num_baskets",
    )
    max_episode_steps: int = Field(500, gt=0, description="Max steps per episode")
    num_garments: int = Field(5, gt=0, description="Number of garments per episode (laundry)")

    @model_validator(mode="after")
    def positions_match_count(self) -> Self:
        """Validate position list lengths match counts."""
        if len(self.peg_positions) != self.num_pegs:
            msg = (
                f"peg_positions length ({len(self.peg_positions)})"
                f" must match num_pegs ({self.num_pegs})"
            )
            raise ValueError(msg)
        if len(self.basket_positions) != self.num_baskets:
            msg = (
                f"basket_positions length ({len(self.basket_positions)})"
                f" must match num_baskets ({self.num_baskets})"
            )
            raise ValueError(msg)
        return self


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
    require_cuda: bool = Field(
        False,
        description=(
            "Fail training when a CUDA device is unavailable or a non-CUDA device is selected"
        ),
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


class TrainingPipelineConfig(BaseModel):
    """GPU pre-training pipeline orchestrator configuration (ADR-005)."""

    phases: list[str] = Field(
        default_factory=lambda: ["rssm", "warmstart", "bdi", "constitutional_rl"],
        description="Ordered training phases to execute",
    )
    batch_sizes: dict[str, int] = Field(
        default_factory=lambda: {
            "rssm": 16,
            "warmstart": 32,
            "bdi": 32,
            "constitutional_rl": 64,
        },
        description="Per-phase base batch sizes",
    )
    thermal_limit_celsius: float = Field(
        85.0,
        gt=0,
        description="GPU temperature threshold to pause training (Celsius)",
    )
    thermal_pause_seconds: float = Field(
        30.0,
        gt=0,
        description="Seconds to wait when thermal limit exceeded",
    )
    thermal_sysfs_path: str = Field(
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
        description="Sysfs path to read GPU temperature (millidegrees Celsius)",
    )
    vram_headroom_mb: int = Field(
        512,
        gt=0,
        description="VRAM headroom to reserve (MB) — batch tuner avoids using this",
    )
    checkpoint_dir: str = Field(
        "checkpoints",
        description="Directory for phase checkpoint files",
    )
    amp_enabled: bool = Field(
        True,
        description="Enable AMP (Automatic Mixed Precision) for GPU phases",
    )
    resume_from_phase: str | None = Field(
        None,
        description="Phase name to resume from (skips prior phases)",
    )


class TrainingGenerationConfig(BaseModel):
    """Synthetic data generation settings for Phase 0."""

    log_every_n_episodes: int = Field(
        100,
        gt=0,
        description="Episode logging cadence during synthetic data generation",
    )


class TrainingAnnotationConfig(BaseModel):
    """Annotation collection and heuristic labeling settings for Phase 0b."""

    n_episodes: int = Field(500, gt=0, description="Annotation collection episode count")
    max_steps: int = Field(50, gt=0, description="Max steps per annotation episode")
    log_every_n_episodes: int = Field(
        100,
        gt=0,
        description="Episode logging cadence during annotation collection",
    )
    human_safety_radius_m: float = Field(
        0.5,
        gt=0,
        description="Human proximity threshold for the protect_human label (m)",
    )
    battery_warn_v: float = Field(
        10.8,
        ge=0,
        description="Battery threshold for the charge label (V); 0 disables",
    )
    obstacle_clearance_m: float = Field(
        0.25,
        gt=0,
        description="Obstacle distance threshold for the avoid_obstacle label (m)",
    )
    idle_speed_threshold: float = Field(
        0.05,
        ge=0,
        description="Planar speed threshold below which the label may become idle",
    )
    idle_omega_threshold: float = Field(
        0.1,
        ge=0,
        description="Angular speed threshold below which the label may become idle",
    )
    wait_speed_threshold: float = Field(
        0.1,
        ge=0,
        description="Planar speed threshold below which the label may become wait",
    )
    wait_omega_threshold: float = Field(
        0.05,
        ge=0,
        description="Angular speed threshold below which the label may become wait",
    )
    turn_omega_threshold: float = Field(
        0.5,
        ge=0,
        description="Angular speed threshold above which the label becomes turn",
    )
    approach_clear_distance_m: float = Field(
        1.0,
        gt=0,
        description="Obstacle distance threshold above which the path is clear for approach",
    )
    approach_speed_threshold: float = Field(
        0.2,
        ge=0,
        description="Planar speed threshold above which the label may become approach_target",
    )
    backtrack_speed_threshold: float = Field(
        -0.2,
        le=0,
        description="Forward velocity threshold below which the label becomes backtrack",
    )


class TrainingWarmstartConfig(BaseModel):
    """Warm-start statistics and UCB tuning settings for Phase 2."""

    latent_stats_max_episodes: int = Field(
        100,
        gt=0,
        description="Maximum episodes sampled when computing latent statistics",
    )
    tuning_episodes: int = Field(
        100,
        gt=0,
        description="Episodes evaluated per UCB candidate during warm-start tuning",
    )
    rollout_steps: int = Field(
        20,
        gt=0,
        description="Imagined rollout steps per warm-start tuning episode",
    )


class TrainingConstitutionalConfig(BaseModel):
    """Constitutional RL rollout logging and validation context settings."""

    log_every_n_episodes: int = Field(
        100,
        gt=0,
        description="Training episode logging cadence for constitutional RL",
    )
    validation_battery_v: float = Field(
        12.0,
        gt=0,
        description="Battery voltage used for constitutional rollout validation context (V)",
    )
    validation_obstacle_dist_m: float = Field(
        2.0,
        gt=0,
        description="Obstacle distance used for constitutional rollout validation context (m)",
    )
    validation_mcts_sims: int = Field(
        50,
        gt=0,
        description="MCTS simulation count used for constitutional rollout validation context",
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
    generation: TrainingGenerationConfig = Field(
        default_factory=_settings_default_factory(TrainingGenerationConfig)
    )
    annotation: TrainingAnnotationConfig = Field(
        default_factory=_settings_default_factory(TrainingAnnotationConfig)
    )
    warmstart: TrainingWarmstartConfig = Field(
        default_factory=_settings_default_factory(TrainingWarmstartConfig)
    )
    constitutional: TrainingConstitutionalConfig = Field(
        default_factory=_settings_default_factory(TrainingConstitutionalConfig)
    )
    gpu: GPUConfig = Field(
        default_factory=lambda: GPUConfig(
            device=None,
            require_cuda=False,
            enable_amp=True,
            memory_limit_gb=6.0,
        ),
    )


class MicrophoneConfig(BaseModel):
    """USB microphone configuration."""

    enabled: bool = Field(True, description="Enable audio capture from this microphone")
    device_index: int | None = Field(None, description="ALSA device index (None=auto-detect)")
    device_name: str = Field("USB", description="USB device name substring for auto-detect")
    sample_rate: int = Field(16000, gt=0, description="Audio sample rate (Hz)")
    channels: int = Field(1, gt=0, le=2, description="Audio channels (1=mono, 2=stereo)")
    chunk_size: int = Field(1024, gt=0, description="Samples per read chunk")
    format: Literal["float32", "int16"] = Field("float32", description="Audio sample format")
    n_mels: int = Field(64, gt=0, description="Number of mel filter bank bins")
    n_fft: int = Field(512, gt=0, description="FFT window size for mel spectrogram")
    hop_length: int = Field(256, gt=0, description="Hop length for mel spectrogram")


class SpeakerConfig(BaseModel):
    """USB speaker output configuration."""

    enabled: bool = Field(True, description="Enable audio playback through this speaker")
    device_index: int | None = Field(None, description="ALSA device index (None=auto-detect)")
    device_name: str = Field("USB", description="USB device name substring for auto-detect")
    sample_rate: int = Field(22050, gt=0, description="Audio output sample rate (Hz)")
    channels: int = Field(1, gt=0, le=2, description="Audio output channels (1=mono, 2=stereo)")
    chunk_size: int = Field(1024, gt=0, description="Samples per write chunk")
    format: Literal["float32", "int16"] = Field("float32", description="Audio sample format")


class VoiceConfig(BaseModel):
    """Rocky voice engine configuration."""

    enabled: bool = Field(False, description="Enable Rocky voice output")
    cooldown_s: float = Field(5.0, gt=0, description="Min seconds between utterances")
    personality: Literal["rocky"] = Field(
        "rocky", description="Voice personality (only 'rocky' supported)"
    )
    tts_model_path: str | None = Field(
        None, description="Path to piper voice model (None=disable TTS model loading)"
    )
    tts_sample_rate: int = Field(22050, gt=0, description="TTS output sample rate (Hz)")
    queue_size: int = Field(16, gt=0, description="Max queued speech requests")
    queue_poll_timeout_s: float = Field(1.0, gt=0, description="Worker queue poll timeout (s)")
    phrase_overrides: dict[str, list[str]] = Field(
        default_factory=dict, description="Custom phrase overrides by event name"
    )
    intensity_threshold: float = Field(
        0.7,
        ge=0,
        le=1,
        description="Minimum intensity for Rocky voice transform effects",
    )


class FaceDisplayConfig(BaseModel):
    """SSD1306 OLED face-display configuration.

    All thresholds consumed by the affect→expression mapping and the blink
    animation live here so there are no magic numbers in driver or
    controller code. New deployments must opt in by setting ``enabled=True``;
    existing YAML files (which omit the section entirely) remain unaffected
    because :class:`Settings` defaults the field to ``None``.
    """

    enabled: bool = Field(False, description="Enable the face-display subsystem")
    i2c_bus: int = Field(7, ge=0, description="I²C bus index (Jetson Orin Nano header = 7)")
    i2c_address: int = Field(0x3C, ge=0, le=0x7F, description="SSD1306 I²C address")
    width: int = Field(128, gt=0, description="Panel width in pixels")
    height: int = Field(64, gt=0, description="Panel height in pixels")
    rotate: int = Field(0, ge=0, le=3, description="Rotation in 90° steps (0..3)")
    refresh_hz: float = Field(10.0, gt=0, description="Maximum face-controller update rate (Hz)")
    boot_message: str = Field("MSE-6 online", description="Boot banner text")
    idle_blink_interval_s: float = Field(
        4.0,
        ge=0,
        description="Idle blink period (s); 0 disables the blink animation",
    )
    blink_close_duration_s: float = Field(
        0.15, gt=0, description="How long the eyes stay closed during a blink"
    )
    min_dwell_s: float = Field(
        0.6,
        ge=0,
        description="Hysteresis dwell — minimum time on an expression before switching",
    )
    fallback_to_mock_on_error: bool = Field(
        True,
        description="Fall back to the mock driver when the I²C probe fails",
    )
    valence_happy_min: float = Field(
        0.35, ge=-1.0, le=1.0, description="Valence threshold for HAPPY"
    )
    valence_sad_max: float = Field(-0.35, ge=-1.0, le=1.0, description="Valence threshold for SAD")
    arousal_alert_min: float = Field(
        0.55, ge=-1.0, le=1.0, description="Arousal threshold for ALERT"
    )
    arousal_sleepy_max: float = Field(
        -0.45, ge=-1.0, le=1.0, description="Arousal threshold for SLEEPY"
    )
    angry_valence_max: float = Field(
        -0.25, ge=-1.0, le=1.0, description="Valence ceiling for ANGRY"
    )
    angry_arousal_min: float = Field(0.45, ge=-1.0, le=1.0, description="Arousal floor for ANGRY")
    idle_sleepy_after_s: float = Field(
        20.0,
        gt=0,
        description="Idle duration after which the face goes SLEEPY",
    )


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
    reward: RewardConfig = Field(default_factory=_settings_default_factory(RewardConfig))
    curiosity: CuriosityConfig = Field(default_factory=_settings_default_factory(CuriosityConfig))
    metacognitive: MetacognitiveConfig = Field(
        default_factory=_settings_default_factory(MetacognitiveConfig)
    )
    mission_parser: MissionParserConfig = Field(
        default_factory=_settings_default_factory(MissionParserConfig)
    )
    offline_rl: OfflineRLConfig = Field(default_factory=_settings_default_factory(OfflineRLConfig))
    ppo: PPOConfig = Field(default_factory=_settings_default_factory(PPOConfig))
    telemetry: TelemetryConfig = Field(default_factory=_settings_default_factory(TelemetryConfig))
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
