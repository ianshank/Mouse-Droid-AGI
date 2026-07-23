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


from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
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

# ---------------------------------------------------------------------------
# Public Literal type aliases — single source of truth for label values used
# across config schemas and telemetry metric helpers. Keeping these here
# (rather than inlining string literals at each call site) means a backend
# rename only needs to touch this module.
# ---------------------------------------------------------------------------

VLABackendLiteral = Literal["none", "mock", "distilled_onnx"]
"""VLA policy backend identifier. Source of truth: :class:`VLAConfig.backend`.

Includes ``"none"`` (the disabled default). For label values on metrics
that only fire from a *running* backend (e.g.
``mousedroid_vla_timeouts_total{mode}``) use the narrower
:data:`VLAActiveBackendLiteral` alias below."""

VLAActiveBackendLiteral = Literal["mock", "distilled_onnx"]
"""Subset of :data:`VLABackendLiteral` excluding ``"none"``.

Use this for any metric or callback where a value of ``"none"`` is
operationally impossible (the disabled backend cannot run inference, so
it cannot fire a timeout or emit a latency sample). Narrowing at the
call site prevents accidental cardinality growth from spurious
``{mode="none"}`` series."""

ReplayOutcomeLiteral = Literal["ok", "schema_mismatch"]
"""LMDB replay-record deserialization outcome. Drives the
``mousedroid_replay_records_total{outcome}`` Prometheus counter labels.
``"ok"`` = record passed schema-version check; ``"schema_mismatch"`` =
record was skipped because its ``SCHEMA_VERSION`` differed from the
runtime constant in :mod:`mousedroid.experience.record`."""


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
    snapshot_jpeg_quality: int = Field(
        90,
        ge=1,
        le=100,
        description=(
            "Pillow JPEG quality (1-100) for the camera snapshot path used "
            "by ``scripts/verify_sensors.py --sensor camera --save-frame``. "
            "Higher = larger files + more visible focus / banding detail; "
            "operators inspecting a post-adjustment lens issue may bump to "
            "100 for lossless inspection, or drop to 70 in disk-pressed "
            "deployments. Default 90 matches the long-standing Pillow "
            "default used by the telemetry MJPEG stream."
        ),
    )
    v4l2_grayscale_extract: bool = Field(
        True,
        description=(
            "Workaround for the JetsonCSICamera's ``v4l2`` backend when the "
            "sensor is IMX708 (or any sensor that only exposes RG10 Bayer "
            "raw via V4L2). The kernel driver advertises ``YUYV`` at the "
            "active format but the bytes are Bayer-packed, so OpenCV's "
            "YUYV->BGR conversion produces solid green / uniform output. "
            "When ``True`` (default), ``capture_raw_jpeg`` extracts the "
            "green channel of the resulting 3-plane frame as luma and "
            "returns a grayscale-cloned RGB JPEG — the operator sees the "
            "scene (with mosaic artefacts) instead of solid green. Flip to "
            "``False`` once the container rebuilds with the "
            "``nvarguscamerasrc`` GStreamer plugin (or a host-side libargus "
            "capture daemon) so the raw frame is properly debayered + "
            "white-balanced and the workaround can be retired."
        ),
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


class RangeF(BaseModel):
    """Inclusive ``[low, high]`` range for a randomly sampled float parameter."""

    low: float = Field(description="Inclusive lower bound")
    high: float = Field(description="Inclusive upper bound")

    @model_validator(mode="after")
    def _check_ordered(self) -> Self:
        if self.low > self.high:
            msg = f"RangeF.low ({self.low}) must be <= high ({self.high})"
            raise ValueError(msg)
        return self


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


class ESP32Config(BaseModel):
    """ESP32 communication configuration for Wave Rover motor control."""

    enabled: bool = Field(
        True,
        description=(
            "Enable the ESP32 motor-controller driver. Default ``True`` "
            "preserves byte-identical pre-PR-104-harden-2 behaviour. Operators "
            "running the orchestrator on a Jetson WITHOUT the ESP32 plugged "
            "in (dev / dashboard verification) flip this to ``False`` so the "
            "factory swaps in :class:`MockESP32Driver` regardless of "
            "``mock_hardware`` — avoids the prior workaround of monkey-"
            "patching ``orchestrator.start()`` to swallow connect failures. "
            "The mock driver short-circuits ``connect()`` / ``send_velocity`` "
            "/ ``emergency_stop`` so the orchestrator can tick at full speed "
            "while features-only smokes (camera + LiDAR + Hailo) run live."
        ),
    )
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
    smoke_test_velocity_mps: float = Field(
        0.05,
        ge=0,
        description=(
            "Target forward velocity for the rover hardware smoke test "
            "(see tests/hardware/test_motor_smoke.py). Kept low so an "
            "untethered rover can stop within tabletop bounds. Set to "
            "0.0 to permanently lock the smoke harness into zero-motion "
            "mode (preferred for benches with no roll-off protection); "
            "the runtime ``allow_motion`` gate in assert_power_chain "
            "remains authoritative regardless of this setpoint."
        ),
    )
    smoke_test_settle_s: float = Field(
        0.5,
        gt=0,
        description="Settle time after sending velocity before reading encoders (s)",
    )
    smoke_test_min_velocity_fraction: float = Field(
        0.5,
        gt=0,
        le=1.0,
        description=(
            "Minimum encoder velocity expressed as a fraction of the "
            "setpoint that the smoke test asserts on real hardware."
        ),
    )
    smoke_test_allow_motion: bool = Field(
        False,
        description=(
            "Hard safety gate for tests/hardware/test_motor_smoke.py. "
            "When False (default), the velocity round-trip stops short of "
            "actually sending a non-zero command — useful when the rover is "
            "on a table or otherwise unattended. Set True (YAML override or "
            "MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true) only when the "
            "rover is on rollers / tethered / monitored."
        ),
    )
    emergency_stop_budget_ms: float = Field(
        50.0,
        gt=0,
        description="Maximum acceptable latency for emergency_stop ack (ms)",
    )
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
    debug_log_max_chars: int = Field(
        200,
        ge=16,
        le=4096,
        description=(
            "Maximum character length for raw serial-line payloads emitted in "
            "DEBUG / WARNING log events (``esp32_raw_line``, "
            "``esp32_non_json_response``, ``esp32_response_not_object``). The "
            "default 200 keeps log files compact during normal smoke runs; "
            "increase to 1024+ when triaging firmware-protocol drift where "
            "the full payload matters. Lower bound 16 ensures the truncated "
            'string carries at least the JSON-framing bytes ``{"T": ...}``.'
        ),
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
    synthetic_input_shape: tuple[int, int, int] = Field(
        (640, 640, 3),
        description=(
            "Zero-tensor shape (height, width, channels) the Hailo-8 smoke "
            "feeds to ``infer_sync('yolo', ...)`` when the runtime does not "
            "expose its input-vstream shape via reflection. Default matches "
            "the YOLO11-disk-detector input contract; operators with a "
            "custom HEF override to match their compiled model."
        ),
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
    ewc_fisher_batch_size: int = Field(
        1, gt=0, description="Batch size for Fisher estimation random inputs"
    )
    ewc_fallback_input_dim: int = Field(
        1,
        gt=0,
        description=(
            "Fallback input dimension when model has no Linear layers for Fisher estimation"
        ),
    )
    progressive_enabled: bool = Field(False, description="Enable progressive column growth")


class OnDeviceLearningConfig(BaseModel):
    """On-device incremental-learning configuration (Phase 6).

    Lets the rover update its own policy/world-model weights *between* cloud
    retraining cycles from fresh on-device experience, gated by a
    safety-regression bound that reverts to cloud weights on underperformance.
    Default-OFF and backwards-compatible — wired as an ``Optional`` block on
    ``Settings`` so existing YAML loads byte-identically. Every value is
    config-driven; ``slot_dir`` is deliberately repo-relative (resolved by the
    factory/orchestrator under the configured experience root
    ``ExperienceConfig.path``) so NO absolute host path is hardcoded.
    """

    enabled: bool = Field(
        False,
        description="Master switch for the on-device incremental-learning loop (default-off)",
    )
    trigger_min_new_records: int = Field(
        500,
        gt=0,
        description=(
            "Minimum fresh experience records that must accumulate before an "
            "on-device update cycle is triggered"
        ),
    )
    check_interval_s: float = Field(
        300.0,
        gt=0,
        description=(
            "Slow-cadence period (seconds) between replay-trigger checks. The "
            "on-device update runs on its own background task OUTSIDE the 30 Hz "
            "hot loop; this is how often the coordinator probes the new-record "
            "count against ``trigger_min_new_records``. Defaults to 5 min so a "
            "default-on deployment never busy-polls the replay store."
        ),
    )
    update_steps: int = Field(
        50,
        gt=0,
        description="Number of bounded gradient steps per on-device update cycle",
    )
    regression_tolerance: float = Field(
        0.05,
        ge=0,
        description=(
            "Maximum allowed held-out recon+KL loss INCREASE above the live "
            "baseline before the on-device candidate is reverted (LOWER loss is "
            "better): PROMOTE iff candidate_loss <= baseline_loss + tolerance. "
            "ge=0 permits a zero-tolerance gate"
        ),
    )
    held_out_fraction: float = Field(
        0.1,
        gt=0,
        le=1,
        description=(
            "Fraction of the replay sample reserved for held-out scoring "
            "(0 < f <= 1). CONFIG SEAM, not wired into the WS-E3 recon-loss gate "
            "— the gate derives its disjoint held-out window from "
            "refine_sequence_length/refine_batch_episodes; retained for a future "
            "fraction-based held-out sizing"
        ),
    )
    ewc_lambda: float = Field(
        1.0,
        ge=0,
        description=(
            "EWC Fisher-penalty strength for the bounded online update "
            "(ge=0 permits an unregularized step)"
        ),
    )
    learning_rate: float = Field(
        1e-4,
        gt=0,
        description="Learning rate for the bounded on-device gradient steps",
    )
    slot_dir: str = Field(
        "on_device_slot",
        description=(
            "Experience-root-relative leaf for the on-device weight slot. NOT an "
            "absolute host path: the factory/orchestrator resolves it UNDER the "
            "configured experience root (``<ExperienceConfig.path>/<slot_dir>``) "
            "so any operator override of the experience path is inherited for "
            "free. On-device-updated weights land here, never overwriting the "
            "cloud-pulled slot."
        ),
    )

    @field_validator("slot_dir")
    @classmethod
    def _validate_slot_dir(cls, v: str) -> str:
        """Reject slot_dir values that escape the experience root.

        ``slot_dir`` is resolved as ``<ExperienceConfig.path>/<slot_dir>``, so
        an absolute path, a parent-traversal (``..``) component, or an empty /
        whitespace-only value would break that containment contract and let
        on-device weights land outside the configured experience root. Validated
        at YAML load so a misconfigured deployment fails fast with a clear,
        operator-actionable message instead of silently writing off-root.
        """
        from pathlib import PurePosixPath, PureWindowsPath

        slot = v.strip()
        # Check absoluteness under BOTH POSIX and Windows semantics so a
        # ``/abs/path`` (slot is resolved on the Jetson/Linux target) is caught
        # regardless of the host OS the config is validated on, and ``..``
        # traversal in either separator style is rejected.
        posix = PurePosixPath(slot)
        windows = PureWindowsPath(slot)
        is_absolute = posix.is_absolute() or windows.is_absolute()
        has_traversal = ".." in posix.parts or ".." in windows.parts
        if not slot or is_absolute or has_traversal:
            msg = (
                "on_device_learning.slot_dir must be a non-empty relative path "
                "without parent traversal (resolved under "
                "ExperienceConfig.path); got " + repr(v)
            )
            raise ValueError(msg)
        return slot

    scoring_seed: int = Field(
        1234,
        description=(
            "Safety-gate scoring: fixed RNG seed making the held-out recon+KL loss "
            "score deterministic. Same seed + same held-out batch + same weights "
            "ALWAYS yields the identical loss, so the promote/revert decision is "
            "reproducible. Config-driven so no seed is hardcoded."
        ),
    )
    enable_hot_swap: bool = Field(
        False,
        description=(
            "Phase-6 ENABLEMENT: master switch for hot-swapping a promoted "
            "on-device candidate into the live world model through the C1 atomic-"
            "swap seam. Promotion (``slot_store.mark_active``) stays SEPARATE from "
            "activation: a candidate can pass the regression gate and be marked "
            "active without ever being swapped into the running model. Default "
            "``False`` keeps the orchestrator byte-identical to #134 — no swap ever "
            "occurs. Requires ``enabled=True`` (validated below)."
        ),
    )
    refine_sequence_length: int = Field(
        16,
        gt=0,
        description=(
            "Phase-6 ENABLEMENT: temporal length T of each ``(B, T, ...)`` sequence "
            "the RSSM refiner assembles from replay for ``train_sequence`` (WS-E2). "
            "Config-driven so the refinement window is never hardcoded."
        ),
    )
    refine_batch_episodes: int = Field(
        4,
        gt=0,
        description=(
            "Phase-6 ENABLEMENT: batch dimension B (number of replay episodes) per "
            "RSSM-refinement sequence batch (WS-E2). Config-driven so the batch "
            "size is never hardcoded."
        ),
    )

    @model_validator(mode="after")
    def _require_enabled_for_hot_swap(self) -> Self:
        """Reject ``enable_hot_swap=True`` while the master switch is off.

        Hot-swapping a candidate into the live model is meaningless unless the
        on-device learning loop that PRODUCES candidates is itself enabled.
        Catching the contradiction at YAML load gives the operator a clear,
        actionable message instead of a silently inert hot-swap flag. Mirrors the
        cross-field gate style of ``_require_endpoints_when_enabled``.
        """
        if self.enable_hot_swap and not self.enabled:
            msg = (
                "on_device_learning.enable_hot_swap=true requires "
                "on_device_learning.enabled=true (hot-swap activates a candidate "
                "the disabled learning loop never produces)"
            )
            raise ValueError(msg)
        return self


class GrowthConfig(BaseModel):
    """Growth-pillar knowledge-distillation configuration.

    Distils the wired VLA teacher policy into a compact student *between* cloud
    retraining cycles, on a slow-cadence background task OUTSIDE the 30 Hz hot
    loop. Default-OFF and backwards-compatible — wired as an ``Optional`` block on
    ``Settings`` so existing YAML loads byte-identically. Every value is
    config-driven; ``slot_dir`` is deliberately repo-relative (resolved by the
    factory under the configured experience root ``ExperienceConfig.path``) so NO
    absolute host path is hardcoded. The distilled student is PERSISTED to a
    SHA-256 slot, never hot-swapped into the live policy — deployment stays a
    soak-gated operator decision.
    """

    enabled: bool = Field(
        False,
        description="Master switch for the growth-pillar distillation loop (default-off)",
    )
    trigger_min_new_records: int = Field(
        500,
        gt=0,
        description=(
            "Minimum fresh experience records that must accumulate before a "
            "distillation cycle is triggered"
        ),
    )
    check_interval_s: float = Field(
        300.0,
        gt=0,
        description=(
            "Slow-cadence period (seconds) between distillation-trigger checks. The "
            "distillation runs on its own background task OUTSIDE the 30 Hz hot "
            "loop; this is how often the coordinator probes the new-record count "
            "against ``trigger_min_new_records``. Defaults to 5 min so a default-on "
            "deployment never busy-polls."
        ),
    )
    distill_steps: int = Field(
        50,
        gt=0,
        description="Number of bounded distillation gradient steps per cycle",
    )
    batch_size: int = Field(
        32,
        gt=0,
        description="Latent-minibatch size sampled per distillation step",
    )
    temperature: float = Field(
        2.0,
        gt=0,
        description=(
            "Softmax temperature (classification-objective distillers only; ignored "
            "by the regression objective used for the continuous VLA action policy)"
        ),
    )
    alpha: float = Field(
        1.0,
        ge=0,
        le=1,
        description=(
            "Weight of the soft (teacher-matching) loss vs the hard-target loss. "
            "Default 1.0 = pure teacher-matching self-distillation (no ground-truth "
            "action labels), which is how the VLA policy is distilled."
        ),
    )
    learning_rate: float = Field(
        1e-4,
        gt=0,
        description="Learning rate for the bounded distillation gradient steps",
    )
    student_hidden_dim: int = Field(
        64,
        gt=0,
        description=(
            "Hidden-layer width of the compact student MLP. Keep it small — "
            "compression is the point of the growth pillar."
        ),
    )
    slot_dir: str = Field(
        "growth_slot",
        description=(
            "Experience-root-relative leaf for the distilled-student weight slot. "
            "NOT an absolute host path: the factory resolves it UNDER the configured "
            "experience root (``<ExperienceConfig.path>/<slot_dir>``) so any operator "
            "override of the experience path is inherited for free. Distilled "
            "students land here, never overwriting the cloud-pulled slot."
        ),
    )

    @field_validator("slot_dir")
    @classmethod
    def _validate_slot_dir(cls, v: str) -> str:
        """Reject slot_dir values that escape the experience root.

        Mirrors ``OnDeviceLearningConfig._validate_slot_dir``: ``slot_dir`` is
        resolved as ``<ExperienceConfig.path>/<slot_dir>``, so an absolute path, a
        parent-traversal (``..``) component, or an empty / whitespace-only value
        would break that containment contract. Validated at YAML load so a
        misconfigured deployment fails fast.
        """
        from pathlib import PurePosixPath, PureWindowsPath

        slot = v.strip()
        posix = PurePosixPath(slot)
        windows = PureWindowsPath(slot)
        is_absolute = posix.is_absolute() or windows.is_absolute()
        has_traversal = ".." in posix.parts or ".." in windows.parts
        if not slot or is_absolute or has_traversal:
            msg = (
                "growth.slot_dir must be a non-empty relative path without parent "
                "traversal (resolved under ExperienceConfig.path); got " + repr(v)
            )
            raise ValueError(msg)
        return slot


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
    n_batch: int = Field(512, gt=0, description="Prompt batch size for llama-cpp context")
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
    query_system_prompt: str = Field(
        "You are Rocky, a friendly Star Wars MSE-6 Mouse Droid assistant. "
        "Answer the operator's question concisely in one or two short sentences. "
        "You are a small rover; you cannot perform actions from this channel — "
        "this is question-and-answer only, not a command channel.",
        description=(
            "System prompt for the conversational ``answer_query`` path "
            "(free-text Q&A), kept separate from ``system_prompt`` so the "
            "navigation translator keeps emitting JSON while the query path "
            "returns prose. Used by every backend's ``answer_query``."
        ),
    )
    query_max_tokens: int = Field(
        256,
        gt=0,
        description=(
            "Max generation tokens for the ``answer_query`` conversational "
            "path. Separate from ``max_tokens`` (which sizes the terse JSON "
            "GoalVector response) so operators can allow longer prose answers "
            "without enlarging every navigation translation."
        ),
    )
    injection_patterns: list[str] = Field(
        default_factory=lambda: [
            r"ignore (previous|above|all) instructions?",
            r"system prompt",
            r"you are now",
        ],
        description="Regex patterns to detect prompt injection attempts",
    )

    # Tier C2.3 — OpenAI-compatible HTTP backend knobs.
    backend: Literal["llama_cpp", "openai_compatible", "anthropic"] = Field(
        "llama_cpp",
        description=(
            "LLM backend dispatch. Default ``llama_cpp`` preserves pre-Tier-"
            "C2.3 behaviour — ``build_llm_gateway`` instantiates the existing "
            "in-process GGUF loader. ``openai_compatible`` instantiates the "
            "Tier C2.3 ``OpenAICompatibleLLMGateway`` which talks HTTP to "
            "``{base_url}/v1/chat/completions`` (Ollama 0.1.18+ exposes this "
            "endpoint; LM Studio and OpenAI also conform). ``anthropic`` "
            "instantiates the ``AnthropicLLMGateway`` (Claude Messages API) "
            "for cloud deliberative mission translation — it reuses "
            "``model_name`` (a Claude model id, e.g. "
            "``claude-haiku-4-5``), ``api_key`` (or the ``ANTHROPIC_API_KEY`` "
            "env var when unset), ``system_prompt``, ``temperature``, "
            "``max_tokens`` and ``request_timeout_s``. The ``anthropic`` SDK "
            "is an OPTIONAL dependency — install with "
            '``pip install -e ".[anthropic]"``.'
        ),
    )
    base_url: str = Field(
        "http://127.0.0.1:11434",
        description=(
            "Base URL for the ``openai_compatible`` backend. Default targets "
            "the canonical local Ollama port. Env override: "
            "``MOUSEDROID_LLM__BASE_URL``. Examples: "
            "``http://localhost:1234`` (LM Studio), "
            "``https://api.openai.com`` (OpenAI cloud)."
        ),
    )
    model_name: str = Field(
        "gemma-4-e4b",
        description=(
            "Model identifier passed in the ``model`` field of "
            "``/v1/chat/completions``. Default matches the operator's local "
            "Ollama tag. Env override: ``MOUSEDROID_LLM__MODEL_NAME``."
        ),
    )
    api_key: SecretStr | None = Field(
        None,
        description=(
            "Optional bearer token forwarded as ``Authorization: Bearer "
            "<key>``. ``None`` (default) is correct for anonymous local "
            "Ollama. Env override: ``MOUSEDROID_LLM__API_KEY``. Stored as "
            "``SecretStr`` so it never appears in repr / structlog output."
        ),
    )
    request_timeout_s: float = Field(
        10.0,
        gt=0.0,
        description=(
            "Wall-clock timeout for a single ``/v1/chat/completions`` POST "
            "(``openai_compatible``) or ``messages.create`` call "
            "(``anthropic``). Default 10s covers the ``latency_target_ms`` "
            "(500ms) with 20x headroom for Jetson-on-battery deployments. "
            "Smaller than the orchestrator's tick budget so a slow LLM never "
            "starves the control loop. Cloud Claude round-trips are seconds — "
            "raise this (e.g. 15-30s) when ``backend='anthropic'``."
        ),
    )

    # Tier C-rover — cloud-primary / local-secondary failover knobs.
    fallback_backend: Literal["none", "llama_cpp", "openai_compatible"] = Field(
        "none",
        description=(
            "Optional LOCAL backend used when the primary ``backend`` is "
            "unavailable or degraded (e.g. the Jetson is off-network and "
            "``backend='anthropic'`` cannot reach the Claude API). Default "
            "``none`` disables failover so existing single-backend "
            "deployments are byte-identical. When set, "
            "``build_llm_gateway`` wraps the primary + this secondary in a "
            "``FallbackLLMGateway`` composite. Restricted to local backends "
            "(``llama_cpp`` GGUF, or ``openai_compatible`` pointed at a local "
            "Ollama / LM Studio) so the rover stays autonomous without "
            "connectivity. Set equal to ``backend`` is a no-op (the composite "
            "is skipped)."
        ),
    )
    fallback_model_name: str | None = Field(
        None,
        description=(
            "Optional ``model_name`` override applied ONLY to the "
            "``fallback_backend`` gateway. ``None`` (default) reuses "
            "``model_name``. Needed when the primary and secondary backends "
            "want different model identifiers — e.g. primary "
            "``backend='anthropic'`` with ``model_name='claude-haiku-4-5'`` "
            "and ``fallback_backend='openai_compatible'`` needing a local "
            "Ollama tag here. The canonical ``anthropic`` -> ``llama_cpp`` "
            "pairing needs no override (llama_cpp loads ``model_path``, not "
            "``model_name``)."
        ),
    )
    fallback_retry_cooldown_s: float = Field(
        30.0,
        gt=0.0,
        description=(
            "Seconds the ``FallbackLLMGateway`` composite waits before "
            "re-probing a degraded primary backend. A mobile rover sees "
            "transient WAN dropouts, so once the cloud primary degrades the "
            "composite periodically re-attempts it (rather than pinning to "
            "the local secondary until the next process restart). A "
            "successful re-probe clears the primary's degraded state and "
            "resumes cloud serving. Only consulted when "
            "``fallback_backend != 'none'``."
        ),
    )


class VLAConfig(BaseModel):
    """Vision-Language-Action policy configuration (Phase 3a).

    Default ``backend = "none"`` keeps the VLA branch fully disabled so
    pre-Phase-3a behavior is preserved byte-identical. Selecting
    ``"mock"`` activates the deterministic ``MockVLA`` reference; the
    Phase 3b ``"distilled_onnx"`` backend will reuse this same config
    block.
    """

    # ``model_filename`` / ``model_repo_id`` etc. clash with pydantic's
    # default protected ``model_`` namespace; opt out so the warnings do
    # not fire under tests / CI.
    model_config = {"protected_namespaces": ()}

    backend: VLABackendLiteral = Field(
        "none",
        description=(
            "VLA backend. 'none' (default) leaves the VLA branch unwired. "
            "'mock' selects the in-tree zero-dependency MockVLA. "
            "'distilled_onnx' is reserved for Phase 3b. "
            "See ``VLABackendLiteral`` in schema.py for the canonical type."
        ),
    )
    canned_action: list[float] | None = Field(
        None,
        description=(
            "Optional fixed action vector for MockVLA. Length must equal "
            "model.action_dim. None => zero action."
        ),
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence value emitted by MockVLA on every predict() call.",
    )
    fallback_on_timeout: bool = Field(
        True,
        description=(
            "When True, a VLA inference timeout transparently falls back "
            "to the nav_agent. When False, the orchestrator emits a "
            "'vla_timeout_safe_stop' event and returns a zero action so "
            "the safety monitor can escalate on the next tick."
        ),
    )
    # ----- Phase 3b: distilled ONNX backend -----
    model_repo_id: str | None = Field(
        None,
        description=(
            "HuggingFace repo id (e.g., 'lerobot/smolvla') used by "
            "weights_manager.download_weights_from_huggingface to fetch "
            "the ONNX file when ``backend='distilled_onnx'``. None => "
            "expect ``model_path`` to already exist locally under "
            "``cache_dir``."
        ),
    )
    model_filename: str = Field(
        "model.onnx",
        description="Filename of the ONNX graph inside the HF repo / cache dir.",
    )
    cache_dir: str | None = Field(
        "weights/vla",
        description=(
            "Local directory containing the ONNX model. Defaults to "
            "'weights/vla'; override via YAML to relocate the cache."
        ),
    )
    providers: list[str] | None = Field(
        None,
        description=(
            "Optional explicit ORT execution-provider chain. None => the "
            "default fallback chain "
            "['TensorrtExecutionProvider', 'CUDAExecutionProvider', "
            "'CPUExecutionProvider']. Unavailable providers are skipped "
            "automatically by ``DistilledVLAOnnx.warmup``."
        ),
    )
    warmup_iterations: int = Field(
        1,
        ge=0,
        description=(
            "Number of dummy inference passes after session creation to "
            "prime CUDA/TensorRT kernels. 0 disables warmup."
        ),
    )
    h_input_name: str = Field(
        "h",
        description="ONNX input name for the deterministic latent ``h``.",
    )
    z_input_name: str = Field(
        "z",
        description="ONNX input name for the stochastic latent ``z``.",
    )
    action_output_name: str = Field(
        "action",
        description="ONNX output name for the action tensor.",
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


class MissionReplannerConfig(BaseModel):
    """Tier C2.3 — LLM-backed mission replanner adapter configuration.

    Tunables for ``LLMGatewayMissionReplanner`` (built by
    :func:`mousedroid.factory.build_mission_replanner` when
    ``mission.llm_replanner_enabled`` is ``True``). Distinct from
    :class:`LLMReplannerConfig` defined later in this module — that one
    configures the robot-arm symbolic planner. The two share a naming
    prefix but are unrelated subsystems.
    """

    max_prompt_chars: int = Field(
        512,
        gt=0,
        description=(
            "Maximum characters in the augmented goal_text prompt forwarded "
            "to the LLM gateway. The adapter clips longer prompts at this "
            "boundary so a runaway goal_text cannot exceed the gateway's "
            "context window. Default 512 mirrors the rule-based parser's "
            "command-length policy."
        ),
    )
    include_progress_in_prompt: bool = Field(
        True,
        description=(
            "When True (default), the adapter appends "
            "``(last_progress=<float>)`` to the prompt so the LLM sees the "
            "stall context. Operators can disable when their LLM is tuned "
            "for raw goals only."
        ),
    )


class MissionConfig(BaseModel):
    """Mission lifecycle state-machine configuration (Tier C2 / C2.2 / C2.3).

    Drives the ``MissionLifecycle`` state machine that wraps
    :class:`InMemoryTaskTracker` and adds VLM-driven goal-progress feedback
    plus LLM-driven adaptive replan. When ``replan_enabled=False`` (the
    default), the lifecycle never trips into ``REPLANNING`` and never calls
    the LLM gateway — existing deployments produce byte-identical pre-PR
    behaviour because the orchestrator does not build a lifecycle at all
    when this block is at defaults.

    Tier C2.3 adds four fields (``vlm_progress_enabled``,
    ``vlm_mock_progress_value``, ``llm_replanner_enabled``, ``replanner``)
    that gate the VLM progress head + LLM replanner wiring inside
    :func:`build_orchestrator`. All four default to safe values so
    existing YAML loads unchanged.
    """

    replan_enabled: bool = Field(
        False,
        description=(
            "Enable adaptive LLM-driven replan when VLM progress stalls. "
            "Default ``False`` preserves byte-identical pre-PR behaviour."
        ),
    )
    success_threshold: float = Field(
        0.90,
        ge=0.0,
        le=1.0,
        description=(
            "VLM progress score must cross this value to transition the mission to ``SUCCEEDED``."
        ),
    )
    stall_threshold: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description=(
            "VLM progress score below this value counts as a stalled tick. "
            "``stall_window_ticks`` consecutive stalls trip replan."
        ),
    )
    stall_window_ticks: int = Field(
        30,
        gt=0,
        description=(
            "Number of consecutive low-progress ticks before the lifecycle "
            "transitions to ``REPLANNING``. At 30 Hz this is ~1 second."
        ),
    )
    max_replans_per_mission: int = Field(
        3,
        ge=0,
        description=(
            "Hard cap on replans per mission. Once exceeded the lifecycle "
            "transitions to ``FAILED`` with reason='replan_limit_exceeded'."
        ),
    )
    vlm_progress_enabled: bool = Field(
        False,
        description=(
            "Tier C2.3: build a ``VLMProgressHead`` for the mission "
            "lifecycle. Default False preserves pre-Tier-C2.3 byte-identical "
            "behaviour (factory short-circuits to None). When True the head "
            "uses ``MockVLMProgress(mock_progress_value)`` by default; a "
            "real VLM backend is a separate sprint."
        ),
    )
    vlm_mock_progress_value: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Constant value the default ``MockVLMProgress`` backend returns. "
            "Default 0.95 sits above the default ``success_threshold=0.90`` "
            "so a smoke-mode mission transitions to SUCCEEDED on the first "
            "scored tick — useful for the boot-time smoke test."
        ),
    )
    llm_replanner_enabled: bool = Field(
        False,
        description=(
            "Tier C2.3: build an ``LLMGatewayMissionReplanner`` for the "
            "mission lifecycle. Default False preserves pre-Tier-C2.3 "
            "behaviour. Requires the LLM gateway to be enabled — when "
            "``cfg.llm.enabled is False`` the factory still short-circuits "
            "to None even with this flag True (with a structured warning)."
        ),
    )
    replanner: MissionReplannerConfig = Field(
        default_factory=MissionReplannerConfig,
        description="Sub-block tuning the LLM replanner adapter.",
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
    real_supervised_weight: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Phase 2: weight on the auxiliary BC-style supervised loss applied to real "
            "replay batches drawn from the LMDB store. 0.0 disables the BC term — "
            "training is then byte-identical to the pre-Phase-2 path. See also "
            "``bc_lr`` and ``bc_batch_size`` for optional dedicated BC optimizer tuning, "
            "and ``use_replay_mixer`` to source batches from the sim/real mixer."
        ),
    )
    bc_lr: float | None = Field(
        None,
        gt=0,
        description=(
            "Phase 2.1: optional dedicated learning rate for the BC auxiliary loss. "
            "When ``None`` the policy optimizer (and its learning_rate) is reused — "
            "byte-identical to the pre-Phase-2.1 path. When set, a separate "
            "``bc_optimizer`` is built over policy parameters."
        ),
    )
    bc_batch_size: int | None = Field(
        None,
        gt=0,
        description=(
            "Phase 2.1: optional dedicated mini-batch size for the BC step. "
            "When ``None`` the main ``batch_size`` is reused. Reserved for future "
            "use when the BC update is decoupled from the actor-critic batch; "
            "currently consumed by the trainer constructor and exposed in checkpoints."
        ),
    )
    use_replay_mixer: bool = Field(
        False,
        description=(
            "Phase 2.1: when True, draw training batches from the sim/real "
            "``ReplayMixer`` instead of the single ``OfflineRLDataset`` LMDB. "
            "Default False preserves byte-identical behavior to the single-LMDB path."
        ),
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
    replay_seed: int | None = Field(
        None,
        description="Random seed for episodic replay sampling (None = non-deterministic)",
    )


class ExperimentLoggerConfig(BaseModel):
    """Experiment-logger configuration for training runs (per-step + per-phase metrics).

    Wired into :class:`PipelineOrchestrator` and :class:`OfflineRLTrainer`
    via :func:`mousedroid.factory.build_experiment_logger`. Defaults to OFF
    (``backend="none"``) so a YAML predating this feature loads unchanged
    (CLAUDE.md invariant #9). Selecting ``backend="mlflow"`` requires the
    ``mousedroid[mlflow]`` extras (``mlflow-skinny``); a missing dep
    degrades gracefully to the NoOp logger with a structured warning.
    """

    backend: Literal["none", "mlflow"] = Field(
        "none",
        description=(
            "Experiment-logger backend. ``none`` (default) selects the NoOp "
            "logger — byte-identical to pre-feature behavior. ``mlflow`` "
            "selects the MlflowClient-backed logger writing to "
            "``tracking_uri`` (default ``file:./mlruns``)."
        ),
    )
    tracking_uri: str = Field(
        "file:./mlruns",
        description=(
            "MLflow tracking URI. ``file:./mlruns`` (default) writes to a "
            "local directory relative to the factory's resolution time (the "
            "factory pins this to an absolute path to avoid CWD surprises). "
            "Set to ``http://host:port`` to use a remote tracking server."
        ),
    )
    experiment_name: str = Field(
        "mousedroid",
        min_length=1,
        description="MLflow experiment name (created if missing).",
    )
    run_name: str | None = Field(
        None,
        description=(
            "Optional human-readable run name for the parent (pipeline) run. "
            "When ``None`` the logger falls back to its configured default "
            '(this field) or the ``"pipeline"`` sentinel.'
        ),
    )
    log_step_every_n: int = Field(
        1,
        gt=0,
        description=(
            "Per-update-step metric throttle. ``1`` (default) logs every "
            "update_step call. Set higher for very-long training runs to "
            "reduce store-write overhead."
        ),
    )
    log_artifacts: bool = Field(
        True,
        description=(
            "When True, the orchestrator logs the resolved Settings JSON "
            "snapshot as a parent-run artifact at start, plus the per-phase "
            "checkpoint file as a child-run artifact on phase completion."
        ),
    )


class ObservabilityConfig(BaseModel):
    """Top-level observability configuration for the training stack.

    Currently contains the experiment-logger sub-config; future fields
    (training-side Prometheus metrics, W&B integration, etc.) land here
    to keep ``Settings`` flat.
    """

    experiment_logger: ExperimentLoggerConfig = Field(
        default_factory=ExperimentLoggerConfig,
        description="Per-run experiment-logger config (MLflow file backend).",
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
    track_llm_gateway: bool = Field(
        True,
        description=(
            "Expose deliberative LLM-gateway observability for the Anthropic "
            "Claude tier: token-usage counter (labels: model, token_type), "
            "round-trip latency histogram, per-tier served counter (labels: "
            "tier, outcome), and a latency-budget-exceeded counter (label: "
            "model). Emitted only when the gateway runs with a MetricsRegistry "
            "and actually translates — safe to leave on."
        ),
    )
    track_curiosity: bool = Field(True, description="Expose curiosity intrinsic reward gauge")
    track_sensor_recovery: bool = Field(True, description="Expose sensor recovery counter")
    track_cloud: bool = Field(
        True,
        description=(
            "Expose cloud digital twin metrics: publish counters, publish "
            "latency histogram, circuit breaker state, and experience "
            "export backlog gauges. Emitted only when a cloud sink is "
            "actually wired into the orchestrator — safe to leave on."
        ),
    )
    track_mcp: bool = Field(
        True,
        description=(
            "Expose MCP server metrics: request counter, per-tool call "
            "counter (label: tool, result), and request latency histogram. "
            "Emitted only when the MCP server is actually built — safe to "
            "leave on."
        ),
    )
    track_on_device_learning: bool = Field(
        True,
        description=(
            "Expose the Phase-6 on-device-learning revert counter "
            "(label: reason). Pure-add: omitted from /metrics until the first "
            "revert, so default deployments render byte-identically. Safe to "
            "leave on."
        ),
    )
    track_growth_distillation: bool = Field(
        True,
        description=(
            "Expose the growth-pillar distillation counter (label: outcome). "
            "Pure-add: omitted from /metrics until the first distillation cycle, "
            "so default deployments render byte-identically. Safe to leave on."
        ),
    )
    track_voice_degradation: bool = Field(
        True,
        description=(
            "Expose the voice-subsystem degradation counters: "
            "``voice_speaker_degraded_total`` (label: subsystem — the USB "
            "speaker exhausted its reconnect retries or the engine fell back "
            "to a MockSpeaker) and ``voice_tts_synthesize_failures_total`` "
            "(label: api — a Piper synthesis call raised). Pure-add: each "
            "family is omitted from /metrics until its first increment, so "
            "default deployments render byte-identically. Safe to leave on."
        ),
    )
    loop_latency_buckets_ms: tuple[float, ...] = Field(
        (1.0, 2.5, 5.0, 10.0, 20.0, 33.0, 50.0, 100.0, 200.0, float("inf")),
        description="Histogram bucket boundaries for control-loop latency (ms)",
    )
    llm_latency_buckets_ms: tuple[float, ...] = Field(
        (25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, float("inf")),
        description="Histogram bucket boundaries for LLM translation latency (ms)",
    )
    llm_gateway_latency_buckets_ms: tuple[float, ...] = Field(
        (50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, float("inf")),
        description=(
            "Histogram bucket boundaries for cloud LLM-gateway round-trip "
            "latency (ms). Wider than llm_latency_buckets_ms because cloud "
            "Claude round-trips are seconds, not ms. The 500 ms default "
            "latency_target_ms and the 5000 ms cloud-pilot overlay value "
            "(config/jetson_claude_pilot.yaml) both land on bucket boundaries."
        ),
    )
    mcp_latency_buckets_ms: tuple[float, ...] = Field(
        (5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0, float("inf")),
        description="Histogram bucket boundaries for MCP request latency (ms)",
    )
    vla_inference_seconds_buckets: tuple[float, ...] = Field(
        (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, float("inf")),
        description=(
            "Histogram bucket boundaries for VLA policy inference latency (seconds). "
            "Phase 3b: covers the 30 Hz orchestrator budget (~33 ms) up to long-tail "
            "fallbacks beyond 1 s. Operator-tunable per deployment."
        ),
    )
    world_model_observe_step_seconds_buckets: tuple[float, ...] = Field(
        (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, float("inf")),
        description=(
            "Histogram bucket boundaries for DualStreamRSSM.observe_step latency "
            "(seconds). Default envelope covers <1 ms baseline up to long-tail "
            "PyTorch ticks beyond 100 ms. The 10 ms target on Orin Nano (with "
            "cfg.world_model.engine=onnx_trt + TensorRT EP) lands within the "
            "(0.005, 0.01] bucket; the portable dev gate is 33 ms (30 Hz tick). "
            "Operator-tunable per deployment."
        ),
    )
    cloud_weight_update_download_seconds_buckets: tuple[float, ...] = Field(
        (0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, float("inf")),
        description=(
            "Histogram bucket boundaries for OTA weight-update download latency "
            "(seconds). Tier C1: covers cellular-fleet downloads on the order of "
            "tens of MB. Operator-tunable per deployment."
        ),
    )
    mission_duration_seconds_buckets: tuple[float, ...] = Field(
        (1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, float("inf")),
        description=(
            "Histogram bucket boundaries for mission active duration (seconds). "
            "Tier C2 (C2.3): covers short single-objective missions (< 1 min) "
            "through multi-minute autonomous navigation runs (> 10 min)."
        ),
    )

    @field_validator(
        "loop_latency_buckets_ms",
        "llm_latency_buckets_ms",
        "llm_gateway_latency_buckets_ms",
        "mcp_latency_buckets_ms",
        "vla_inference_seconds_buckets",
        "world_model_observe_step_seconds_buckets",
        "cloud_weight_update_download_seconds_buckets",
        "mission_duration_seconds_buckets",
    )
    @classmethod
    def _validate_histogram_buckets(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Enforce monotonically ascending, strictly positive bucket boundaries.

        A trailing ``float("inf")`` sentinel is permitted (and conventional for
        Prometheus histograms) but not required — the registry appends one at
        runtime if missing. When present, ``float("inf")`` MUST be the last
        element; an ``inf`` in any other position would yield surprising bucket
        cardinality after the runtime ``sorted(...)`` call in ``MetricsRegistry``.
        Negative, zero, or duplicate boundaries would silently corrupt bucket
        accumulation, so they're rejected at schema-load time.
        """
        inf = float("inf")
        if not value:
            msg = "histogram bucket tuple must be non-empty"
            raise ValueError(msg)
        # Reject ``inf`` anywhere except the trailing position.
        inf_positions = [i for i, b in enumerate(value) if b == inf]
        if inf_positions and inf_positions != [len(value) - 1]:
            msg = (
                f"histogram bucket boundaries may only contain +inf as the "
                f"trailing sentinel; got {value!r}"
            )
            raise ValueError(msg)
        finite = [b for b in value if b != inf]
        if finite != sorted(finite):
            msg = f"histogram bucket boundaries must be monotonically ascending; got {value!r}"
            raise ValueError(msg)
        if any(b <= 0.0 for b in finite):
            msg = f"histogram bucket boundaries must be strictly positive; got {value!r}"
            raise ValueError(msg)
        if len(set(finite)) != len(finite):
            msg = f"histogram bucket boundaries must be unique (no duplicates); got {value!r}"
            raise ValueError(msg)
        return value


class ModelConfig(BaseModel):
    """Neural network model dimensions."""

    vision_dim: int = Field(256, ge=0, description="Vision feature input dim (0=disabled)")
    ultrasonic_dim: int = Field(1, ge=0, description="Ultrasonic input dim (0=disabled)")
    motor_state_dim: int = Field(4, gt=0, description="Motor state dim [vx, vy, omega, battery]")
    hidden_dim: int = Field(256, gt=0, description="RNN hidden dim")
    latent_dim: int = Field(64, gt=0, description="Latent state dim")
    action_dim: int = Field(3, gt=0, description="Action dim [vx, vy, omega]")
    obs_dim: int = Field(256, gt=0, description="Fused observation embedding dim")
    vision_proj_dim: int = Field(128, ge=0, description="Vision projection dim (0=disabled)")
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

    # Latent state health monitoring
    latent_norm_threshold: float = Field(
        50.0,
        gt=0.0,
        description="h-state L2 norm above this value triggers a latent_saturated warning",
    )
    latent_recovery_buffer_size: int = Field(
        5,
        gt=0,
        description="Number of recent valid (h, z) pairs kept for NaN recovery",
    )

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

    # RSSM dynamics-pretraining KL knobs (Phase 5). Read by RSSM.train_sequence;
    # build_rssm_trainable copies operator overrides from TrainingConfig onto these.
    kl_beta: float = Field(
        1.0, ge=0.0, description="KL weight in the RSSM training ELBO (recon + kl_beta*KL)."
    )
    kl_balance_alpha: float = Field(
        0.8, ge=0.0, le=1.0, description="Dreamer KL-balancing weight (prior-update term)."
    )
    kl_free_nats: float = Field(
        1.0, ge=0.0, description="Free-bits floor (nats) for the RSSM training KL."
    )
    logvar_clamp: float = Field(
        10.0,
        gt=0.0,
        description="Symmetric |logvar| clamp before exp() in the balanced-KL "
        "(fp16 AMP overflow guard). Tunable here rather than hardcoded.",
    )

    @model_validator(mode="after")
    def _validate_optional_modalities(self) -> Self:
        """Validate optional modality dimension pairs."""
        if (self.vision_dim == 0) != (self.vision_proj_dim == 0):
            msg = (
                "vision_dim and vision_proj_dim must both be zero (disabled) "
                "or both non-zero (enabled) together"
            )
            raise ValueError(msg)

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


class WorldModelConfig(BaseModel):
    """World-model runtime engine selector (Tier B2).

    Drives :func:`mousedroid.factory.build_world_model` dispatch. The
    default (``engine="torch"``) preserves byte-identical behavior with
    pre-B2 deployments: existing ``config/*.yaml`` files without a
    ``world_model:`` block load unchanged and continue to use the
    PyTorch :class:`~mousedroid.world_model.dual_stream_rssm.DualStreamRSSM`.

    Flip ``engine="onnx_trt"`` in ``config/jetson_production.yaml`` to
    serve ``observe_step`` from an exported ``.onnx`` via
    ``onnxruntime`` + TensorRT execution provider. ``imagine_step`` (MCTS
    rollouts) always runs on the PyTorch model regardless of engine; the
    factory wires both paths.
    """

    engine: Literal["torch", "onnx_trt"] = Field(
        "torch",
        description=(
            "World-model inference engine. 'torch' runs DualStreamRSSM "
            "as a PyTorch module (default, byte-identical pre-PR). "
            "'onnx_trt' loads an exported .onnx via onnxruntime with "
            "the TensorrtExecutionProvider -> CUDAExecutionProvider -> "
            "CPUExecutionProvider fallback chain."
        ),
    )
    onnx_path: str | None = Field(
        None,
        description=(
            "Filesystem path to the exported .onnx for engine='onnx_trt'. "
            "When None and engine='onnx_trt', the factory falls back to "
            "downloading from onnx_repo_id via HuggingFace Hub."
        ),
    )
    onnx_repo_id: str = Field(
        "ianshank/mousedroid-dual-stream-rssm",
        description=(
            "HuggingFace Hub repo holding the .onnx artifact for the "
            "ONNX runtime path. Mirrors the [vla] backend convention."
        ),
    )
    onnx_filename: str = Field(
        "observe_step.onnx",
        description="Filename inside the HF repo / cache directory.",
    )
    onnx_cache_dir: str = Field(
        "weights/dual_stream_rssm",
        description=(
            "Filesystem directory where the factory caches the HF-downloaded "
            ".onnx artifact. Mirrors VLAConfig.cache_dir. Operators can point "
            "this at /opt/mousedroid/weights/... on Jetson deployments so the "
            "download persists across container restarts."
        ),
    )
    onnx_warmup_iterations: int = Field(
        1,
        ge=0,
        description="Dummy inferences run during ONNX session warmup.",
    )


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


class VLMProgressConfig(BaseModel):
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


class RewardConfig(BaseModel):
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


# ---------------------------------------------------------------------------
# 4WD rover sim-to-real configuration (Phase A scaffold)
# ---------------------------------------------------------------------------


class RoverInertialConfig(BaseModel):
    """Mass-property overrides for the MSE-6 shell + 4WD chassis URDF.

    Used by the Isaac Lab env stub (and future MuJoCo backend) to update
    the URDF's documentation-quality defaults with values derived from
    the actual 3D-print parameters of the physical droid. A top-heavy
    ``com_offset_xyz_m`` is intentional — the policy must experience the
    roll tendency in sim to generalise to hardware.
    """

    shell_mass_kg: float = Field(0.85, gt=0, description="MSE-6 shell mass (kg)")
    shell_thickness_m: float = Field(
        0.003, gt=0, description="Shell wall thickness for hollow inertia (m)"
    )
    shell_infill: float = Field(0.20, gt=0, le=1.0, description="Print infill fraction [0, 1]")
    com_offset_xyz_m: tuple[float, float, float] = Field(
        (0.0, 0.0, 0.04),
        description="COM offset from base_link origin (top-heavy: positive z)",
    )
    wheel_mass_kg: float = Field(0.06, gt=0, description="Per-wheel mass (kg)")


class MujocoSimConfig(BaseModel):
    """MuJoCo backend parameters (consumed only when ``rover.sim.backend == 'mujoco'``).

    Every physics knob is config-driven (invariant #3). ``wheel_slip_default``
    is a documented OBSERVATION-NOISE proxy — MuJoCo has no first-class slip
    parameter — applied as multiplicative noise on wheel_vel/pose, NOT a
    contact-solver field.
    """

    mjcf_path: str = Field(
        "assets/rover/mse6_4wd.xml",
        min_length=1,
        description="Repo-relative path to the skid-steer MJCF (resolved against repo root).",
    )
    arena_half_extent_m: float = Field(
        2.0, gt=0.0, description="Half-size of the walled arena (walls give the lidar a signal)."
    )
    lidar_num_sectors: int = Field(
        16, gt=0, description="Number of rangefinder sectors fanned around yaw."
    )
    lidar_max_range_m: float = Field(
        4.0, gt=0.0, description="Rangefinder clip; readings normalised to [0,1] by this."
    )
    lidar_ring_radius_m: float = Field(
        0.11, gt=0.0, description="Radius of the lidar-site ring on the chassis (matches MJCF)."
    )
    lidar_mount_z_m: float = Field(
        0.03, description="Z offset of the lidar sites above the chassis origin (matches MJCF)."
    )
    noise_rng_seed: int = Field(
        0, ge=0, description="Default seed for the slip observation-noise RNG (reset overrides)."
    )
    battery_voltage_const_v: float = Field(
        12.0, gt=0.0, description="Constant battery voltage stamped into motor_state[3]."
    )
    wheel_friction_default: float = Field(
        1.0, gt=0.0, description="Default tangential friction (geom_friction[:,0])."
    )
    wheel_slip_default: float = Field(
        0.0, ge=0.0, description="Observation-noise proxy magnitude (NOT a MuJoCo field)."
    )
    motor_gain_default: float = Field(
        1.0, gt=0.0, description="Default actuator gain (actuator_gainprm[:,0])."
    )
    chassis_mass_default_kg: float = Field(
        2.7, gt=0.0, description="Default chassis mass (body_mass + inertia recompute)."
    )
    render_vision: bool = Field(
        False,
        description="Render an RGB camera for vision-on RSSM fine-tuning (off by default).",
    )
    render_width: int = Field(64, gt=0, description="Offscreen RGB render width (px).")
    render_height: int = Field(64, gt=0, description="Offscreen RGB render height (px).")
    camera_name: str = Field(
        "rover_cam", min_length=1, description="Name of the MJCF camera to render from."
    )


class RoverSimConfig(BaseModel):
    """Simulation backend selection and physics timing for rover training."""

    backend: Literal["isaac_lab", "mujoco", "mock"] = Field(
        "mock",
        description=(
            "Sim backend. 'mock' (NumPy, no physics) is the default so CI "
            "and unit tests run without GPU/Isaac dependencies."
        ),
    )
    urdf_path: str = Field(
        "assets/rover/mse6_4wd.urdf",
        description="Path to the rover URDF (relative to repo root)",
    )
    sim_dt_s: float = Field(
        1.0 / 120.0, gt=0, description="Physics step (s); decimation maps to control rate"
    )
    decimation: int = Field(
        4, ge=1, description="Physics steps per control step (30 Hz control at dt=1/120)"
    )
    episode_length_s: float = Field(
        20.0, gt=0, description="Max episode duration before truncation (s)"
    )
    num_envs: int = Field(1, ge=1, description="Parallel envs (use 4096+ for Isaac Lab training)")
    headless: bool = Field(True, description="Run Isaac Lab without a viewer")
    inertial: RoverInertialConfig = Field(
        default_factory=RoverInertialConfig,
        description="Mass-property overrides for the URDF defaults",
    )
    mujoco: MujocoSimConfig = Field(
        default_factory=MujocoSimConfig,
        description="MuJoCo backend parameters (used only when backend == 'mujoco').",
    )


# Action vector dimensionality per supported mode. Centralised so env
# classes don't carry a magic ``2`` — adding e.g. a 4-wheel mecanum mode
# in Phase B is a single-line dict update plus the new mode literal.
_ROVER_ACTION_DIM_BY_MODE: dict[str, int] = {
    "differential": 2,
    "body_velocity": 2,
}


class RoverActionConfig(BaseModel):
    """Action space configuration for the rover policy."""

    mode: Literal["differential", "body_velocity"] = Field(
        "differential",
        description=(
            "'differential' -> [left_wheel_rad_s, right_wheel_rad_s]; "
            "'body_velocity' -> [vx_mps, omega_rads]."
        ),
    )
    max_wheel_rad_s: float = Field(
        25.0, gt=0, description="Hard cap on per-wheel angular velocity (rad/s)"
    )
    slew_rad_s2: float = Field(
        60.0,
        gt=0,
        description=(
            "Max wheel angular acceleration (rad/s^2). Consumed by the "
            "Phase B neurosymbolic action validator; recorded here so the "
            "URDF, env, and safety layer share one source of truth."
        ),
    )

    @property
    def action_dim(self) -> int:
        """Return the action-vector dimensionality implied by ``mode``."""
        return _ROVER_ACTION_DIM_BY_MODE[self.mode]


class RoverObservationConfig(BaseModel):
    """Observation-space toggles for the rover env."""

    include_imu: bool = Field(True, description="6-D linear-accel + ang-vel vector")
    include_wheel_encoders: bool = Field(True, description="4-D wheel angular velocities")
    include_chassis_pose: bool = Field(
        True, description="4-D [x, y, cos(theta), sin(theta)] body pose"
    )
    include_lidar_sectors: bool = Field(True, description="Sector-binned LiDAR clearance features")
    lidar_num_sectors: int = Field(
        16, ge=1, description="Number of angular sectors for LiDAR features"
    )

    def enabled_keys(self) -> tuple[str, ...]:
        """Return the obs-dict keys implied by the enabled modality toggles.

        Single source of truth for the observation contract — the mock and
        Isaac Lab env classes plus the factory log call this so the keys
        and their order can never drift between backends.
        """
        keys: list[str] = []
        if self.include_imu:
            keys.append("imu")
        if self.include_chassis_pose:
            keys.append("chassis_pose")
        if self.include_wheel_encoders:
            keys.append("wheel_vel")
        if self.include_lidar_sectors:
            keys.append("lidar")
        return tuple(keys)


class RoverTaskConfig(BaseModel):
    """Placeholder goal-reach task parameters for the Phase A mock env.

    The mock env's reward is a placeholder ``-||pose - goal_xy_m||`` that
    terminates inside ``goal_reach_radius_m``. The full reward shaper
    (Phase C, ``mousedroid.training.rover_reward``) will replace both
    fields with a structured multi-objective signal; until then these
    knobs let callers steer the placeholder without editing code.
    """

    goal_xy_m: tuple[float, float] = Field(
        (2.0, 0.0),
        description="World-frame goal pose (x, y) in metres",
    )
    goal_reach_radius_m: float = Field(
        0.10,
        gt=0,
        description="Distance to goal at which the episode terminates (m)",
    )


class RoverRewardConfig(BaseModel):
    """Reward weights for the Isaac Lab rover env (Tier C4 — Phase B baseline).

    Implements the design documented in ADR-009 (Isaac Lab Phase B). The
    Isaac Lab ``step()`` body composes the per-step reward as::

        reward = (
            forward_velocity_weight * forward_velocity_mps
            - collision_weight * is_colliding
        )

    Both weights are operator-tunable; no hardcoded reward weights live
    inside :mod:`mousedroid.sim.isaaclab`. Backwards-compatible default
    on :class:`RoverConfig` is ``reward=None``; the Isaac Lab env raises
    a clear :class:`ValueError` when built without an explicit reward
    block so operators set it intentionally per ADR-009.
    """

    forward_velocity_weight: float = Field(
        0.01,
        ge=0,
        description=(
            "Reward per m/s forward (linear body-frame velocity). Must "
            "be ``>= 0``; negative values would invert the safety sign "
            "(rewarding reverse motion) and contradict ADR-009."
        ),
    )
    collision_weight: float = Field(
        0.1,
        ge=0,
        description=(
            "Penalty per collision frame (subtracted from reward). Must "
            "be ``>= 0``; negative values would reward crashes and "
            "violate the constitutional safety invariant."
        ),
    )


class RoverConfig(BaseModel):
    """Top-level rover sim-to-real configuration (None preserves legacy).

    Optional on the root :class:`Settings`. When ``None``, existing YAML
    files load unchanged and the orchestrator behaves as before.
    """

    sim: RoverSimConfig = Field(default_factory=RoverSimConfig)
    action: RoverActionConfig = Field(default_factory=RoverActionConfig)
    observation: RoverObservationConfig = Field(default_factory=RoverObservationConfig)
    task: RoverTaskConfig = Field(default_factory=RoverTaskConfig)
    reward: RoverRewardConfig | None = Field(
        None,
        description=(
            "Isaac Lab reward weights (Tier C4). ``None`` preserves "
            "byte-identical pre-PR behaviour; the Isaac Lab env raises "
            "``ValueError`` when built without an explicit block so "
            "operators set the weights intentionally per ADR-009."
        ),
    )


class SafetyProjectorConfig(BaseModel):
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
    projector: SafetyProjectorConfig = Field(
        default_factory=_settings_default_factory(SafetyProjectorConfig),
        description=(
            "Geometric safety action projection block (Tier C2). "
            "Default ``projector.enabled=false`` preserves byte-identical "
            "pre-C2 behaviour — the orchestrator skips the projection seam "
            "entirely when disabled."
        ),
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
        description=(
            "Paths that bypass authentication. Each entry must start with '/' "
            "and contain only lowercase letters, digits, hyphens, underscores, "
            "and forward slashes."
        ),
    )

    @field_validator("exempt_paths")
    @classmethod
    def _validate_exempt_paths(cls, paths: list[str]) -> list[str]:
        """Reject paths with traversal components, query strings, or unusual chars.

        Also reject (a) trailing slashes on non-root entries and (b)
        empty segments (``//``). The middleware uses segment-exact
        matching so ``/health/`` and ``/health`` would be different
        exemptions — silently accepting a trailing slash would make
        operator misconfigurations invisible. Addresses Gemini /
        Copilot review (PR #78).

        Prevents config typos that could widen the exemption surface (e.g.
        '/healthz' unintentionally exempting '/health') from going unnoticed.
        """
        import re

        exempt_re = re.compile(r"^/[a-z0-9_/\-]*$")
        for path in paths:
            if not exempt_re.match(path):
                raise ValueError(
                    f"exempt_paths entry {path!r} is invalid: must start with '/' "
                    "and contain only [a-z0-9_/-] (no query strings, no '..')"
                )
            if len(path) > 1 and path.endswith("/"):
                raise ValueError(
                    f"exempt_paths entry {path!r} must not have a trailing slash "
                    "(non-root). Use '/health' rather than '/health/'."
                )
            if "//" in path:
                raise ValueError(
                    f"exempt_paths entry {path!r} contains an empty segment ('//'). "
                    "Use single-slash boundaries only."
                )
        return paths


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
        "0.0.0.0",  # noqa: S104 — intentional all-interfaces default for the rover WiFi dashboard
        description="Server bind address (0.0.0.0 = all interfaces)",
    )
    port: int = Field(8080, gt=0, le=65535, description="Server port")
    port_discovery_strategy: Literal["fixed", "fallback_range", "kernel_assigned"] = Field(
        "fixed",
        description=(
            "Port binding strategy. 'fixed': bind exactly to port (raises on conflict). "
            "'fallback_range': try port, port+1, ..., port+port_discovery_max_attempts. "
            "'kernel_assigned': bind to port 0 and let the OS assign a free port."
        ),
    )
    port_discovery_max_attempts: int = Field(
        10,
        gt=0,
        le=100,
        description=(
            "Number of consecutive ports to try when port_discovery_strategy='fallback_range'."
        ),
    )
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

    # ------------------------------------------------------------------
    # PR #4: live streaming, mock visibility, serialization negotiation,
    # mDNS readiness, and sensor-liveness fields. All optional, all with
    # safe defaults so existing YAML files load unchanged.
    # ------------------------------------------------------------------
    lidar_raw_publish_hz: float = Field(
        5.0,
        gt=0,
        le=30,
        description=(
            "Target broadcast rate (Hz) for the /ws/v1/lidar/raw WebSocket "
            "stream. The LD19 driver runs at ~10 Hz natively; this rate "
            "downsamples on the server side before fan-out to clients."
        ),
    )
    lidar_raw_queue_size: int = Field(
        16,
        gt=0,
        le=1024,
        description=(
            "Internal queue depth for raw LiDAR scan publishing. When the "
            "queue is full new scans are dropped (non-blocking) to keep the "
            "control loop responsive."
        ),
    )
    lidar_raw_ws_path: str = Field(
        "/ws/v1/lidar/raw",
        description=(
            "WebSocket path for the raw LiDAR scan stream. Versioned so "
            "future protocol breaks land on /ws/v2/* without breaking "
            "existing dashboards."
        ),
    )
    mock_force_real_when_enabled: bool = Field(
        True,
        description=(
            "When True (default) and mock_hardware=True, the factory still "
            "builds the real aiohttp TelemetryServer on localhost instead of "
            "the no-op MockTelemetryServer so the dashboard can be exercised "
            "locally. Existing tests that construct MockTelemetryServer "
            "directly remain unaffected; the legacy force_real_server flag "
            "still wins when set explicitly."
        ),
    )
    mock_telemetry_source_enabled: bool = Field(
        True,
        description=(
            "When True and mock_hardware=True, factory wires a "
            "MockTelemetrySource that synthesises plausible scan + camera "
            "data into the publisher so the dashboard renders meaningful "
            "patterns without a real rover attached."
        ),
    )
    msgpack_client_lib_url: str = Field(
        "https://github.com/msgpack/msgpack-javascript",
        description=(
            "Public URL pointing to a msgpack JS decoder. Surfaced in the "
            "dashboard error banner when the server is configured for "
            "msgpack but the connecting client lacks a decoder."
        ),
    )
    mdns_register_timeout_s: float = Field(
        5.0,
        gt=0,
        le=60,
        description=(
            "Maximum time TelemetryServer.start() waits for the mDNS "
            "register call (in a thread pool) to complete or fail before "
            "continuing startup. Timeout is non-fatal: server keeps "
            "running, mDNS becomes best-effort and the failure is "
            "recorded via FailureRecorder."
        ),
    )
    ws_protocol_version: int = Field(
        1,
        ge=1,
        le=99,
        description=(
            "Server-side WebSocket protocol version advertised in the "
            "handshake hello-ack. Clients should send their accepted "
            "versions in the hello message."
        ),
    )
    ws_handshake_timeout_s: float = Field(
        2.0,
        gt=0,
        le=30,
        description=(
            "Maximum time to wait for the optional client hello negotiation "
            "message before falling back to the server-configured "
            "serialization. Keeps the path backwards-compatible with "
            "non-negotiating clients."
        ),
    )
    sensor_liveness_stale_s: float = Field(
        2.0,
        gt=0,
        le=60,
        description=(
            "Age threshold (seconds) above which a sensor's data is "
            "reported as 'stale' rather than 'live' in the liveness map. "
            "Tune per deployment based on the slowest sensor's expected "
            "update rate."
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
    metrics_labels: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Deployment labels (e.g. env, region, fleet) attached to cloud "
            "exports. Keys/values must be non-empty strings. Backwards "
            "compatible: empty dict by default."
        ),
    )

    @model_validator(mode="after")
    def _validate_required_cloud_fields(self) -> GCPConfig:
        """Enforce non-empty identifiers when the digital twin is enabled.

        ``GCPConfig`` itself is optional on :class:`Settings`; when present
        it must identify the project, robot, and every destination that
        downstream sinks will target so they never silently publish to
        empty topic / bucket names.

        Returns:
            The validated instance (unchanged when valid).

        Raises:
            ValueError: If any required identifier is empty / whitespace.
        """
        required: dict[str, str] = {
            "project_id": self.project_id,
            "robot_id": self.robot_id,
            "pubsub.telemetry_topic": self.pubsub.telemetry_topic,
            "pubsub.experience_topic": self.pubsub.experience_topic,
            "storage.bucket": self.storage.bucket,
        }
        empty = [key for key, value in required.items() if not value or not value.strip()]
        if empty:
            raise ValueError("GCPConfig requires non-empty values for: " + ", ".join(sorted(empty)))

        for label_key, label_value in self.metrics_labels.items():
            if not label_key or not label_key.strip():
                raise ValueError("GCPConfig.metrics_labels keys must be non-empty")
            if not isinstance(label_value, str) or not label_value.strip():
                raise ValueError(
                    f"GCPConfig.metrics_labels[{label_key!r}] must be a non-empty string"
                )

        if self.pubsub.telemetry_topic == self.pubsub.experience_topic:
            raise ValueError("GCPConfig.pubsub.telemetry_topic and experience_topic must differ")
        return self


# ---------------------------------------------------------------------------
# Tier C1 — Closed-loop cloud retraining + OTA weight updates
# ---------------------------------------------------------------------------


#: Default ``world_model_repo_id`` literal. Defined as a module-level constant
#: so the field default and the
#: ``_warn_on_default_world_model_repo`` validator share one canonical value
#: and a future rename touches one place. This is the maintainer's personal
#: HF Hub repo — operators MUST override before enabling the world-model
#: poller in production.
_WORLD_MODEL_DEFAULT_REPO_ID: str = "ianshank/mousedroid-dual-stream-rssm"


class WeightUpdatePollConfig(BaseModel):
    """Configuration for the HuggingFace Hub OTA weight-update poller.

    Default ``poll_interval_s = 0.0`` disables the poller entirely so
    existing YAML files load with byte-identical pre-Tier-C1 behaviour.
    """

    poll_interval_s: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Background poll interval in seconds. ``0.0`` disables the poller "
            "entirely (default — preserves byte-identical pre-Tier-C1 "
            "behaviour). Operators flip this to e.g. 300.0 to poll every "
            "five minutes for new artifacts."
        ),
    )
    policy_repo_id: str = Field(
        "ianshank/mousedroid-policy-v2",
        description="HuggingFace Hub repo ID containing the trained policy artifact.",
    )
    policy_filename: str = Field(
        "policy.onnx",
        description="Filename within ``policy_repo_id`` of the policy artifact.",
    )
    world_model_repo_id: str = Field(
        _WORLD_MODEL_DEFAULT_REPO_ID,
        description="HuggingFace Hub repo ID containing the trained world-model artifact.",
    )
    world_model_filename: str = Field(
        "observe_step.onnx",
        description="Filename within ``world_model_repo_id`` of the observe_step ONNX export.",
    )
    cache_dir: str = Field(
        "weights/cloud_updates",
        description=(
            "Local directory the poller writes verified artifacts into. "
            "Resolved relative to the runtime CWD unless absolute."
        ),
    )
    sha256_manifest_filename: str = Field(
        "sha256.txt",
        description=(
            "Filename inside the HF repo carrying the expected hex-encoded "
            "SHA-256 digest for the downloaded artifact. Single-line file "
            "containing only the hex digest. SAFETY-CRITICAL: a download is "
            "refused if the local SHA does not match this manifest."
        ),
    )
    reset_state_on_swap: bool = Field(
        True,
        description=(
            "Reset h/z to zeros after swap. Default ``True`` because the "
            "orchestrator's ``tick()`` body runs ``_update_world_model`` "
            "BEFORE ``_select_action``, so a swap mid-sprint leaves the next "
            "tick's ``observe_step`` receiving ``(h, z)`` computed by the OLD "
            "world model. Zeroing the recurrent state on swap is the only "
            "way to avoid that one-tick cross-model contamination — trade-off "
            "is one tick of context loss, which is acceptable for an OTA "
            "event operators expect to happen at minute-scale, not 30 Hz."
        ),
    )
    download_timeout_s: float = Field(
        60.0,
        gt=0.0,
        description="Per-download wall-clock timeout (seconds).",
    )
    max_retries: int = Field(
        3,
        ge=0,
        description="Maximum retry attempts per artifact (forwarded to weights_manager).",
    )
    world_model_enabled: bool = Field(
        False,
        description=(
            "Enable a second OTA poller targeting ``world_model_repo_id`` / "
            "``world_model_filename``. Default ``False`` preserves "
            "byte-identical pre-C1.2 behaviour — only the policy poller is "
            "built. Operators flip to ``True`` when the world-model export "
            "pipeline is producing artifacts to OTA-deploy."
        ),
    )
    upload_extensions: tuple[str, ...] = Field(
        (".onnx", ".pt", ".npz", ".json", ".safetensors"),
        description=(
            "File extensions ``training/upload_weights.py::sync_gcs_to_hf`` "
            "publishes to HF Hub when running the cloud-trainer leg of the "
            "OTA loop. Default includes ``.onnx`` + ``.safetensors`` so the "
            "world-model export and HF-native weight formats round-trip "
            "without operator intervention. Stored as a hashable ``tuple`` "
            "(not a ``set``) so the Pydantic schema stays hashable."
        ),
    )
    gcs_artifact_prefix: str = Field(
        "trained/",
        min_length=1,
        description=(
            "Object prefix inside ``gcp.training.training_bucket`` that the "
            "``--from-gcs`` CLI mode lists. Trailing slash preserved verbatim "
            "(forwarded to ``bucket.list_blobs(prefix=...)``). Default "
            "matches the cloud trainer's output convention. MUST be non-empty: "
            "an empty / whitespace prefix would enumerate the entire training "
            "bucket and publish every matching artifact extension to HF Hub — "
            "a high-impact operator footgun. Enforced both by ``min_length=1`` "
            "and the ``_reject_blank_gcs_artifact_prefix`` validator below."
        ),
    )

    @field_validator("gcs_artifact_prefix", mode="after")
    @classmethod
    def _reject_blank_gcs_artifact_prefix(cls, value: str) -> str:
        """Reject whitespace-only prefixes (Copilot MED follow-up, PR #98).

        ``min_length=1`` blocks the literal empty string but lets a whitespace
        prefix like ``"  /"`` slip through, which would also list the bucket
        root once ``bucket.list_blobs`` strips it. Strip + non-empty check is
        the only safe gate.
        """
        if not value.strip():
            msg = (
                "cloud.weight_update.gcs_artifact_prefix must be a non-blank string; "
                "an empty / whitespace prefix would publish every artifact in the "
                "training bucket to HF Hub. Set it to e.g. 'trained/' or a "
                "fleet-specific subpath."
            )
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _warn_on_default_world_model_repo(self) -> WeightUpdatePollConfig:
        """Warn when ``world_model_enabled=True`` but the repo is left at default.

        The default ``world_model_repo_id`` is the maintainer's personal HF
        Hub repo. An operator who flips ``world_model_enabled`` without
        explicitly overriding the repo + filename would silently OTA-deploy
        weights from that repo into production — a footgun the validator
        surfaces at config-load time rather than after the first poll cycle.
        The validator only logs; it does NOT raise, so operators who *intend*
        to consume the default repo (the maintainer themselves, e2e tests)
        keep working.

        Returns:
            The unchanged instance.
        """
        if self.world_model_enabled and self.world_model_repo_id == _WORLD_MODEL_DEFAULT_REPO_ID:
            # Local import — avoid circular-import risk during settings build.
            from mousedroid.logging.setup import get_logger

            _log = get_logger(__name__)
            _log.warning(
                "world_model_poller_using_default_repo",
                repo_id=self.world_model_repo_id,
                hint=(
                    "Set ``cloud.weight_update.world_model_repo_id`` to your "
                    "fleet's HF Hub repo to avoid silently OTA-deploying "
                    "from the maintainer's personal repo."
                ),
            )
        return self


class CloudConfig(BaseModel):
    """Tier C1 cloud retraining loop umbrella configuration.

    Owns the OTA weight-update poller block. Orthogonal to :class:`GCPConfig`
    which covers the Pub/Sub / GCS data pipeline.
    """

    weight_update: WeightUpdatePollConfig = Field(
        default_factory=_settings_default_factory(WeightUpdatePollConfig),
        description="HuggingFace Hub OTA weight-update poller configuration.",
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
    planner_backend: Literal["pyperplan", "fast_downward", "recursive"] = Field(
        "pyperplan",
        description=(
            "Primary symbolic-planning backend. ``pyperplan`` solves the "
            "generated PDDL via Pyperplan in a hard-interruptible subprocess; "
            "``fast_downward`` is not yet wired and transparently uses the "
            "Pyperplan backend; ``recursive`` forces the deterministic "
            "guaranteed-optimal Tower-of-Hanoi solver as the primary. The "
            "recursive solver is ALWAYS the fallback regardless of this value, "
            "so a planner returns a plan for any valid (>= 3-peg) Tower-of-"
            "Hanoi configuration."
        ),
    )
    llm_replanner_enabled: bool = Field(
        False,
        description="Enable LLM-based adaptive replanning on execution failure",
    )
    max_replan_attempts: int = Field(3, gt=0, description="Max replanning attempts before abort")
    planning_timeout_s: float = Field(5.0, gt=0, description="Maximum planning time (s)")
    llm_replanner: LLMReplannerConfig | None = Field(
        None,
        description=(
            "LLM-backed replanner config (None=use legacy symbolic fallback). "
            "Backwards compatible: existing arm runs are unchanged when omitted."
        ),
    )


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


class TrainingReplayConfig(BaseModel):
    """Replay-ingestion settings for RSSM and activation training flows."""

    enabled: bool = Field(
        False,
        description="Enable LMDB-backed replay ingestion alongside or instead of synthetic data",
    )
    source_path: str | None = Field(
        None,
        description="Optional LMDB replay source path (None uses experience.path)",
    )
    terminal_gap_s: float = Field(
        5.0,
        gt=0,
        description="Timestamp gap used to infer episode boundaries in LMDB replay",
    )
    real_episode_ratio: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Number of real replay episodes to include per synthetic episode. "
            "Ignored when synthetic data is absent, in which case all available replay episodes "
            "are used."
        ),
    )
    max_real_episodes: int | None = Field(
        None,
        gt=0,
        description="Optional cap on replay episodes mixed into one training dataset build",
    )
    seed: int | None = Field(
        None,
        description="Optional seed used when selecting a subset of replay episodes",
    )


class ReplayMixerConfig(BaseModel):
    """Phase 2 sim/real episode mixer configuration.

    Mirrors :class:`mousedroid.training.replay.mixer.MixerConfig` so YAML can
    drive the mixer without importing the implementation. All fields default
    to inert values — `alpha_target=0.0` means sim-only and produces the same
    behavior as legacy training paths.
    """

    alpha_target: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Target probability of drawing from the real replay source. "
            "0.0 disables mixing (sim-only); 1.0 is real-only."
        ),
    )
    alpha_ramp_steps: int = Field(
        1000,
        gt=0,
        description=("Number of mix steps to linearly ramp alpha from 0 to alpha_target."),
    )
    chunk_size: int = Field(
        64,
        gt=0,
        description="LMDB read chunk size for the async replay reader.",
    )
    seed: int | None = Field(
        None,
        description="Optional RNG seed for deterministic mixing.",
    )
    log_every_n: int = Field(
        500,
        gt=0,
        description="Emit a `mixer_ratio_check` log every N draws.",
    )
    debug_log_every_n: int = Field(
        0,
        ge=0,
        description=(
            "Emit a structlog DEBUG line ('mixer_draw' / 'replay_chunk_decoded') "
            "every N mixer/reader operations. 0 disables debug logs entirely. "
            "Useful for live triage on Jetson; set to e.g. 100 to surface state "
            "without flooding the journal. The throttle is independent of "
            "`log_every_n`, which controls INFO-level cadence."
        ),
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
    # Phase 5 — MuJoCo->RSSM dynamics-pretraining knobs (opt-in; default OFF so
    # pre-feature behaviour is unchanged). build_rssm_trainable copies
    # rssm_free_nats / rssm_kl_balance_alpha onto the ModelConfig at build time.
    rssm_pretrain_enabled: bool = Field(
        False,
        description="Opt-in: run the MuJoCo->RSSM dynamics pretraining loop in the rssm phase.",
    )
    rssm_free_nats: float = Field(
        1.0, ge=0.0, description="Free-bits floor (nats) for the RSSM training KL."
    )
    rssm_kl_balance_alpha: float = Field(
        0.8, ge=0.0, le=1.0, description="Dreamer KL-balancing weight (prior-update term)."
    )
    rssm_grad_clip: float = Field(
        100.0, gt=0.0, description="Global grad-norm clip for RSSM pretraining."
    )
    rssm_checkpoint_name: str = Field(
        "rssm_pretrained.pt",
        min_length=1,
        description="Filename for the RSSM pretrain checkpoint (under weights_dir).",
    )
    rssm_explore_action_rad_s: float = Field(
        6.0,
        gt=0.0,
        description="Exploration wheel-command bound (rad/s) for sim seed episodes.",
    )
    rssm_explore_smoothing: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="EMA weight on the previous action for the smoothed-random explore policy.",
    )
    rssm_data_seed: int = Field(
        0, ge=0, description="Seed for the sim episode generator (exploration + reset stream)."
    )
    rssm_vision_finetune_enabled: bool = Field(
        False,
        description="Opt-in: run the vision-on RSSM fine-tune phase (renders RGB + MeanPool).",
    )
    rssm_finetune_checkpoint: str = Field(
        "",
        description="Path to the vision-OFF pretrained RSSM checkpoint to fine-tune (migrated on).",
    )
    rssm_finetune_epochs: int = Field(
        50, gt=0, description="Epochs for the vision-on fine-tune phase."
    )
    rssm_vision_checkpoint_name: str = Field(
        "rssm_vision_finetuned.pt",
        min_length=1,
        description="Filename for the vision-on fine-tuned checkpoint (under weights_dir).",
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
    replay: TrainingReplayConfig = Field(
        default_factory=_settings_default_factory(TrainingReplayConfig)
    )
    replay_mixer: ReplayMixerConfig = Field(
        default_factory=_settings_default_factory(ReplayMixerConfig),
        description=(
            "Phase 2 sim/real interleaver configuration. Defaults are inert "
            "(alpha_target=0.0) so existing training pipelines are unchanged."
        ),
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
    write_timeout_s: float = Field(
        0.5,
        gt=0,
        description="Max seconds to wait for speaker buffer space before failing a write",
    )
    write_poll_interval_s: float = Field(
        0.01,
        gt=0,
        description="Seconds between speaker buffer readiness polls",
    )
    reconnect_backoff_initial_s: float = Field(
        0.5,
        gt=0,
        description="Initial backoff delay (seconds) between USB speaker open retries",
    )
    reconnect_backoff_max_s: float = Field(
        10.0,
        gt=0,
        description="Maximum backoff delay (seconds) between USB speaker open retries",
    )
    reconnect_max_attempts: int = Field(
        3,
        ge=1,
        description="Maximum USB speaker open attempts before raising SpeakerUnavailable",
    )


class VoiceConfig(BaseModel):
    """Rocky voice engine configuration."""

    enabled: bool = Field(False, description="Enable Rocky voice output")
    cooldown_s: float = Field(5.0, gt=0, description="Min seconds between utterances")
    personality: str = Field(
        "rocky",
        description=(
            "Voice personality name. If personality_to_model_map contains this key, "
            "its model path overrides tts_model_path; otherwise tts_model_path is used."
        ),
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
    personality_to_model_map: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of personality name → Piper model path. "
            "When the active personality has an entry here it overrides tts_model_path. "
            "Paths must be absolute (resolved inside the container)."
        ),
    )
    event_intensity_thresholds: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-event intensity threshold overrides (0.0-1.0). "
            "Keyed by event name; falls back to intensity_threshold when absent."
        ),
    )
    tts_failure_threshold: int = Field(
        3,
        ge=1,
        description=(
            "Consecutive TTS synthesis failures before promoting warning log to ERROR. "
            "Counter resets on any successful synthesis."
        ),
    )
    cooldown_per_event: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-event cooldown overrides (seconds). Keyed by event name; "
            "events not listed fall back to the global cooldown_s. "
            "Must be > 0 for each entry."
        ),
    )
    token_bucket_capacity: int = Field(
        3,
        gt=0,
        description=(
            "Max tokens per priority-class bucket. Each HIGH/NORMAL priority "
            "class has its own token bucket; EMERGENCY is never rate-limited."
        ),
    )
    token_bucket_refill_rate: float = Field(
        1.0,
        gt=0,
        description="Token-bucket refill rate (tokens/second) per priority class.",
    )
    output_volume: float = Field(
        1.0,
        ge=0.0,
        description=(
            "Linear gain applied to synthesized samples before they reach the "
            "speaker. 1.0 = unity gain; values >1 amplify but are clipped to "
            "[-1, 1] in float32 to keep DAC output in the safe range."
        ),
    )

    @field_validator("personality_to_model_map", mode="after")
    @classmethod
    def _validate_personality_model_map(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate and normalize personality→model path map.

        Strips whitespace from each value so runtime consumers (Piper loader)
        receive exactly the validated path. Empty/whitespace-only and relative
        paths are rejected at schema-load time.
        """
        from pathlib import PurePosixPath

        normalized: dict[str, str] = {}
        for key, value in v.items():
            stripped = value.strip()
            if not stripped:
                raise ValueError(
                    f"personality_to_model_map[{key!r}] must be a non-empty path, got {value!r}"
                )
            if not PurePosixPath(stripped).is_absolute():
                raise ValueError(
                    f"personality_to_model_map[{key!r}] must be an absolute path, got {value!r}"
                )
            normalized[key] = stripped
        return normalized

    @field_validator("event_intensity_thresholds", mode="after")
    @classmethod
    def _validate_event_thresholds(cls, v: dict[str, float]) -> dict[str, float]:
        for key, value in v.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"event_intensity_thresholds[{key!r}] must be in [0.0, 1.0], got {value!r}"
                )
        return v

    @field_validator("cooldown_per_event", mode="after")
    @classmethod
    def _validate_cooldown_per_event(cls, v: dict[str, float]) -> dict[str, float]:
        for key, value in v.items():
            if value <= 0.0:
                raise ValueError(f"cooldown_per_event[{key!r}] must be > 0.0, got {value!r}")
        return v

    @model_validator(mode="after")
    def _validate_event_names_in_phrase_bank(self) -> Self:
        """Ensure event keys reference known phrase-bank events.

        Validates that every key in ``event_intensity_thresholds`` and
        ``cooldown_per_event`` exists in the default phrase bank or has been
        registered via ``phrase_overrides``. Typos that previously fell back
        silently to the global defaults now fail at config-load time.
        """
        # Local import keeps the schema module decoupled from the voice
        # package import graph at module load time.
        from mousedroid.voice.phrase_bank import DEFAULT_PHRASES

        known: set[str] = set(DEFAULT_PHRASES.keys()) | set(self.phrase_overrides.keys())

        bad_thresholds = sorted(set(self.event_intensity_thresholds.keys()) - known)
        bad_cooldowns = sorted(set(self.cooldown_per_event.keys()) - known)

        if bad_thresholds or bad_cooldowns:
            parts: list[str] = []
            if bad_thresholds:
                parts.append(
                    f"event_intensity_thresholds contains unknown event(s): {bad_thresholds!r}"
                )
            if bad_cooldowns:
                parts.append(f"cooldown_per_event contains unknown event(s): {bad_cooldowns!r}")
            parts.append(
                "Known events come from mousedroid.voice.phrase_bank.DEFAULT_PHRASES "
                "and any keys registered via phrase_overrides."
            )
            raise ValueError(" ".join(parts))
        return self

    def resolved_tts_model_path(self) -> str | None:
        """Return the effective TTS model path for the configured personality.

        Resolution order:

        1. ``personality_to_model_map[personality]`` — per-personality override
           (values are validated as absolute paths at schema load time).
        2. ``tts_model_path`` — global fallback (used as-is).

        Returns:
            Path string, or ``None`` when no model is configured.
        """
        mapped = self.personality_to_model_map.get(self.personality)
        return mapped if mapped is not None else self.tts_model_path


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
    idle_action_epsilon: float = Field(
        1e-3,
        gt=0,
        description=(
            "Action magnitude below which the agent is considered idle. "
            "Tolerates small NN-output noise so the SLEEPY path can trigger."
        ),
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


class MCPResourcesConfig(BaseModel):
    """Read-only MCP resource exposure toggles and bounds.

    All limits are config-driven so dashboards and clients can request
    larger or smaller windows without code changes. Defaults mirror the
    telemetry log buffer envelope.
    """

    telemetry_enabled: bool = Field(
        True,
        description="Expose `mousedroid://telemetry/*` resources",
    )
    logs_enabled: bool = Field(
        True,
        description="Expose `mousedroid://logs/tail` resource",
    )
    config_enabled: bool = Field(
        True,
        description="Expose `mousedroid://config/redacted` resource",
    )
    memory_enabled: bool = Field(
        False,
        description="Expose `mousedroid://memory/episodes/recent` resource",
    )
    recent_frames_max: int = Field(
        64,
        gt=0,
        le=4096,
        description="Maximum recent telemetry frames a client may request",
    )
    log_tail_max: int = Field(
        200,
        gt=0,
        le=10_000,
        description="Maximum log entries returnable in a single read",
    )
    config_cache_ttl_s: float = Field(
        1.0,
        gt=0,
        le=60.0,
        description="TTL (seconds) for the redacted-config snapshot cache",
    )


class MCPConfig(BaseModel):
    """Model Context Protocol server configuration.

    The MCP server is fully optional and disabled by default. When
    enabled, it bridges the existing :class:`ToolRegistry`, telemetry
    pipeline, log buffer, and (optionally) episodic memory to any
    MCP-compliant client over stdio, SSE, or streamable HTTP.

    All thresholds, timeouts, and toggles are config-driven; no values
    are hardcoded in the server implementation.
    """

    enabled: bool = Field(False, description="Enable MCP server")
    transport: Literal["stdio", "sse", "streamable_http"] = Field(
        "stdio",
        description="MCP transport protocol",
    )
    host: str = Field(
        "127.0.0.1",
        description="Bind address (loopback by default for safety)",
    )
    port: int = Field(8765, gt=0, le=65535, description="Server port (HTTP/SSE only)")
    auth_token_env_var: str = Field(
        "MOUSEDROID_MCP_TOKEN",
        description="Environment variable holding bearer token (never in YAML)",
    )
    tools_allowlist: list[str] | None = Field(
        None,
        description="Explicit allowlist of tool names; None = all registry tools",
    )
    tools_denylist: list[str] = Field(
        default_factory=list,
        description="Tools that must never be exposed (always wins over allowlist)",
    )
    actuation_tools: list[str] = Field(
        default_factory=lambda: [
            "calibrate_ultrasonic",
            "tensorrt_compile",
            "export_experience",
            "set_velocity",
        ],
        description=(
            "Tools considered actuation/side-effecting (config-driven, not hardcoded). "
            "`emergency_stop` is intentionally NOT in this default list — refusing "
            "an e-stop call during a safety emergency would defeat its purpose. "
            "`read_encoders` is read-only and stays out of the list as well. "
            "Existing YAML overrides win; this default only changes for clients that "
            "never set the field."
        ),
    )
    expose_actuation_tools: bool = Field(
        False,
        description="If False, actuation_tools are hidden from list_tools and refused",
    )
    resources: MCPResourcesConfig = Field(default_factory=MCPResourcesConfig)
    request_timeout_s: float = Field(
        30.0,
        gt=0,
        description="Per-tool-call timeout (seconds)",
    )
    rate_limit_rps: float = Field(
        10.0,
        gt=0,
        description="Per-session token-bucket rate limit (requests per second)",
    )
    sample_telemetry_hz: float = Field(
        10.0,
        gt=0,
        le=60.0,
        description="Background sampler rate that pulls TelemetryFrames into MCP buffer",
    )
    circuit_breaker: CircuitBreakerConfig | None = Field(
        None,
        description="Circuit breaker override; falls back to root cfg.circuit_breaker",
    )
    redact_key_pattern: str = Field(
        r"(?i)token|secret|api[_-]?key|password|credential",
        description="Regex (case-insensitive) for keys whose values must be redacted",
    )
    bind_transport: bool = Field(
        False,
        description=(
            "Bind the configured transport via the optional `mcp` SDK. "
            "Defaults to False so unit tests and in-process callers keep "
            "the bridge usable without spinning up a real server. Set "
            "True in deployment YAML (or via MOUSEDROID_MCP__BIND_TRANSPORT=true) "
            "to expose the server over stdio/SSE/streamable_http."
        ),
    )
    smoke_test_poll_rps: float = Field(
        5.0,
        gt=0,
        description="MCP resource polling rate during the rover hardware smoke (RPS)",
    )
    smoke_test_duration_s: float = Field(
        2.0,
        gt=0,
        description="Duration of the MCP-polling-during-actuation smoke window (s)",
    )
    bind_external: bool = Field(
        False,
        description=(
            "Permit binding a non-loopback host (e.g. 0.0.0.0) for cross-host "
            "OpenClaw access. When False, ``host`` other than 127.0.0.1/localhost "
            "fails validation early so an operator does not accidentally expose "
            "the MCP server. Pair with ``transport`` in {sse, streamable_http} "
            "and a non-empty ``MOUSEDROID_MCP_TOKEN`` env var."
        ),
    )

    @field_validator("tools_denylist")
    @classmethod
    def _no_required_in_denylist(cls, v: list[str]) -> list[str]:
        """Reject denylists that include required liveness tools.

        Args:
            v: Proposed denylist.

        Returns:
            The validated denylist.

        Raises:
            ValueError: If a required tool name is included.
        """
        if "health_check" in v:
            msg = "health_check cannot be denied (required liveness signal)"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _require_token_for_remote(self) -> Self:
        """Refuse to enable a non-loopback transport without an auth token.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If MCP is enabled on a non-loopback bind without
                a token in the configured environment variable.
        """
        if not self.enabled:
            return self
        if self.transport == "stdio":
            return self
        if self.host == "127.0.0.1" or self.host == "localhost":
            return self
        import os

        if not os.environ.get(self.auth_token_env_var):
            msg = (
                f"MCP enabled on non-loopback host '{self.host}' requires the "
                f"{self.auth_token_env_var} environment variable to be set"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _bind_transport_only_for_supported(self) -> Self:
        """Validate bind_transport ↔ transport ↔ external-bind interplay.

        Three guards run here so misconfigurations fail fast at config
        load rather than from a background task hours later:

        1. ``bind_transport=true`` requires a known transport string.
        2. Non-loopback ``host`` requires ``bind_external=true`` so an
           operator never exposes the server by accident.
        3. ``bind_external=true`` requires a non-stdio transport (stdio
           has no listener anyway) AND a non-empty token in the
           ``auth_token_env_var`` env var so the listening port can never
           accept unauthenticated requests.

        Returns:
            The validated config instance.

        Raises:
            ValueError: When any of the three guards trip.
        """
        if not self.bind_transport:
            return self
        supported = {"stdio", "sse", "streamable_http"}
        if self.transport not in supported:
            msg = (
                f"mcp.bind_transport=true is only supported with "
                f"mcp.transport in {sorted(supported)}; "
                f"got mcp.transport={self.transport!r}."
            )
            raise ValueError(msg)
        is_loopback = self.host == "127.0.0.1" or self.host == "localhost"
        if not is_loopback and not self.bind_external:
            msg = (
                f"mcp.host={self.host!r} is non-loopback but "
                "mcp.bind_external is False. Set bind_external=true "
                "explicitly to expose the MCP server outside the host."
            )
            raise ValueError(msg)
        if self.bind_external:
            if self.transport == "stdio":
                msg = "mcp.bind_external=true requires a network transport (sse/streamable_http)"
                raise ValueError(msg)
            import os

            if not os.environ.get(self.auth_token_env_var):
                msg = (
                    f"mcp.bind_external=true requires {self.auth_token_env_var} to be set "
                    "in the environment; refusing to bind without a bearer secret."
                )
                raise ValueError(msg)
        return self


class HarnessTrackerConfig(BaseModel):
    """Task-tracker configuration for the agent harness.

    The tracker persists in-memory state of submitted tasks and their
    acceptance predicates; the orchestrator consults it once per tick.
    Disabled by default — enabling is opt-in and adds no work to the
    30 Hz hot loop while ``enabled=False``.
    """

    enabled: bool = Field(
        False,
        description="Enable in-memory task tracker (None=disabled)",
    )
    history_size: int = Field(
        256,
        gt=0,
        description="Bounded deque size for completed-task history",
    )
    default_timeout_s: float = Field(
        30.0,
        gt=0,
        description="Fallback timeout (s) for tasks that do not specify one",
    )
    max_active: int = Field(
        8,
        gt=0,
        description="Hard cap on simultaneously active tasks",
    )


class HarnessJournalConfig(BaseModel):
    """Persistent agent ledger backend selection and tunables."""

    backend: Literal["null", "jsonl", "lmdb"] = Field(
        "null",
        description="Journal backend (null=disabled)",
    )
    path: Path = Field(
        Path("var/harness/journal"),
        description="Journal directory (LMDB) or file path (JSONL)",
    )
    map_size_gb: float = Field(
        1.0,
        gt=0,
        description="LMDB map size cap in GB (LMDB backend only)",
    )
    flush_every_n: int = Field(
        16,
        gt=0,
        description="Flush LMDB transactions every N writes",
    )
    queue_max: int = Field(
        1024,
        gt=0,
        description="Max queued entries; on full, oldest is dropped (warn-log)",
    )


class HarnessHooksConfig(BaseModel):
    """Tick-loop middleware configuration."""

    enabled_hooks: list[str] = Field(
        default_factory=list,
        description="Names of hooks to wire from the registry (empty=no-op)",
    )
    error_policy: Literal["raise", "warn", "swallow"] = Field(
        "warn",
        description="How hook exceptions are handled",
    )
    journal_events: bool = Field(
        True,
        description="When True, default JournalAppendHook is auto-registered",
    )
    fail_fast: bool = Field(
        False,
        description="Abort tick on first hook failure (overrides error_policy)",
    )


class HarnessApprovalConfig(BaseModel):
    """Human-in-the-loop / policy approval configuration."""

    gate: Literal["auto", "cli", "callback", "policy"] = Field(
        "auto",
        description="Approval gate strategy (auto=AutoApproveGate)",
    )
    require_approval_tool_patterns: list[str] = Field(
        default_factory=list,
        description="fnmatch patterns of tool names that require approval",
    )
    require_approval_skill_patterns: list[str] = Field(
        default_factory=list,
        description="fnmatch patterns of skill names that require approval",
    )
    cli_timeout_s: float = Field(
        30.0,
        gt=0,
        description="CLI approval prompt timeout (s)",
    )
    on_timeout: Literal["deny", "approve"] = Field(
        "deny",
        description="Decision when approval times out (default: fail-closed)",
    )
    callback_dotted_path: str | None = Field(
        None,
        description="Dotted path to async callable for callback gate",
    )


class SkillsConfig(BaseModel):
    """Sub-agent / skill registry configuration."""

    enabled: bool = Field(
        False,
        description="Enable skill registry and sub-agent delegation",
    )
    manifest_glob: str = Field(
        "config/skills/*.yaml",
        description="Glob for YAML skill manifests",
    )
    markdown_agent_dirs: list[Path] = Field(
        default_factory=lambda: [Path("src/mousedroid/agents")],
        description="Directories scanned for markdown agent definitions",
    )
    default_system_prompt: str = Field(
        "",
        description="Fallback system prompt when a skill omits its own",
    )
    backend: Literal["llm_gateway", "anthropic", "noop"] = Field(
        "noop",
        description="Default sub-agent backend",
    )


class HarnessConfig(BaseModel):
    """Top-level agent-harness configuration.

    Bundles task tracker, hook registry, journal, approval gate, and skills
    sub-models. Every nested section ships a working default; the entire
    harness is opt-in via ``Settings.harness`` (None=disabled).
    """

    tracker: HarnessTrackerConfig = Field(
        default_factory=_settings_default_factory(HarnessTrackerConfig),
    )
    hooks: HarnessHooksConfig = Field(
        default_factory=_settings_default_factory(HarnessHooksConfig),
    )
    journal: HarnessJournalConfig = Field(
        default_factory=_settings_default_factory(HarnessJournalConfig),
    )
    approval: HarnessApprovalConfig = Field(
        default_factory=_settings_default_factory(HarnessApprovalConfig),
    )
    skills: SkillsConfig = Field(
        default_factory=_settings_default_factory(SkillsConfig),
    )


class LLMReplannerConfig(BaseModel):
    """Configuration for the LLM-backed arm replanner.

    Disabled by default; when enabled, ``backend`` selects the concrete
    implementation. ``model``, ``max_tokens``, ``temperature`` and the
    request envelope come from this config so no values are hardcoded
    in the backend modules.
    """

    enabled: bool = Field(
        False,
        description="Enable LLM-backed replanning (None=disabled)",
    )
    backend: Literal["null", "llama", "anthropic"] = Field(
        "null",
        description="Replanner backend selection",
    )
    model: str = Field(
        "claude-sonnet-4-6",
        description="Model identifier passed to the backend",
    )
    max_tokens: int = Field(
        1024,
        gt=0,
        description="Per-request max tokens",
    )
    temperature: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    system_prompt: str = Field(
        "",
        description="System prompt passed to the backend",
    )
    api_key_env_var: str = Field(
        "ANTHROPIC_API_KEY",
        description="Env var holding the API key (Anthropic backend only)",
    )
    request_timeout_s: float = Field(
        30.0,
        gt=0,
        description="Per-request timeout (s)",
    )
    max_retries: int = Field(
        3,
        ge=0,
        description="Max exponential-backoff retries on transient errors",
    )


class OpenClawConfig(BaseModel):
    """OpenClaw integration — multi-channel NL control plane.

    OpenClaw runs on a dedicated Mac mini host and dispatches NL commands
    into MouseDroid either via the REST ``POST /api/v1/mission``
    endpoint or via the MCP server (cross-host SSE / streamable_http).
    Both channels enforce the same prompt-injection envelope, the same
    rate-limit token bucket, and (for actuation skills) the same safety
    gate — wiring described in ``docs/openclaw_skills/README.md``.

    Disabled by default. Existing YAML files load unchanged because the
    ``openclaw`` field on :class:`Settings` defaults to ``None`` and every
    field on this model has a default.
    """

    enabled: bool = Field(
        False,
        description="Enable the OpenClaw control plane (REST + MCP gating)",
    )
    mac_mini_origin: str | None = Field(
        None,
        description=(
            "Origin URL of the OpenClaw host (e.g. https://mini.tail-xxxx.ts.net). "
            "When set AND ``telemetry.cors_origins`` is restrictive (does not "
            "contain '*'), :class:`TelemetryServer` automatically appends this "
            "origin to the CORS allow-list at boot so the OpenClaw dashboard "
            "can hit the REST mission endpoint without operators having to "
            "duplicate the URL in two YAML keys."
        ),
    )
    allowed_channels: tuple[Literal["rest", "mcp"], ...] = Field(
        ("rest", "mcp"),
        description="Channels the dispatcher accepts; others are refused.",
    )
    dm_pairing_required: bool = Field(
        True,
        description=(
            "Mac-mini-side hint: enforce dmPolicy=pairing in OpenClaw config. "
            "Mirrored here so operator docs and integration tests stay in sync."
        ),
    )
    max_command_len: int = Field(
        512,
        gt=0,
        description="Maximum NL command length accepted by the dispatcher.",
    )
    shared_memory_path: Path | None = Field(
        None,
        description=(
            "Filesystem path (Tailscale-shared dir or NFS mount) where the "
            "Phase D MarkdownReplayExporter writes MEMORY.md. None disables "
            "the exporter entirely."
        ),
    )
    mdns_service_name: str = Field(
        "_mousedroid._tcp.local.",
        description="Advisory mDNS service name; Tailscale MagicDNS is preferred.",
    )
    command_dedup_window_s: float = Field(
        5.0,
        gt=0,
        description="In-memory TTL window for idempotency_key dedup on REST.",
    )
    export_every_n_ticks: int = Field(
        600,
        gt=0,
        description=(
            "How often the MEMORY.md exporter is allowed to fire (ticks). "
            "At the default 30 Hz control loop this is one snapshot every 20 s."
        ),
    )
    rest_rate_limit_rps: float = Field(
        2.0,
        gt=0,
        description="POST /api/v1/mission token-bucket refill rate (req/s).",
    )
    rest_rate_limit_burst: int = Field(
        4,
        gt=0,
        description="POST /api/v1/mission token-bucket burst capacity.",
    )
    require_actuation_ack: bool = Field(
        True,
        description=(
            "Skills declared with metadata['actuation']=True require this flag "
            "AND mcp.expose_actuation_tools=true. Defence-in-depth even when "
            "an operator flips one of the two by accident."
        ),
    )
    export_max_entries: int = Field(
        32,
        gt=0,
        description=(
            "Cap on episodic samples included in each MEMORY.md snapshot "
            "(threaded into MarkdownReplayExporter)."
        ),
    )
    export_entry_truncate_chars: int = Field(
        240,
        gt=0,
        description=(
            "Per-entry display cap (chars) in MEMORY.md so large episodic "
            "payloads don't blow the OpenClaw agent's context window."
        ),
    )


class USBCEndpointSpec(BaseModel):
    """A single USB-C endpoint the smoke gate expects to find under by-id."""

    name: str = Field(..., min_length=1, description="Logical role, e.g. rover_esp32")
    by_id_glob: str = Field(
        ...,
        min_length=1,
        description="Glob applied under by_id_root (e.g. '*CP2102N*-if00-port0').",
    )
    required: bool = Field(
        True,
        description="If False, missing endpoint is a WARN instead of FAIL.",
    )


class USBCDiscoveryConfig(BaseModel):
    """Config-driven enumeration of USB-C endpoints required for smoke."""

    enabled: bool = Field(
        False,
        description="Master switch — keeps default YAML inert.",
    )
    by_id_root: Path = Field(
        default=Path("/dev/serial/by-id"),
        description="Filesystem root scanned for endpoints.",
    )
    required_endpoints: list[USBCEndpointSpec] = Field(
        default_factory=list,
        description=(
            "Ordered list of USB-C endpoints the smoke gate must resolve. "
            "Each entry declares a ``name`` (operator-readable handle used "
            "in structured logs + factory overrides like "
            "``_resolve_esp32_serial_via_usbc_discovery('rover_esp32')``), a "
            "``by_id_glob`` (matched against ``by_id_root``), and an "
            "optional ``required`` bool (True → MISSING is FAIL, False → "
            "WARN). Empty by default so non-discovery overlays load "
            "unchanged; populate on the rover-side production overlay. "
            "Validated by ``_require_endpoints_when_enabled`` — an empty "
            "list with ``enabled=True`` is rejected at YAML-load time."
        ),
    )

    @model_validator(mode="after")
    def _require_endpoints_when_enabled(self) -> USBCDiscoveryConfig:
        if self.enabled and not self.required_endpoints:
            raise ValueError("usbc_discovery.enabled=true requires at least one required_endpoint")
        return self


class HostEnvConfig(BaseModel):
    """Host env-file durability check (F-017, WS-3.1).

    Drives the WARN-only ``host_env_keys`` preflight check: the deployed
    ``docker.env`` on the rover must carry at least the key set the committed
    template documents, so per-host overrides (``MOUSEDROID_LLM__ENABLED``,
    ``MOUSEDROID_LLM__N_GPU_LAYERS``) survive a reflash instead of silently
    vanishing. Names only — values are never read into a result or log.
    """

    enabled: bool = Field(
        False,
        description=(
            "Master switch - keeps default YAML inert (the check no-ops OK "
            "when disabled). Enable on the Jetson production overlay."
        ),
    )
    env_file: Path = Field(
        default=Path("/etc/mousedroid/docker.env"),
        description="Deployed per-host env file whose key-set is verified.",
    )
    template_file: Path = Field(
        default=Path("/opt/mousedroid/config/docker.env.example"),
        description=(
            "Committed template whose key-set is the required minimum. "
            "Points at the repo checkout on the rover by default."
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
    world_model: WorldModelConfig = Field(
        default_factory=_settings_default_factory(WorldModelConfig)
    )
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
