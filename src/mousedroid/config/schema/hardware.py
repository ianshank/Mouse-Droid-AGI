"""Hardware and sensor configuration models.

Camera, ESP32 motor-controller, LiDAR, ultrasonic range sensor, Hailo-8
accelerator, Jetson platform tunables, and the USB-C discovery / host
env-file durability checks used by the Jetson smoke harness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, Field, model_validator

from mousedroid.config.schema._primitives import ESP32CommandSetLiteral, Self


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


WAVESHARE_STOCK_BAUD: Final[int] = 115200
"""UART baud of stock Waveshare ``General_Driver`` firmware.

Confirmed against the vendor host driver (``ugv_rpi/base_ctrl.py`` opens
115200). Lives here — not in :mod:`mousedroid.comms.command_set` — because
its only consumer is the :class:`ESP32Config` after-validator below, and a
config→comms import would invert the layering."""


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
    # F-025 — stock-Waveshare firmware command-set selector.
    command_set: ESP32CommandSetLiteral = Field(
        "legacy",
        description=(
            "Firmware command-set dispatch. Default ``legacy`` preserves "
            "byte-identical pre-F-025 behaviour — the private "
            '``{"T":1,"vx","vy","omega"}`` protocol. ``waveshare_stock`` '
            "speaks the stock ``General_Driver`` firmware "
            "(``waveshareteam/ugv_base_general``): ``CMD_ROS_CTRL`` "
            '``{"T":13,"X","Z"}`` velocity in physical units, a '
            "``CMD_HEART_BEAT_SET`` chassis failsafe armed at connect, and "
            "battery voltage read from the ``FEEDBACK_BASE_INFO`` frame "
            'instead of the legacy ``{"T":2}`` poll (which stock firmware '
            "interprets as a motor-PID WRITE). Consumed by "
            "``mousedroid.comms.command_set.resolve_command_codec``."
        ),
    )
    heartbeat_enabled: bool = Field(
        True,
        description=(
            "Arm the chassis-side heartbeat failsafe (``CMD_HEART_BEAT_SET``) "
            "at connect when ``command_set='waveshare_stock'``. With the "
            "failsafe armed, the firmware halts the motors on its own when no "
            "command arrives within the heartbeat window — the software "
            "watchdog restarts the *container*, but only this stops the "
            "*wheels* after a wedged Jetson or dropped USB link. No-op under "
            "the legacy command set (that firmware has no heartbeat command)."
        ),
    )
    heartbeat_window_multiple: float = Field(
        3.0,
        gt=0,
        description=(
            "Chassis heartbeat window expressed as a multiple of the driver's "
            "worst-case command gap — the largest of 1/keepalive_hz, "
            "command_timeout_s and degraded_poll_interval_s (see "
            "``mousedroid.comms.command_set.heartbeat_window_ms``). With the "
            "shipped defaults that gap is 1.0 s, so the window is 3000 ms: a "
            "hung host halts the wheels within three seconds, while neither a "
            "timed-out read nor a degraded-mode probe cycle can trip it. "
            "Deriving from the max (rather than keepalive_hz alone) means "
            "tightening a timeout automatically tightens the failsafe. Values "
            "below 1.0 put the window inside a legitimate blocking budget and "
            "are warned about at load time."
        ),
    )
    chassis_has_wheel_encoders: bool = Field(
        True,
        description=(
            "Whether the drive motors carry wheel encoders. The WAVE ROVER "
            "chassis ships encoder-less (vendor audit R3) — its stock "
            "firmware reports commanded speed, not measured speed, so the "
            "hardware smoke test's encoder-velocity-fraction assertion is "
            "unsatisfiable there. When ``False`` the motion-quality criterion "
            "re-scopes to 'command accepted + e-stop within budget'. Default "
            "``True`` preserves the historical assertion for chassis that do "
            "have encoders."
        ),
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

    @model_validator(mode="after")
    def _apply_command_set_coupling(self) -> Self:
        """Couple transport settings to the stock command set (F-025).

        Both rules are inert under the default ``legacy`` selector:

        * ``waveshare_stock`` requires ``protocol='serial'`` — stock
          ``General_Driver`` firmware exposes no HTTP ``/cmd`` API, so a
          wifi pairing could only ever silently no-op. Reject at load time.
        * Stock firmware runs its UART at 115 200; the legacy schema default
          is 1 000 000, at which a live stock board reads as line noise
          (vendor audit R2 — this is how a healthy board can be diagnosed
          dead), so the stock baud is derived.

        **Why "still at the schema default" and not "unset"**: an earlier
        revision keyed the derivation off ``"serial_baud" not in
        model_fields_set``, which is dead on every real deployment — the
        shipped overlays pin ``serial_baud: 1000000`` and the loader passes
        the merged YAML straight into ``Settings(**merged)``, so the key is
        always "set". The operator would follow the runbook, get 1 Mbaud
        against stock firmware, and mis-diagnose a healthy board as dead —
        the exact failure F-025 exists to prevent. Keying off the *effective
        value* instead means a config that merely restates the legacy
        default still derives, while a deliberate non-default pin (e.g.
        921600) always wins.

        Plain assignment is safe here: the model is not frozen and
        ``validate_assignment`` is off, so this does not re-enter validation.
        """
        if self.command_set != "waveshare_stock":
            return self
        if self.protocol == "wifi":
            raise ValueError(
                "command_set='waveshare_stock' requires protocol='serial'; "
                "stock General_Driver firmware exposes no HTTP /cmd API"
            )
        legacy_default = type(self).model_fields["serial_baud"].default
        if self.serial_baud == legacy_default:
            self.serial_baud = WAVESHARE_STOCK_BAUD
            # Local import — avoid circular-import risk during settings build
            # (matches the world-model validator's pattern below).
            from mousedroid.logging.setup import get_logger

            get_logger(__name__).info(
                "esp32_stock_baud_derived",
                from_baud=legacy_default,
                to_baud=WAVESHARE_STOCK_BAUD,
                hint="pin esp32.serial_baud to a non-default value to override",
            )
        self._warn_heartbeat_window_shorter_than_blocking_budgets()
        return self

    def _warn_heartbeat_window_shorter_than_blocking_budgets(self) -> None:
        """Warn when the chassis failsafe can fire during normal operation.

        The heartbeat window is derived from ``keepalive_hz``, but the driver
        has its own blocking budgets that stall the command stream for longer:
        a single timed-out read burns ``command_timeout_s``, and degraded mode
        deliberately throttles to one probe per ``degraded_poll_interval_s``.
        When either exceeds the window, the firmware halts the wheels during
        conditions the host considers normal-but-slow — the failsafe working
        exactly as designed against a mis-tuned window.

        This warns rather than raises: the right value is deployment-specific
        (a bench rover on rollers may genuinely want a tight window), and
        refusing to boot over a tuning question would be worse than a loud,
        greppable warning naming the offending field.
        """
        if not self.heartbeat_enabled:
            return
        # Local import — see the baud-derivation branch above.
        from mousedroid.comms.command_set import heartbeat_window_ms
        from mousedroid.logging.setup import get_logger

        window_ms = heartbeat_window_ms(self)
        budgets = {
            "command_timeout_s": self.command_timeout_s,
            "degraded_poll_interval_s": self.degraded_poll_interval_s,
        }
        offenders = {
            name: seconds for name, seconds in budgets.items() if seconds * 1000.0 > window_ms
        }
        if offenders:
            get_logger(__name__).warning(
                "esp32_heartbeat_window_below_blocking_budget",
                window_ms=window_ms,
                offenders_ms={name: seconds * 1000.0 for name, seconds in offenders.items()},
                hint=(
                    "raise esp32.heartbeat_window_multiple (or lower the offending "
                    "timeout) so the chassis failsafe cannot fire mid-operation"
                ),
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


class HealthConfig(BaseModel):
    """Health monitoring configuration."""

    check_interval_s: float = Field(5.0, gt=0, description="Health check interval (s)")
    gpu_temp_warn_c: float = Field(75.0, gt=0, description="GPU temp warning threshold (C)")
    gpu_temp_critical_c: float = Field(90.0, gt=0, description="GPU temp critical threshold (C)")
    memory_warn_pct: float = Field(85.0, gt=0, le=100, description="Memory warning threshold (%)")


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


class MotorLimitsConfig(BaseModel):
    """Physical safety limits for rover drive motors."""

    max_linear_velocity: float = Field(
        default=1.0,
        ge=0.0,
        le=3.0,
        description="Maximum permissible linear velocity limit in metres per second.",
    )
    max_angular_velocity: float = Field(
        default=1.5,
        ge=0.0,
        le=4.0,
        description="Maximum permissible angular velocity limit in radians per second.",
    )
    watchdog_timeout_s: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Watchdog emergency stop timeout in seconds when no heartbeat is received.",
    )


class MotorControllerConfig(BaseModel):
    """Configuration for generic async motor controller drivers."""

    enabled: bool = Field(
        default=True,
        description="Enable motor controller driver subsystem.",
    )
    serial_port: str = Field(
        default="/dev/ttyUSB0",
        description="Default fallback serial port for motor controller UART.",
    )
    baudrate: int = Field(
        default=115200,
        ge=9600,
        le=921600,
        description="Serial baudrate for controller communication.",
    )
    limits: MotorLimitsConfig = Field(
        default_factory=MotorLimitsConfig,
        description="Physical safety velocity limits and watchdog timeout settings.",
    )
