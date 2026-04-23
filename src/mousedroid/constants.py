"""Shared dimension constants — single source of truth for default values.

These mirror the defaults in :class:`~mousedroid.config.schema.ModelConfig`
and related config classes.  Import from here instead of duplicating magic
numbers across modules.
"""

from __future__ import annotations

DEFAULT_VISION_DIM: int = 256
"""Default vision feature dimension (mirrors ``ModelConfig.vision_dim``)."""

DEFAULT_MOTOR_STATE_DIM: int = 4
"""Motor state dimension ``[vx, vy, omega, battery_v]``."""

DEFAULT_ACTION_DIM: int = 3
"""Action dimension ``[vx, vy, omega]``."""

DEFAULT_MAX_DISTANCE_M: float = 4.0
"""Default max ultrasonic range in metres (mirrors ``UltrasonicConfig.max_range_m``)."""

DEFAULT_AUDIO_CHUNK_SIZE: int = 1024
"""Default audio chunk size in samples (mirrors ``MicrophoneConfig.chunk_size``)."""

DEFAULT_BATTERY_VOLTAGE: float = 12.6
"""Default battery voltage fallback when sensor data is unavailable."""

N_SENSOR_MODALITIES: int = 4
"""Number of base sensor modalities tracked: vision, ultrasonic, motor, audio."""

N_SENSOR_MODALITIES_WITH_LIDAR: int = 5
"""Number of sensor modalities when LiDAR is active (adds slot [4])."""

SENSOR_SLOT_MAP: dict[str, int] = {
    "vision": 0,
    "ultrasonic": 1,
    "motor": 2,
    "audio": 3,
    "lidar": 4,
}
"""Stable valid-mask slot assignment for encoder and sensing components."""

DEFAULT_LIDAR_FEATURE_DIM: int = 36
"""Default LiDAR feature vector dimension (36 sectors of 10 degrees)."""

DEFAULT_LIDAR_MAX_RANGE_M: float = 12.0
"""Default FHL-LD19 maximum detection range in metres."""

DEFAULT_LIDAR_MIN_RANGE_M: float = 0.15
"""Default FHL-LD19 minimum detection range in metres."""

MILLISECONDS_PER_SECOND: float = 1000.0
"""Conversion factor from seconds to milliseconds."""

WEIGHT_INIT_SCALE: float = 0.01
"""Default scale factor for random weight initialisation in numpy MLPs."""

DEFAULT_BELIEF_DIM: int = 128
"""Default belief latent dimension (mirrors ``ModelConfig.belief_dim``)."""

DEFAULT_DESIRE_DIM: int = 64
"""Default desire latent dimension (mirrors ``ModelConfig.desire_dim``)."""

DEFAULT_INTENTION_CLASSES: int = 10
"""Default number of intention categories (mirrors ``ModelConfig.intention_classes``)."""

DEFAULT_AFFECT_DIM: int = 2
"""Default affect output dim [valence, arousal] (mirrors ``ModelConfig.affect_dim``)."""

DEFAULT_POLICY_HIDDEN_DIM: int = 64
"""Hidden layer dimensionality for PolicyMLP and ValueMLP networks."""

DEFAULT_UCB_CANDIDATES: tuple[float, ...] = (0.5, 1.0, 1.41, 2.0, 3.0)
"""Default UCB exploration constants evaluated during warm-start tuning."""

DEFAULT_UCB_TARGET_MS: float = 50.0
"""Default target median planning latency used during UCB tuning."""

DEFAULT_GRADIENT_SCALE: float = 2.0
"""Default gradient scale for simple mean-squared-error reconstruction losses."""

BACKTRACK_SPEED_THRESHOLD: float = -0.2
"""Forward velocity below this threshold is classified as backtracking."""

APPROACH_CLEAR_DISTANCE_M: float = 1.0
"""Obstacle distance above this threshold is treated as a clear path forward."""

APPROACH_SPEED_THRESHOLD: float = 0.2
"""Planar speed above this threshold is treated as a deliberate forward approach."""

DEFAULT_TARGET_LOOP_MS: float = 33.0
"""Target control loop duration in milliseconds (30 Hz)."""

# ---------------------------------------------------------------------------
# Reproducible RNG seeds for numpy MLP sub-networks
# ---------------------------------------------------------------------------

BELIEF_ENCODER_SEED: int = 42
"""RNG seed for BeliefEncoder weight initialisation."""

DESIRE_ENCODER_SEED: int = 43
"""RNG seed for DesireEncoder weight initialisation."""

INTENTION_PREDICTOR_SEED: int = 44
"""RNG seed for IntentionPredictor weight initialisation."""

AFFECT_ESTIMATOR_SEED: int = 45
"""RNG seed for AffectEstimator weight initialisation."""

POLICY_MLP_SEED: int = 100
"""RNG seed for PolicyMLP weight initialisation."""

VALUE_MLP_SEED: int = 101
"""RNG seed for ValueMLP weight initialisation."""

# ---------------------------------------------------------------------------
# Network constants
# ---------------------------------------------------------------------------

CONNECTIVITY_CHECK_HOST: str = "8.8.8.8"
"""Remote host used for outbound connectivity checks (Google Public DNS)."""

CONNECTIVITY_CHECK_PORT: int = 80
"""Port used for outbound UDP connectivity probes."""

LOOPBACK_IP: str = "127.0.0.1"
"""Loopback address returned when no network is available."""

# ---------------------------------------------------------------------------
# Cognitive core timing defaults
# ---------------------------------------------------------------------------

SLOW_LOOP_INTERVAL_S: float = 1.0
"""Default interval for the slow BDI + metacognitive loop (seconds)."""

SLOW_QUEUE_MAXSIZE: int = 2
"""Maximum backlog for the cognitive slow-loop work queue."""

FAST_STATE_DIM: int = 128
"""Expected dimensionality of fast-tick state vectors."""

# ---------------------------------------------------------------------------
# Telemetry defaults
# ---------------------------------------------------------------------------

LOG_SUBSCRIBER_QUEUE_SIZE: int = 100
"""Default maxsize for log-stream subscriber queues."""

MAX_LOG_ENTRIES: int = 1000
"""Upper bound for log entries returned in a single REST request."""

MDNS_SERVICE_TYPE: str = "_http._tcp.local."
"""mDNS/Zeroconf service type for telemetry advertisement."""

# ---------------------------------------------------------------------------
# Motor state indices
# ---------------------------------------------------------------------------

MOTOR_STATE_BATTERY_INDEX: int = 3
"""Index of battery voltage within the motor_state array."""

TELEMETRY_QUEUE_TIMEOUT_S: float = 1.0
"""Timeout for telemetry queue polling in broadcast/log loops."""

# ---------------------------------------------------------------------------
# Sensor constants
# ---------------------------------------------------------------------------

HC_SR04_TRIGGER_PULSE_S: float = 0.00001
"""HC-SR04 ultrasonic trigger pulse duration (10 microseconds)."""

LIDAR_HEADER_BYTE: int = 0x54
"""FHL-LD19 frame header byte."""

LIDAR_POINTS_PER_FRAME: int = 12
"""Number of measurement points per FHL-LD19 data frame."""

LIDAR_DEFAULT_BAUD_RATE: int = 230400
"""Default serial baud rate for the FHL-LD19 LiDAR."""

LIDAR_FRAME_SIZE: int = 47
"""Total byte length of one FHL-LD19 data frame."""

LIDAR_VER_LEN_BYTE: int = 0x2C
"""FHL-LD19 version/length byte expected after header."""

LIDAR_ANGLE_SCALE: float = 0.01
"""Scale factor to convert raw LD19 angle values to degrees."""

LIDAR_FULL_ROTATION_DEG: float = 360.0
"""Full rotation in degrees for scan assembly boundary detection."""

LIDAR_CRC8_POLYNOMIAL: int = 0x4D
"""CRC-8 polynomial used by LD19 frame checksum."""

LIDAR_SCAN_TIMEOUT_MULTIPLIER: float = 2.0
"""Multiplier on scan period to compute read deadline."""

LIDAR_MM_PER_M: float = 1000.0
"""Number of millimetres per metre — use as divisor (mm→m) or multiplier (m→mm)."""

LIDAR_DEFAULT_MOCK_CONFIDENCE: int = 200
"""Default confidence value for mock LiDAR points."""

GB_TO_BYTES: int = 1_073_741_824
"""Conversion factor from gigabytes to bytes."""

# ---------------------------------------------------------------------------
# Jetson profiler constants
# ---------------------------------------------------------------------------

MILLIDEGREE_DIVISOR: float = 1000.0
"""Divisor to convert millidegree readings to degrees Celsius."""

GPU_LOAD_PERCENTAGE_DIVISOR: float = 10.0
"""Divisor to convert raw GPU load sysfs values to percentage (e.g. 750 / 10 = 75%)."""

# ---------------------------------------------------------------------------
# Numeric stability
# ---------------------------------------------------------------------------

SOFTMAX_EPSILON: float = 1e-8
"""Epsilon for softmax numerical stability."""

IQL_EXP_ADVANTAGE_CLAMP_MAX: float = 100.0
"""Clamp ceiling for exp(beta * advantage) in IQL policy extraction."""

EMA_DEFAULT_ALPHA: float = 0.1
"""Default exponential moving average smoothing factor."""
