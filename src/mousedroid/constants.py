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
"""Number of sensor modalities tracked: vision, ultrasonic, motor, audio."""

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

MDNS_SERVICE_TYPE: str = "_http._tcp.local."
"""mDNS/Zeroconf service type for telemetry advertisement."""

# ---------------------------------------------------------------------------
# Motor state indices
# ---------------------------------------------------------------------------

MOTOR_STATE_BATTERY_INDEX: int = 3
"""Index of battery voltage within the motor_state array."""
