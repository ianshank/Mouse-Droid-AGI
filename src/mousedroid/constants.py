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
