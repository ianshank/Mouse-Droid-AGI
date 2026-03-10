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

N_SENSOR_MODALITIES: int = 4
"""Number of sensor modalities tracked: vision, ultrasonic, motor, audio."""
