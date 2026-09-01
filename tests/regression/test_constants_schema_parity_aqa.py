"""AQA pin — constants.py's schema-mirroring defaults must equal the live schema.

``src/mousedroid/constants.py`` (imported by 46 modules) carries ~9 constants
whose docstrings explicitly claim to mirror a Pydantic schema default (e.g.
``DEFAULT_VISION_DIM``'s docstring says "mirrors ``ModelConfig.vision_dim``").
Nothing previously asserted that claim against the live schema class —
``tests/unit/test_constants.py`` only sanity-checks the constants in
isolation (positivity, uniqueness). This isn't hypothetical: an audit found
``tests/unit/test_bug_fixes.py`` already locally redeclaring
``DEFAULT_BATTERY_VOLTAGE = 12.0``, diverged from ``constants.py``'s ``12.6``,
undetected until now.

Assertions read the ``FieldInfo`` off ``model_fields`` rather than
instantiating the schema class, per
``.claude/skills/test-tier-mirror/SKILL.md``: a refactor that replaced
``Field(...)`` with a property override must still be caught.
"""

from __future__ import annotations

import pytest

from mousedroid.config.schema.hardware import (
    LidarConfig,
    MotorControllerConfig,
    UltrasonicConfig,
)
from mousedroid.config.schema.voice import MicrophoneConfig
from mousedroid.config.schema.world_model import ModelConfig
from mousedroid.constants import (
    DEFAULT_AFFECT_DIM,
    DEFAULT_AUDIO_CHUNK_SIZE,
    DEFAULT_BELIEF_DIM,
    DEFAULT_DESIRE_DIM,
    DEFAULT_INTENTION_CLASSES,
    DEFAULT_LIDAR_MAX_RANGE_M,
    DEFAULT_MAX_DISTANCE_M,
    DEFAULT_MOTOR_BAUDRATE,
    DEFAULT_VISION_DIM,
)

# (constant value, schema class, field name) — every constants.py entry whose
# docstring explicitly claims to mirror a schema default.
_MIRRORED_PAIRS = [
    (DEFAULT_VISION_DIM, ModelConfig, "vision_dim"),
    (DEFAULT_BELIEF_DIM, ModelConfig, "belief_dim"),
    (DEFAULT_DESIRE_DIM, ModelConfig, "desire_dim"),
    (DEFAULT_INTENTION_CLASSES, ModelConfig, "intention_classes"),
    (DEFAULT_AFFECT_DIM, ModelConfig, "affect_dim"),
    (DEFAULT_MAX_DISTANCE_M, UltrasonicConfig, "max_range_m"),
    (DEFAULT_LIDAR_MAX_RANGE_M, LidarConfig, "max_range_m"),
    (DEFAULT_MOTOR_BAUDRATE, MotorControllerConfig, "baudrate"),
    (DEFAULT_AUDIO_CHUNK_SIZE, MicrophoneConfig, "chunk_size"),
]


@pytest.mark.parametrize(
    ("constant_value", "config_cls", "field_name"),
    _MIRRORED_PAIRS,
    ids=[f"{cls.__name__}.{field}" for _, cls, field in _MIRRORED_PAIRS],
)
def test_constant_mirrors_live_schema_default(
    constant_value: object, config_cls: type, field_name: str
) -> None:
    info = config_cls.model_fields[field_name]
    assert info.default == constant_value, (
        f"constants.py's mirror of {config_cls.__name__}.{field_name} has "
        f"drifted: constant={constant_value!r}, live schema default="
        f"{info.default!r}. Update whichever one is stale."
    )
