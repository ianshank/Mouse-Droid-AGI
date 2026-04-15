"""Property-based roundtrip tests for MouseDroidExperienceRecord using Hypothesis."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mousedroid.constants import (
    DEFAULT_ACTION_DIM,
    DEFAULT_MOTOR_STATE_DIM,
    DEFAULT_VISION_DIM,
)
from mousedroid.experience.record import MouseDroidExperienceRecord

# Safe float strategies
_safe_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
_safe_floats32 = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=32
)
_timestamp_floats = st.floats(
    min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False
)


def _float32_array(size: int) -> st.SearchStrategy[np.ndarray]:
    """Return a strategy that produces float32 arrays of *size* elements."""
    return arrays(dtype=np.float32, shape=(size,), elements=_safe_floats32)


@given(
    timestamp=_timestamp_floats,
    vision_features=_float32_array(DEFAULT_VISION_DIM),
    distance_m=_safe_floats,
    motor_state=_float32_array(DEFAULT_MOTOR_STATE_DIM),
    action=_float32_array(DEFAULT_ACTION_DIM),
    reward=_safe_floats,
    surprise=_safe_floats,
)
@settings(max_examples=200)
def test_serialize_deserialize_roundtrip(
    timestamp: float,
    vision_features: np.ndarray,
    distance_m: float,
    motor_state: np.ndarray,
    action: np.ndarray,
    reward: float,
    surprise: float,
) -> None:
    """Serialize then deserialize must reproduce the original record field-by-field."""
    original = MouseDroidExperienceRecord(
        timestamp=timestamp,
        vision_features=vision_features,
        distance_m=distance_m,
        motor_state=motor_state,
        action=action,
        reward=reward,
        surprise=surprise,
    )

    serialized = original.serialize()
    restored = MouseDroidExperienceRecord.deserialize(serialized)

    # Scalar fields
    assert restored.schema_version == original.schema_version
    assert restored.timestamp == original.timestamp, (
        f"timestamp mismatch: {restored.timestamp} != {original.timestamp}"
    )
    assert restored.distance_m == original.distance_m, (
        f"distance_m mismatch: {restored.distance_m} != {original.distance_m}"
    )
    assert restored.reward == original.reward, (
        f"reward mismatch: {restored.reward} != {original.reward}"
    )
    assert restored.surprise == original.surprise, (
        f"surprise mismatch: {restored.surprise} != {original.surprise}"
    )

    # Array fields — use np.allclose for float comparison
    assert np.allclose(restored.vision_features, original.vision_features, atol=1e-6), (
        "vision_features mismatch after roundtrip"
    )
    assert np.allclose(restored.motor_state, original.motor_state, atol=1e-6), (
        "motor_state mismatch after roundtrip"
    )
    assert np.allclose(restored.action, original.action, atol=1e-6), (
        "action mismatch after roundtrip"
    )


@given(
    vision_features=_float32_array(DEFAULT_VISION_DIM),
    motor_state=_float32_array(DEFAULT_MOTOR_STATE_DIM),
    action=_float32_array(DEFAULT_ACTION_DIM),
)
@settings(max_examples=200)
def test_serialized_bytes_are_bytes(
    vision_features: np.ndarray,
    motor_state: np.ndarray,
    action: np.ndarray,
) -> None:
    """serialize() must always return a bytes object."""
    record = MouseDroidExperienceRecord(
        vision_features=vision_features,
        motor_state=motor_state,
        action=action,
    )
    result = record.serialize()
    assert isinstance(result, bytes), f"serialize() returned {type(result)}, expected bytes"


@given(
    vision_features=_float32_array(DEFAULT_VISION_DIM),
    motor_state=_float32_array(DEFAULT_MOTOR_STATE_DIM),
    action=_float32_array(DEFAULT_ACTION_DIM),
)
@settings(max_examples=200)
def test_array_shapes_preserved_after_roundtrip(
    vision_features: np.ndarray,
    motor_state: np.ndarray,
    action: np.ndarray,
) -> None:
    """Array field shapes must be identical before and after roundtrip."""
    original = MouseDroidExperienceRecord(
        vision_features=vision_features,
        motor_state=motor_state,
        action=action,
    )
    restored = MouseDroidExperienceRecord.deserialize(original.serialize())

    assert restored.vision_features.shape == original.vision_features.shape
    assert restored.motor_state.shape == original.motor_state.shape
    assert restored.action.shape == original.action.shape


@given(
    vision_features=_float32_array(DEFAULT_VISION_DIM),
    motor_state=_float32_array(DEFAULT_MOTOR_STATE_DIM),
    action=_float32_array(DEFAULT_ACTION_DIM),
)
@settings(max_examples=100)
def test_embedding_matches_vision_features(
    vision_features: np.ndarray,
    motor_state: np.ndarray,
    action: np.ndarray,
) -> None:
    """The embedding property must return the same array as vision_features."""
    record = MouseDroidExperienceRecord(
        vision_features=vision_features,
        motor_state=motor_state,
        action=action,
    )
    assert np.array_equal(record.embedding, record.vision_features), (
        "embedding property does not match vision_features"
    )


def test_schema_version_mismatch_raises() -> None:
    """Deserializing a record with a wrong schema version must raise ValueError."""
    import msgpack

    bad_data = msgpack.packb({"schema_version": 999, "foo": "bar"})
    try:
        MouseDroidExperienceRecord.deserialize(bad_data)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown schema version")
