from __future__ import annotations

import numpy as np
import pytest

from mousedroid.experience.record import (
    SCHEMA_VERSION,
    MouseDroidExperienceRecord,
    deserialize_any,
)


def test_default_values() -> None:
    record = MouseDroidExperienceRecord()
    assert record.schema_version == 1
    assert record.distance_m == 4.0
    assert record.reward == 0.0
    assert record.surprise == 0.0
    assert record.vision_features.shape == (256,)
    assert record.motor_state.shape == (4,)
    assert record.action.shape == (3,)


def test_serialize_deserialize_roundtrip() -> None:
    record = MouseDroidExperienceRecord(
        distance_m=1.5,
        reward=0.7,
        surprise=0.3,
        vision_features=np.ones(256, dtype=np.float32),
        motor_state=np.array([0.1, 0.2, 0.3, 11.0], dtype=np.float32),
        action=np.array([0.5, -0.5, 0.0], dtype=np.float32),
    )
    data = record.serialize()
    restored = MouseDroidExperienceRecord.deserialize(data)
    assert restored.distance_m == pytest.approx(1.5)
    assert restored.reward == pytest.approx(0.7)
    assert restored.surprise == pytest.approx(0.3)
    np.testing.assert_array_almost_equal(restored.vision_features, record.vision_features)
    np.testing.assert_array_almost_equal(restored.motor_state, record.motor_state)
    np.testing.assert_array_almost_equal(restored.action, record.action)


def test_schema_version_is_always_1() -> None:
    assert SCHEMA_VERSION == 1
    record = MouseDroidExperienceRecord()
    assert record.schema_version == 1


def test_deserialize_wrong_schema_version_raises() -> None:
    import msgpack
    bad_data = msgpack.packb({"schema_version": 99})
    with pytest.raises(ValueError, match="Unknown schema version"):
        MouseDroidExperienceRecord.deserialize(bad_data)


def test_deserialize_any_dispatches_correctly() -> None:
    record = MouseDroidExperienceRecord(reward=42.0)
    data = record.serialize()
    restored = deserialize_any(data)
    assert restored.reward == pytest.approx(42.0)


def test_deserialize_any_unknown_version_raises() -> None:
    import msgpack
    bad_data = msgpack.packb({"schema_version": 999})
    with pytest.raises(ValueError, match="Unknown schema version"):
        deserialize_any(bad_data)


def test_timestamp_is_set() -> None:
    record = MouseDroidExperienceRecord()
    assert record.timestamp > 0.0


def test_custom_fields_preserved() -> None:
    vf = np.random.randn(256).astype(np.float32)
    record = MouseDroidExperienceRecord(vision_features=vf, distance_m=0.5)
    data = record.serialize()
    restored = MouseDroidExperienceRecord.deserialize(data)
    np.testing.assert_array_almost_equal(restored.vision_features, vf)
    assert restored.distance_m == pytest.approx(0.5)
