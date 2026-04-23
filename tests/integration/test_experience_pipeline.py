from __future__ import annotations

from pathlib import Path

import numpy as np

from mousedroid.config.schema import ExperienceConfig
from mousedroid.experience.logger import ExperienceLogger
from mousedroid.experience.record import MouseDroidExperienceRecord
from tests import TEST_EXPERIENCE_MAP_SIZE_GB


def test_record_serialize_deserialize_roundtrip() -> None:
    record = MouseDroidExperienceRecord(
        distance_m=1.5,
        reward=0.7,
        surprise=0.3,
        vision_features=np.random.default_rng(0).standard_normal(256).astype(np.float32),
        motor_state=np.array([0.1, 0.2, 0.0, 12.0], dtype=np.float32),
        action=np.array([0.5, 0.0, 0.1], dtype=np.float32),
    )
    data = record.serialize()
    restored = MouseDroidExperienceRecord.deserialize(data)
    assert restored.distance_m == record.distance_m
    assert restored.reward == record.reward
    assert restored.surprise == record.surprise
    np.testing.assert_array_almost_equal(restored.vision_features, record.vision_features)
    np.testing.assert_array_almost_equal(restored.motor_state, record.motor_state)
    np.testing.assert_array_almost_equal(restored.action, record.action)


def test_logger_open_log_count_close(tmp_path: Path) -> None:
    cfg = ExperienceConfig(path=str(tmp_path / "exp_db"), map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB)
    logger = ExperienceLogger(cfg)
    logger.open()
    record = MouseDroidExperienceRecord()
    logger.log(record)
    assert logger.count() == 1
    logger.close()


def test_logger_multiple_records(tmp_path: Path) -> None:
    cfg = ExperienceConfig(path=str(tmp_path / "exp_db2"), map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB)
    logger = ExperienceLogger(cfg)
    logger.open()
    for i in range(5):
        record = MouseDroidExperienceRecord(reward=float(i))
        logger.log(record)
    assert logger.count() == 5
    logger.close()


def test_logger_count_zero_before_open(tmp_path: Path) -> None:
    cfg = ExperienceConfig(path=str(tmp_path / "exp_db3"), map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB)
    logger = ExperienceLogger(cfg)
    assert logger.count() == 0


def test_logger_close_then_count(tmp_path: Path) -> None:
    cfg = ExperienceConfig(path=str(tmp_path / "exp_db4"), map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB)
    logger = ExperienceLogger(cfg)
    logger.open()
    logger.close()
    assert logger.count() == 0
