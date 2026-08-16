"""Tests for ExperienceLogger — targeted coverage gaps."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mousedroid.config.schema import ExperienceConfig
from mousedroid.experience.logger import ExperienceLogger
from mousedroid.experience.record import MouseDroidExperienceRecord
from tests import TEST_EXPERIENCE_MAP_SIZE_GB


def _make_record(**kwargs) -> MouseDroidExperienceRecord:
    defaults = {
        "distance_m": 1.0,
        "reward": 0.5,
        "surprise": 0.2,
        "vision_features": np.zeros(256, dtype=np.float32),
        "motor_state": np.zeros(4, dtype=np.float32),
        "action": np.zeros(3, dtype=np.float32),
    }
    defaults.update(kwargs)
    return MouseDroidExperienceRecord(**defaults)


def test_log_when_not_open(tmp_path: Path) -> None:
    cfg = ExperienceConfig(path=str(tmp_path / "db"), map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB)
    logger = ExperienceLogger(cfg)
    # Should not raise; just logs a warning
    logger.log(_make_record())
    assert logger.count() == 0


def test_read_when_not_open(tmp_path: Path) -> None:
    cfg = ExperienceConfig(path=str(tmp_path / "db"), map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB)
    logger = ExperienceLogger(cfg)
    assert logger.read(b"\x00" * 8) is None


def test_read_missing_key(tmp_path: Path) -> None:
    cfg = ExperienceConfig(path=str(tmp_path / "db"), map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB)
    logger = ExperienceLogger(cfg)
    logger.open()
    assert logger.read(b"\xff" * 8) is None
    logger.close()


def test_log_triggers_flush(tmp_path: Path) -> None:
    cfg = ExperienceConfig(
        path=str(tmp_path / "db"),
        map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB,
        flush_every_n=1,
    )
    logger = ExperienceLogger(cfg)
    logger.open()
    logger.log(_make_record())
    # After flush, write_count resets to 0
    assert logger._write_count == 0
    assert logger.count() == 1
    logger.close()


def test_log_no_flush_below_threshold(tmp_path: Path) -> None:
    cfg = ExperienceConfig(
        path=str(tmp_path / "db"),
        map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB,
        flush_every_n=5,
    )
    logger = ExperienceLogger(cfg)
    logger.open()
    logger.log(_make_record())
    assert logger._write_count == 1
    logger.close()


def test_close_idempotent(tmp_path: Path) -> None:
    cfg = ExperienceConfig(path=str(tmp_path / "db"), map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB)
    logger = ExperienceLogger(cfg)
    logger.open()
    logger.close()
    logger.close()  # Should not raise
    assert logger._env is None
