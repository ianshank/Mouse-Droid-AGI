"""Tests for RSSM replay ingestion and synthetic/replay mixing."""

from __future__ import annotations

import struct
import time
from pathlib import Path

import lmdb
import numpy as np
import torch

from mousedroid.config.schema import ExperienceConfig, ModelConfig, TrainingReplayConfig
from mousedroid.constants import SENSOR_SLOT_MAP
from mousedroid.experience.record import MouseDroidExperienceRecord
from tests import TEST_EXPERIENCE_MAP_SIZE_GB
from training.rssm_dataset import RSSMSequenceDataset


def _make_record(
    *,
    timestamp: float,
    action_value: float,
    distance_m: float = 1.5,
    vision_dim: int = 256,
    motor_dim: int = 4,
    action_dim: int = 3,
) -> MouseDroidExperienceRecord:
    return MouseDroidExperienceRecord(
        timestamp=timestamp,
        vision_features=np.full(vision_dim, action_value, dtype=np.float32),
        distance_m=distance_m,
        motor_state=np.full(motor_dim, action_value, dtype=np.float32),
        action=np.full(action_dim, action_value, dtype=np.float32),
        reward=action_value,
        surprise=0.0,
    )


def _populate_lmdb(path: Path, timestamps: list[float], action_value: float) -> None:
    env = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for i, ts in enumerate(timestamps):
            record = _make_record(timestamp=ts, action_value=action_value)
            key = struct.pack(">Q", int(ts * 1_000_000) + i)
            txn.put(key, record.serialize())
    env.close()


def _synthetic_episode(action_value: float) -> list[dict[str, torch.Tensor]]:
    valid_mask = torch.zeros(len(SENSOR_SLOT_MAP), dtype=torch.float32)
    valid_mask[SENSOR_SLOT_MAP["vision"]] = 1.0
    valid_mask[SENSOR_SLOT_MAP["motor"]] = 1.0
    valid_mask[SENSOR_SLOT_MAP["ultrasonic"]] = 1.0
    return [
        {
            "vision": torch.full((256,), action_value, dtype=torch.float32),
            "ultrasonic": torch.full((1,), action_value, dtype=torch.float32),
            "motor_state": torch.full((4,), action_value, dtype=torch.float32),
            "valid_mask": valid_mask.clone(),
            "lidar": torch.full((36,), action_value, dtype=torch.float32),
            "action": torch.full((3,), action_value, dtype=torch.float32),
        }
        for _ in range(2)
    ]


class TestRSSMReplayDataset:
    def test_replay_only_dataset_loads_lmdb_episodes(self, tmp_path: Path) -> None:
        base_time = time.time()
        replay_path = tmp_path / "experience"
        _populate_lmdb(replay_path, [base_time, base_time + 0.1, base_time + 10.0], action_value=7.0)

        dataset = RSSMSequenceDataset(
            tmp_path / "missing.pt",
            seq_len=4,
            replay_cfg=TrainingReplayConfig(enabled=True, terminal_gap_s=5.0),
            experience_cfg=ExperienceConfig(
                path=str(replay_path),
                map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB,
            ),
            model_cfg=ModelConfig(lidar_dim=36, lidar_proj_dim=16),
        )

        batch = dataset[0]

        assert len(dataset) == 2
        assert batch["vision"].shape == (4, 256)
        assert batch["ultrasonic"].shape == (4, 1)
        assert batch["motor_state"].shape == (4, 4)
        assert batch["valid_mask"].shape == (4, 5)
        assert batch["lidar"].shape == (4, 36)
        assert batch["actions"].shape == (4, 3)
        assert torch.all(batch["lidar"] == 0.0)
        assert batch["valid_mask"][0, SENSOR_SLOT_MAP["vision"]] == 1.0
        assert batch["valid_mask"][0, SENSOR_SLOT_MAP["motor"]] == 1.0
        assert batch["valid_mask"][0, SENSOR_SLOT_MAP["audio"]] == 0.0

    def test_replay_ratio_mixes_real_episodes_into_synthetic_corpus(self, tmp_path: Path) -> None:
        data_path = tmp_path / "sequences.pt"
        torch.save([_synthetic_episode(action_value=-1.0)], data_path)

        base_time = time.time()
        replay_path = tmp_path / "experience"
        _populate_lmdb(
            replay_path,
            [base_time, base_time + 0.1, base_time + 10.0, base_time + 10.1],
            action_value=5.0,
        )

        dataset = RSSMSequenceDataset(
            data_path,
            seq_len=4,
            replay_cfg=TrainingReplayConfig(
                enabled=True,
                terminal_gap_s=5.0,
                real_episode_ratio=1.0,
                max_real_episodes=1,
            ),
            experience_cfg=ExperienceConfig(
                path=str(replay_path),
                map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB,
            ),
            model_cfg=ModelConfig(lidar_dim=36, lidar_proj_dim=16),
        )

        assert len(dataset) == 2
        synthetic_batch = dataset[0]
        replay_batch = dataset[1]
        assert torch.all(synthetic_batch["actions"][0] == -1.0)
        assert torch.all(replay_batch["actions"][0] == 5.0)
        assert replay_batch["valid_mask"].shape[-1] == 5