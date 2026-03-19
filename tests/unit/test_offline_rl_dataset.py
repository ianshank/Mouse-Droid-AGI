"""Tests for offline RL dataset loader from LMDB experience store."""

from __future__ import annotations

import struct
import time

import numpy as np
import pytest
import torch

from mousedroid.config.schema import ExperienceConfig, ModelConfig
from mousedroid.experience.dataset import OfflineRLDataset
from mousedroid.experience.record import MouseDroidExperienceRecord


def _make_record(
    vision_dim: int = 256,
    motor_dim: int = 4,
    action_dim: int = 3,
    reward: float = 1.0,
    timestamp: float | None = None,
) -> MouseDroidExperienceRecord:
    """Create a test experience record with configurable dimensions."""
    return MouseDroidExperienceRecord(
        timestamp=timestamp or time.time(),
        vision_features=np.random.randn(vision_dim).astype(np.float32),
        distance_m=1.5,
        motor_state=np.random.randn(motor_dim).astype(np.float32),
        action=np.random.randn(action_dim).astype(np.float32),
        reward=reward,
        surprise=0.1,
    )


def _populate_lmdb(
    path: str,
    n_records: int = 10,
    time_gap: float = 0.1,
) -> None:
    """Write test records to an LMDB database."""
    import lmdb

    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    base_time = time.time()

    with env.begin(write=True) as txn:
        for i in range(n_records):
            record = _make_record(
                reward=float(i) * 0.1,
                timestamp=base_time + i * time_gap,
            )
            key = struct.pack(">Q", int((base_time + i * time_gap) * 1_000_000) + i)
            txn.put(key, record.serialize())

    env.close()


@pytest.fixture
def experience_cfg(tmp_path: object) -> ExperienceConfig:
    return ExperienceConfig(
        path=str(tmp_path),
        map_size_gb=1,
        flush_every_n=5,
    )


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig()


class TestOfflineRLDatasetOpen:
    """Test dataset opening and closing."""

    def test_open_missing_path_raises(self, model_cfg: ModelConfig) -> None:
        cfg = ExperienceConfig(
            path="/tmp/nonexistent_test_db_12345",
            map_size_gb=1,
            flush_every_n=5,
        )
        dataset = OfflineRLDataset(cfg, model_cfg)
        with pytest.raises(FileNotFoundError):
            dataset.open()

    def test_open_empty_database(
        self,
        experience_cfg: ExperienceConfig,
        model_cfg: ModelConfig,
        tmp_path: object,
    ) -> None:
        import lmdb

        lmdb.open(str(tmp_path), map_size=10 * 1024 * 1024).close()

        dataset = OfflineRLDataset(experience_cfg, model_cfg)
        dataset.open()
        assert len(dataset) == 0
        dataset.close()

    def test_open_populated_database(
        self,
        experience_cfg: ExperienceConfig,
        model_cfg: ModelConfig,
        tmp_path: object,
    ) -> None:
        _populate_lmdb(str(tmp_path), n_records=10)

        dataset = OfflineRLDataset(experience_cfg, model_cfg)
        dataset.open()
        assert len(dataset) == 9  # n_records - 1 transitions
        dataset.close()


class TestOfflineRLDatasetStateDim:
    """Test state dimension computation."""

    def test_default_state_dim(
        self,
        experience_cfg: ExperienceConfig,
        model_cfg: ModelConfig,
    ) -> None:
        dataset = OfflineRLDataset(experience_cfg, model_cfg)
        # vision_dim(256) + distance(1) + motor_state_dim(4) = 261
        assert dataset.state_dim == 261


class TestOfflineRLDatasetTransitions:
    """Test transition loading."""

    def test_empty_dataset_returns_empty_arrays(
        self,
        experience_cfg: ExperienceConfig,
        model_cfg: ModelConfig,
        tmp_path: object,
    ) -> None:
        import lmdb

        lmdb.open(str(tmp_path), map_size=10 * 1024 * 1024).close()

        dataset = OfflineRLDataset(experience_cfg, model_cfg)
        dataset.open()
        states, actions, rewards, next_states, dones = dataset.get_transitions()
        assert states.shape == (0, dataset.state_dim)
        assert actions.shape == (0, model_cfg.action_dim)
        assert rewards.shape == (0,)
        dataset.close()

    def test_transition_shapes(
        self,
        experience_cfg: ExperienceConfig,
        model_cfg: ModelConfig,
        tmp_path: object,
    ) -> None:
        n = 15
        _populate_lmdb(str(tmp_path), n_records=n)

        dataset = OfflineRLDataset(experience_cfg, model_cfg)
        dataset.open()
        states, actions, rewards, next_states, dones = dataset.get_transitions()
        expected_n = n - 1

        assert states.shape == (expected_n, dataset.state_dim)
        assert actions.shape == (expected_n, model_cfg.action_dim)
        assert rewards.shape == (expected_n,)
        assert next_states.shape == (expected_n, dataset.state_dim)
        assert dones.shape == (expected_n,)
        dataset.close()

    def test_done_flags_from_time_gap(
        self,
        experience_cfg: ExperienceConfig,
        model_cfg: ModelConfig,
        tmp_path: object,
    ) -> None:
        """Records with large time gaps should have done=1."""
        import lmdb

        env = lmdb.open(str(tmp_path), map_size=10 * 1024 * 1024)
        base_time = time.time()

        with env.begin(write=True) as txn:
            for i in range(5):
                # Records 0-2 are close together, then big gap, then 3-4
                if i < 3:
                    ts = base_time + i * 0.1
                else:
                    ts = base_time + 100.0 + i * 0.1
                record = _make_record(timestamp=ts)
                key = struct.pack(">Q", int(ts * 1_000_000) + i)
                txn.put(key, record.serialize())

        env.close()

        dataset = OfflineRLDataset(experience_cfg, model_cfg)
        dataset.open()
        _, _, _, _, dones = dataset.get_transitions(terminal_gap_s=5.0)

        # Transition between record 2 and 3 should be marked done
        assert dones[2] == 1.0
        # Other transitions should not be done
        assert dones[0] == 0.0
        assert dones[1] == 0.0
        dataset.close()


class TestOfflineRLDatasetBatchIterator:
    """Test batch iteration."""

    def test_iterate_batches_yields_tensors(
        self,
        experience_cfg: ExperienceConfig,
        model_cfg: ModelConfig,
        tmp_path: object,
    ) -> None:
        _populate_lmdb(str(tmp_path), n_records=20)

        dataset = OfflineRLDataset(experience_cfg, model_cfg)
        dataset.open()

        batches = list(dataset.iterate_batches(batch_size=8, seed=42))
        assert len(batches) > 0

        for batch in batches:
            assert isinstance(batch["states"], torch.Tensor)
            assert isinstance(batch["actions"], torch.Tensor)
            assert isinstance(batch["rewards"], torch.Tensor)
            assert isinstance(batch["next_states"], torch.Tensor)
            assert isinstance(batch["dones"], torch.Tensor)
            assert batch["states"].shape[1] == dataset.state_dim

        dataset.close()

    def test_iterate_batches_empty_dataset(
        self,
        experience_cfg: ExperienceConfig,
        model_cfg: ModelConfig,
        tmp_path: object,
    ) -> None:
        import lmdb

        lmdb.open(str(tmp_path), map_size=10 * 1024 * 1024).close()

        dataset = OfflineRLDataset(experience_cfg, model_cfg)
        dataset.open()

        batches = list(dataset.iterate_batches(batch_size=8))
        assert batches == []
        dataset.close()

    def test_iterate_batches_covers_all_transitions(
        self,
        experience_cfg: ExperienceConfig,
        model_cfg: ModelConfig,
        tmp_path: object,
    ) -> None:
        n = 25
        _populate_lmdb(str(tmp_path), n_records=n)

        dataset = OfflineRLDataset(experience_cfg, model_cfg)
        dataset.open()

        total_samples = sum(
            batch["states"].shape[0]
            for batch in dataset.iterate_batches(batch_size=8, shuffle=False)
        )
        assert total_samples == n - 1
        dataset.close()
