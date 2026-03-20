"""Tests for offline RL training script."""

from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np
import pytest

from mousedroid.config.schema import ExperienceConfig, OfflineRLConfig, Settings
from mousedroid.experience.record import MouseDroidExperienceRecord


def _populate_lmdb(path: str, n_records: int = 50) -> None:
    """Write test records to LMDB."""
    import lmdb

    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    base_time = time.time()

    with env.begin(write=True) as txn:
        for i in range(n_records):
            record = MouseDroidExperienceRecord(
                timestamp=base_time + i * 0.1,
                vision_features=np.random.randn(256).astype(np.float32),
                distance_m=1.0 + np.random.rand(),
                motor_state=np.random.randn(4).astype(np.float32),
                action=np.random.randn(3).astype(np.float32) * 0.3,
                reward=np.random.rand(),
                surprise=0.1,
            )
            key = struct.pack(">Q", int((base_time + i * 0.1) * 1_000_000) + i)
            txn.put(key, record.serialize())

    env.close()


class TestBuildTrainer:
    """Test trainer construction."""

    def test_build_cql_trainer(self) -> None:
        from training.train_offline_rl import _build_trainer

        trainer = _build_trainer(
            algorithm="cql",
            state_dim=261,
            action_dim=3,
            offline_cfg=OfflineRLConfig(),
            device_str="cpu",
        )
        from mousedroid.learning.offline_rl import CQLTrainer

        assert isinstance(trainer, CQLTrainer)

    def test_build_iql_trainer(self) -> None:
        from training.train_offline_rl import _build_trainer

        trainer = _build_trainer(
            algorithm="iql",
            state_dim=261,
            action_dim=3,
            offline_cfg=OfflineRLConfig(),
            device_str="cpu",
        )
        from mousedroid.learning.offline_rl import IQLTrainer

        assert isinstance(trainer, IQLTrainer)

    def test_build_unknown_raises(self) -> None:
        from training.train_offline_rl import _build_trainer

        with pytest.raises(ValueError, match="Unknown offline RL algorithm"):
            _build_trainer(
                algorithm="ppo",
                state_dim=261,
                action_dim=3,
                offline_cfg=OfflineRLConfig(),
                device_str="cpu",
            )


class TestTrainOfflineRL:
    """Test end-to-end offline RL training."""

    def test_cql_training_produces_checkpoint(self, tmp_path: Path) -> None:
        from training.train_offline_rl import train_offline_rl

        db_path = tmp_path / "experience"
        db_path.mkdir()
        _populate_lmdb(str(db_path), n_records=30)

        output_dir = tmp_path / "output"

        cfg = Settings(
            mock_hardware=True,
            experience=ExperienceConfig(
                path=str(db_path),
                map_size_gb=1,
                flush_every_n=5,
            ),
            offline_rl=OfflineRLConfig(
                algorithm="cql",
                epochs=2,
                batch_size=8,
                checkpoint_every_n_epochs=1,
                log_every_n_epochs=1,
                hidden_dim=32,
            ),
        )

        result_dir, stats = train_offline_rl(
            cfg=cfg,
            output_dir=output_dir,
        )

        assert (result_dir / "final.pt").exists()
        assert stats["algorithm"] == "cql"
        assert stats["total_steps"] > 0

    def test_iql_training_produces_checkpoint(self, tmp_path: Path) -> None:
        from training.train_offline_rl import train_offline_rl

        db_path = tmp_path / "experience"
        db_path.mkdir()
        _populate_lmdb(str(db_path), n_records=30)

        output_dir = tmp_path / "output"

        cfg = Settings(
            mock_hardware=True,
            experience=ExperienceConfig(
                path=str(db_path),
                map_size_gb=1,
                flush_every_n=5,
            ),
            offline_rl=OfflineRLConfig(
                algorithm="iql",
                epochs=2,
                batch_size=8,
                checkpoint_every_n_epochs=1,
                log_every_n_epochs=1,
                hidden_dim=32,
            ),
        )

        result_dir, stats = train_offline_rl(
            cfg=cfg,
            output_dir=output_dir,
        )

        assert (result_dir / "final.pt").exists()
        assert stats["algorithm"] == "iql"

    def test_empty_database_returns_no_data(self, tmp_path: Path) -> None:
        import lmdb
        from training.train_offline_rl import train_offline_rl

        db_path = tmp_path / "experience"
        db_path.mkdir()
        lmdb.open(str(db_path), map_size=10 * 1024 * 1024).close()

        cfg = Settings(
            mock_hardware=True,
            experience=ExperienceConfig(
                path=str(db_path),
                map_size_gb=1,
                flush_every_n=5,
            ),
            offline_rl=OfflineRLConfig(epochs=1, batch_size=8),
        )

        _, stats = train_offline_rl(cfg=cfg, output_dir=tmp_path / "out")
        assert stats["error"] == "no_data"
