"""Regression tests guaranteeing Phase 2 replay changes are backwards compatible.

These tests pin the invariant from the plan: ``TrainingReplayConfig.enabled=False``
must produce a dataset byte-identical to pre-Phase-2 behavior, and
``use_chunked_reader=False`` (the default) must yield the exact same episode
contents as the streaming reader path when both are pointed at the same LMDB.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import lmdb
import numpy as np
import torch
from training.rssm_dataset import (
    RSSMSequenceDataset,
    _load_replay_episodes,
)

from mousedroid.config.schema import (
    ExperienceConfig,
    ModelConfig,
    TrainingReplayConfig,
)
from mousedroid.experience.record import MouseDroidExperienceRecord

GB_TO_BYTES = 1024**3


def _model_cfg() -> ModelConfig:
    return ModelConfig()


def _populate(path: Path, n: int) -> None:
    map_size = max(1, math.ceil(0.001 * GB_TO_BYTES))
    env = lmdb.open(str(path), map_size=map_size, max_dbs=1)
    try:
        with env.begin(write=True) as txn:
            for i in range(n):
                rec = MouseDroidExperienceRecord(
                    timestamp=float(i) * 0.1,
                    vision_features=np.full(8, float(i), dtype=np.float32),
                    distance_m=0.5 + 0.01 * i,
                    motor_state=np.zeros(4, dtype=np.float32),
                    action=np.zeros(3, dtype=np.float32),
                    reward=float(i),
                    surprise=0.0,
                )
                txn.put(struct.pack(">Q", i), rec.serialize())
    finally:
        env.close()


def _make_synth_data(tmp_path: Path) -> Path:
    """Create a tiny .pt file mirroring SyntheticSequenceGenerator output."""
    model_cfg = _model_cfg()
    n_modalities = 4  # SENSOR_SLOT_MAP currently has 4 entries; defensive default.
    from mousedroid.constants import SENSOR_SLOT_MAP

    n_modalities = len(SENSOR_SLOT_MAP)
    episode = []
    for t in range(5):
        episode.append(
            {
                "vision": torch.full((model_cfg.vision_dim,), float(t), dtype=torch.float32),
                "ultrasonic": torch.zeros(model_cfg.ultrasonic_dim, dtype=torch.float32),
                "motor_state": torch.zeros(model_cfg.motor_state_dim, dtype=torch.float32),
                "valid_mask": torch.ones(n_modalities, dtype=torch.float32),
                "lidar": torch.zeros(model_cfg.lidar_dim, dtype=torch.float32),
                "action": torch.zeros(model_cfg.action_dim, dtype=torch.float32),
            }
        )
    pt_path = tmp_path / "synthetic.pt"
    torch.save([episode], pt_path)
    return pt_path


def test_replay_disabled_dataset_identical_to_synth_only(tmp_path: Path) -> None:
    """Plan invariant: enabled=False must NOT trigger any replay loading."""
    pt_path = _make_synth_data(tmp_path)
    model_cfg = _model_cfg()

    # Path A: bare synthetic, no replay config provided.
    ds_a = RSSMSequenceDataset(pt_path, seq_len=5)

    # Path B: synthetic + replay_cfg with enabled=False (the default).
    ds_b = RSSMSequenceDataset(
        pt_path,
        seq_len=5,
        replay_cfg=TrainingReplayConfig(),  # enabled=False
        experience_cfg=ExperienceConfig(path=str(tmp_path / "nonexistent")),
        model_cfg=model_cfg,
    )

    assert len(ds_a) == len(ds_b)
    a = ds_a[0]
    b = ds_b[0]
    for key in a:
        assert torch.equal(a[key], b[key]), f"mismatch in '{key}'"


def test_chunked_reader_produces_identical_episodes_to_in_memory_loader(
    tmp_path: Path,
) -> None:
    """The streaming reader path must yield the same episode contents as the
    legacy in-memory ``OfflineRLDataset`` path.

    Episode boundaries are inferred from ``terminal_gap_s`` in both paths; with
    a uniform 0.1-second timestamp step and ``terminal_gap_s=5.0`` the entire
    DB is one episode.
    """
    db = tmp_path / "replay"
    _populate(db, n=10)
    model_cfg = _model_cfg()
    exp_cfg = ExperienceConfig(path=str(db), map_size_gb=0.001, flush_every_n=1)

    legacy = TrainingReplayConfig(
        enabled=True,
        terminal_gap_s=5.0,
        use_chunked_reader=False,
    )
    chunked = TrainingReplayConfig(
        enabled=True,
        terminal_gap_s=5.0,
        use_chunked_reader=True,
        chunk_size=3,
    )

    eps_legacy = _load_replay_episodes(legacy, exp_cfg, model_cfg)
    eps_chunked = _load_replay_episodes(chunked, exp_cfg, model_cfg)

    assert len(eps_legacy) == len(eps_chunked) == 1
    assert len(eps_legacy[0]) == len(eps_chunked[0]) == 10

    for step_legacy, step_chunked in zip(eps_legacy[0], eps_chunked[0], strict=True):
        for key in step_legacy:
            assert torch.equal(step_legacy[key], step_chunked[key]), (
                f"mismatch at step in field '{key}'"
            )


def test_chunked_reader_respects_terminal_gap(tmp_path: Path) -> None:
    """Records spaced by > terminal_gap_s must be split into separate episodes."""
    db = tmp_path / "replay"
    map_size = max(1, math.ceil(0.001 * GB_TO_BYTES))
    env = lmdb.open(str(db), map_size=map_size, max_dbs=1)
    try:
        with env.begin(write=True) as txn:
            # Two episodes of 3 records, separated by a 10 s gap.
            ts_seq = [0.0, 0.1, 0.2, 10.5, 10.6, 10.7]
            for i, ts in enumerate(ts_seq):
                rec = MouseDroidExperienceRecord(
                    timestamp=ts,
                    vision_features=np.zeros(8, dtype=np.float32),
                    distance_m=0.5,
                    motor_state=np.zeros(4, dtype=np.float32),
                    action=np.zeros(3, dtype=np.float32),
                    reward=float(i),
                    surprise=0.0,
                )
                txn.put(struct.pack(">Q", i), rec.serialize())
    finally:
        env.close()

    cfg = TrainingReplayConfig(
        enabled=True,
        terminal_gap_s=5.0,
        use_chunked_reader=True,
        chunk_size=2,
    )
    exp_cfg = ExperienceConfig(path=str(db), map_size_gb=0.001, flush_every_n=1)
    episodes = _load_replay_episodes(cfg, exp_cfg, _model_cfg())
    assert len(episodes) == 2
    assert len(episodes[0]) == 3
    assert len(episodes[1]) == 3


def test_default_replay_config_fields_preserve_behavior() -> None:
    """All Phase 2 additions to TrainingReplayConfig must default to no-op values."""
    cfg = TrainingReplayConfig()
    assert cfg.enabled is False
    assert cfg.use_chunked_reader is False
    assert cfg.alpha_target == 0.0
    assert cfg.alpha_ramp_steps == 0
    assert cfg.strict_schema is False
    # Existing fields unchanged
    assert cfg.real_episode_ratio == 0.0
    assert cfg.max_real_episodes is None
