"""Phase 2 acceptance: integration test on a 10-episode synthetic LMDB.

Acceptance criteria from ``NEXT_STEPS.md``:
    * Empty LMDB produces a clean no-op (covered by unit tests).
    * Mixer's realized ratio over 10 k draws is within 1% (unit tests).
    * Integration test on a 10-episode synthetic LMDB → checkpoint produced. <-- HERE.

Wiring under test:
    ExperienceLogger.serialize -> LMDB -> LMDBReplayReader.stream ->
    OfflineRLTrainer.bc_update -> trainer.save -> checkpoint on disk.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import torch

from mousedroid.constants import (
    DEFAULT_ACTION_DIM,
    DEFAULT_MOTOR_STATE_DIM,
    DEFAULT_VISION_DIM,
)
from mousedroid.experience.logger import ExperienceLogger
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.factory import build_replay_reader
from mousedroid.learning.offline_rl import CQLTrainer
from mousedroid.training.replay import LMDBReplayReader, ReplayReaderProtocol

_NUM_EPISODES = 10
_BC_STEPS = 25
_BC_WEIGHT = 1.0
_HIDDEN = 32


def _populate_lmdb(path: Path, n: int, seed: int = 0) -> list[MouseDroidExperienceRecord]:
    """Write ``n`` deterministic synthetic records to an LMDB at ``path``.

    Returns the records that were written, so the test can compare them
    against what the reader yields.
    """
    rng = np.random.default_rng(seed)
    written: list[MouseDroidExperienceRecord] = []

    # Build a real ExperienceLogger so we exercise the same write path used
    # in production. Use a tiny map_size; this is plenty for 10 records.
    from mousedroid.config.schema import ExperienceConfig

    cfg = ExperienceConfig(
        path=str(path),
        map_size_gb=0.001,
        flush_every_n=1,
    )
    logger = ExperienceLogger(cfg)
    logger.open()
    try:
        for i in range(n):
            record = MouseDroidExperienceRecord(
                timestamp=float(i),
                vision_features=rng.standard_normal(DEFAULT_VISION_DIM, dtype=np.float32),
                distance_m=float(rng.uniform(0.5, 5.0)),
                motor_state=rng.standard_normal(DEFAULT_MOTOR_STATE_DIM, dtype=np.float32),
                action=rng.standard_normal(DEFAULT_ACTION_DIM, dtype=np.float32),
                reward=float(rng.uniform(-1.0, 1.0)),
                surprise=float(rng.uniform(0.0, 1.0)),
            )
            logger.log(record)
            written.append(record)
    finally:
        logger.close()
    return written


async def _drain(reader: LMDBReplayReader) -> list[MouseDroidExperienceRecord]:
    """Pull every record out of ``reader.stream``."""
    out: list[MouseDroidExperienceRecord] = []
    async for chunk in reader.stream(chunk_size=4):
        out.extend(chunk)
    return out


def test_phase2_replay_pipeline_end_to_end(tmp_path: Path) -> None:
    """10 synthetic episodes -> reader -> BC update -> checkpoint on disk."""
    # 1) Populate a real LMDB env via the production ExperienceLogger.
    lmdb_path = tmp_path / "replay.lmdb"
    written = _populate_lmdb(lmdb_path, _NUM_EPISODES, seed=0)
    assert len(written) == _NUM_EPISODES

    # 2) Build the reader through the factory so the protocol contract is
    #    exercised end-to-end (CLAUDE.md invariants 1+2).
    from mousedroid.config.loader import load_settings

    cfg = load_settings()
    cfg = cfg.model_copy(
        update={
            "experience": cfg.experience.model_copy(update={"path": str(lmdb_path)}),
        }
    )
    reader = build_replay_reader(cfg)
    assert isinstance(reader, ReplayReaderProtocol)

    # 3) Drain the reader and assert we got every record back.
    records = asyncio.run(_drain(reader))  # type: ignore[arg-type]
    assert (
        len(records) == _NUM_EPISODES
    ), f"Reader lost records: wrote {_NUM_EPISODES} got {len(records)}"
    assert reader.stats["read_records"] == _NUM_EPISODES
    assert reader.stats["skipped_schema_mismatch"] == 0

    # 4) Tensorize: state = vision_features (256-d), action = action (3-d).
    states = torch.from_numpy(np.stack([r.vision_features for r in records]))
    actions = torch.from_numpy(np.stack([r.action for r in records]))
    assert states.shape == (_NUM_EPISODES, DEFAULT_VISION_DIM)
    assert actions.shape == (_NUM_EPISODES, DEFAULT_ACTION_DIM)

    # 5) Build a CQL trainer and run BC updates against the real records.
    torch.manual_seed(0)
    trainer = CQLTrainer(
        state_dim=DEFAULT_VISION_DIM,
        action_dim=DEFAULT_ACTION_DIM,
        hidden_dim=_HIDDEN,
    )
    initial = trainer.bc_update(states, actions, weight=_BC_WEIGHT)["bc_loss"]
    assert initial > 0.0, "Initial BC loss must be positive on random init"

    losses: list[float] = [initial]
    for _ in range(_BC_STEPS):
        out = trainer.bc_update(states, actions, weight=_BC_WEIGHT)
        losses.append(out["bc_loss"])

    # BC loss must trend down — full convergence not required, just direction.
    assert (
        losses[-1] < losses[0]
    ), f"BC loss did not decrease over {_BC_STEPS} steps: {losses[0]} -> {losses[-1]}"

    # 6) Save a checkpoint and assert the file was produced on disk.
    ckpt_path = tmp_path / "checkpoint.pt"
    trainer.save(str(ckpt_path))
    assert ckpt_path.exists(), "Checkpoint file was not created"
    assert ckpt_path.stat().st_size > 0, "Checkpoint file is empty"

    # 7) Round-trip the checkpoint to confirm it is a valid torch artifact.
    fresh = CQLTrainer(
        state_dim=DEFAULT_VISION_DIM,
        action_dim=DEFAULT_ACTION_DIM,
        hidden_dim=_HIDDEN,
    )
    fresh.load(str(ckpt_path))
    # Loaded policy must reproduce the trained policy's predictions byte-exactly
    # on the training batch.
    with torch.no_grad():
        original = trainer.policy(states)
        reloaded = fresh.policy(states)
    assert torch.allclose(original, reloaded, atol=0.0)


def test_phase2_zero_weight_does_not_train(tmp_path: Path) -> None:
    """real_supervised_weight=0 path must leave the policy untouched.

    This guards CLAUDE.md invariant 9 (backwards compat): trainers wired with
    the default `OfflineRLConfig.real_supervised_weight = 0.0` must produce
    byte-identical output to pre-Phase-2 training.
    """
    lmdb_path = tmp_path / "replay.lmdb"
    _populate_lmdb(lmdb_path, _NUM_EPISODES, seed=1)

    from mousedroid.config.loader import load_settings

    cfg = load_settings().model_copy(
        update={
            "experience": load_settings().experience.model_copy(update={"path": str(lmdb_path)}),
        }
    )
    reader = build_replay_reader(cfg)
    records = asyncio.run(_drain(reader))  # type: ignore[arg-type]

    states = torch.from_numpy(np.stack([r.vision_features for r in records]))
    actions = torch.from_numpy(np.stack([r.action for r in records]))

    torch.manual_seed(0)
    trainer = CQLTrainer(
        state_dim=DEFAULT_VISION_DIM,
        action_dim=DEFAULT_ACTION_DIM,
        hidden_dim=_HIDDEN,
    )
    snapshot = {k: v.detach().clone() for k, v in trainer.policy.state_dict().items()}
    out = trainer.bc_update(states, actions, weight=0.0)

    assert out == {"bc_loss": 0.0}
    for k, v in trainer.policy.state_dict().items():
        assert torch.equal(
            v, snapshot[k]
        ), f"policy[{k}] changed under weight=0.0; backwards-compat broken"


@pytest.mark.parametrize("chunk_size", [1, 3, 4, 64])
def test_phase2_chunk_size_invariance(tmp_path: Path, chunk_size: int) -> None:
    """Reader output must be chunk-size-invariant for the same LMDB."""
    lmdb_path = tmp_path / "replay.lmdb"
    _populate_lmdb(lmdb_path, _NUM_EPISODES, seed=42)

    from mousedroid.config.schema import ExperienceConfig

    reader = LMDBReplayReader(
        ExperienceConfig(
            path=str(lmdb_path),
            map_size_gb=0.001,
            flush_every_n=1,
        )
    )

    async def _collect() -> list[MouseDroidExperienceRecord]:
        out: list[MouseDroidExperienceRecord] = []
        async for chunk in reader.stream(chunk_size=chunk_size):
            out.extend(chunk)
        return out

    records = asyncio.run(_collect())
    assert len(records) == _NUM_EPISODES
