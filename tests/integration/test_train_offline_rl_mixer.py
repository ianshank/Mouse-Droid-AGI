"""Phase 2.1 — sim/real ``RealSimMixer`` integration in ``train_offline_rl``.

These tests assert that the new ``offline_rl.use_replay_mixer`` toggle:

1. **Defaults to False** — single-LMDB path is byte-identical to pre-Phase-2.1
   training (already covered by ``test_phase21_bc_into_offline_rl.py``; this
   suite only adds the explicit-False assertion).
2. **Falls back gracefully** when ``use_replay_mixer=True`` but
   ``cfg.training.replay`` does not supply a distinct source path — emits
   ``offline_rl_mixer_requested_but_unavailable`` warning and trains over the
   single LMDB.
3. **Engages the mixer** when both ``use_replay_mixer=True`` and a distinct
   ``cfg.training.replay.source_path`` are configured — emits
   ``offline_rl_mixer_active`` info log and trains to completion.

Both LMDB stores are tiny synthetic fixtures seeded for determinism.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from mousedroid.config.schema import (
    ExperienceConfig,
    GPUConfig,
    OfflineRLConfig,
    ReplayMixerConfig,
    Settings,
    TrainingConfig,
    TrainingReplayConfig,
)
from mousedroid.experience.record import MouseDroidExperienceRecord
from tests import TEST_EXPERIENCE_MAP_SIZE_GB

# -----------------------------------------------------------------------------
# Test-only constants (no magic numbers in test bodies).
# -----------------------------------------------------------------------------
_RECORDS_PER_TEST = 32
_VISION_DIM = 256
_MOTOR_DIM = 4
_ACTION_DIM = 3
_HIDDEN_DIM = 16
_EPOCHS = 1
_BATCH_SIZE = 8
_BC_WEIGHT_ACTIVE = 0.5
_RNG_SEED_SIM = 11
_RNG_SEED_REAL = 22
_MIXER_ALPHA_TARGET = 0.4
_MIXER_ALPHA_RAMP = 4


def _populate_lmdb(path: str, *, n_records: int, seed: int) -> None:
    """Write ``n_records`` deterministic records to an LMDB store."""
    import lmdb

    rng = np.random.default_rng(seed)
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    base_time = time.time()

    with env.begin(write=True) as txn:
        for i in range(n_records):
            record = MouseDroidExperienceRecord(
                timestamp=base_time + i * 0.1,
                vision_features=rng.standard_normal(_VISION_DIM).astype(np.float32),
                distance_m=1.0 + float(rng.random()),
                motor_state=rng.standard_normal(_MOTOR_DIM).astype(np.float32),
                action=rng.standard_normal(_ACTION_DIM).astype(np.float32) * 0.3,
                reward=float(rng.random()),
                surprise=0.1,
            )
            key = struct.pack(">Q", int((base_time + i * 0.1) * 1_000_000) + i)
            txn.put(key, record.serialize())

    env.close()


def _make_cfg(
    *,
    sim_db: Path,
    real_db: Path | None,
    use_mixer: bool,
    bc_weight: float = _BC_WEIGHT_ACTIVE,
) -> Settings:
    """Build a Settings object with CPU device + optional replay mixer."""
    training_kwargs: dict[str, Any] = {
        "epochs": _EPOCHS,
        "batch_size": _BATCH_SIZE,
        "gpu": GPUConfig(device="cpu", require_cuda=False),
        "replay_mixer": ReplayMixerConfig(
            alpha_target=_MIXER_ALPHA_TARGET,
            alpha_ramp_steps=_MIXER_ALPHA_RAMP,
            seed=0,
        ),
    }
    if real_db is not None:
        training_kwargs["replay"] = TrainingReplayConfig(
            enabled=True,
            source_path=str(real_db),
        )
    return Settings(
        mock_hardware=True,
        experience=ExperienceConfig(
            path=str(sim_db),
            map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB,
            flush_every_n=5,
        ),
        offline_rl=OfflineRLConfig(
            epochs=_EPOCHS,
            batch_size=_BATCH_SIZE,
            hidden_dim=_HIDDEN_DIM,
            checkpoint_every_n_epochs=_EPOCHS + 1,
            log_every_n_epochs=1,
            real_supervised_weight=bc_weight,
            use_replay_mixer=use_mixer,
        ),
        training=TrainingConfig(**training_kwargs),
    )


def _seed_all(seed: int = 0) -> None:
    """Seed every RNG ``train_offline_rl`` may consult."""
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestUseReplayMixerDefaultFalse:
    """The new toggle defaults to False — no behavior change for legacy YAML."""

    def test_default_offline_rl_config_does_not_enable_mixer(self) -> None:
        cfg = OfflineRLConfig()
        assert cfg.use_replay_mixer is False


class TestMixerFallbackWhenUnavailable:
    """``use_replay_mixer=True`` with no distinct real path falls back safely."""

    def test_falls_back_to_single_lmdb_when_replay_disabled(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from training.train_offline_rl import train_offline_rl

        sim_db = tmp_path / "sim"
        sim_db.mkdir()
        _populate_lmdb(str(sim_db), n_records=_RECORDS_PER_TEST, seed=_RNG_SEED_SIM)

        # use_mixer=True but no real DB → mixer requested but unavailable
        cfg = _make_cfg(sim_db=sim_db, real_db=None, use_mixer=True)
        _seed_all()

        _result_dir, stats = train_offline_rl(cfg=cfg, output_dir=tmp_path / "out")
        captured = capsys.readouterr()

        # Training must still complete on the single LMDB.
        assert "error" not in stats
        # The fallback warning must have fired (structlog → stdout).
        assert (
            "offline_rl_mixer_requested_but_unavailable" in captured.out
        ), f"fallback warning missing from stdout:\n{captured.out!r}"

    def test_falls_back_when_source_path_is_identical(self, tmp_path: Path) -> None:
        """If source_path == experience.path, fall back to single-LMDB path."""
        from training.train_offline_rl import train_offline_rl

        sim_db = tmp_path / "sim"
        sim_db.mkdir()
        _populate_lmdb(str(sim_db), n_records=_RECORDS_PER_TEST, seed=_RNG_SEED_SIM)

        cfg = _make_cfg(sim_db=sim_db, real_db=sim_db, use_mixer=True)
        _seed_all()
        _result_dir, stats = train_offline_rl(cfg=cfg, output_dir=tmp_path / "out")

        assert "error" not in stats
        assert "final_bc_loss" in stats  # BC still applied to the single LMDB


class TestMixerActiveWithDistinctRealStore:
    """``use_replay_mixer=True`` with a distinct real LMDB engages the mixer."""

    def test_mixer_log_fires_and_training_completes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from training.train_offline_rl import train_offline_rl

        sim_db = tmp_path / "sim"
        real_db = tmp_path / "real"
        sim_db.mkdir()
        real_db.mkdir()
        _populate_lmdb(str(sim_db), n_records=_RECORDS_PER_TEST, seed=_RNG_SEED_SIM)
        _populate_lmdb(str(real_db), n_records=_RECORDS_PER_TEST, seed=_RNG_SEED_REAL)

        cfg = _make_cfg(sim_db=sim_db, real_db=real_db, use_mixer=True)
        _seed_all()

        _result_dir, stats = train_offline_rl(cfg=cfg, output_dir=tmp_path / "out")
        captured = capsys.readouterr()

        # Training must complete with BC loss recorded.
        assert "error" not in stats
        assert "final_bc_loss" in stats
        bc_loss = stats["final_bc_loss"]
        assert isinstance(bc_loss, float)
        assert np.isfinite(bc_loss)

        # The mixer-active info log must have fired (single-shot at startup).
        assert (
            "offline_rl_mixer_active" in captured.out
        ), f"offline_rl_mixer_active log missing from stdout:\n{captured.out!r}"
