"""Unit tests for the Phase-6 WS3 on-device coordinator factory helpers.

Covers the factory wiring branches not exercised by the integration path:

* ``_load_replay_batch`` returns an empty ``(0, input_dim)`` tensor when the
  replay store has no records (the safe empty-store branch the coordinator's
  ``load_batch`` callable relies on);
* ``_count_replay_records`` reports zero for an empty store;
* ``build_on_device_coordinator`` returns a non-None coordinator when enabled.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mousedroid.config.schema import Settings
from mousedroid.factory import (
    _count_replay_records,
    _load_replay_batch,
    build_on_device_coordinator,
)
from mousedroid.training.replay.lmdb_reader import LMDBReplayReader

_INPUT_DIM = 16


def _empty_reader(tmp_path: Path) -> LMDBReplayReader:
    """Build a replay reader over an empty experience store."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": str(tmp_path / "empty_root"), "map_size_gb": 0.01},
        }
    )
    return LMDBReplayReader(cfg.experience)


def test_load_replay_batch_empty_store_returns_empty_tensor(tmp_path: Path) -> None:
    """An empty replay store yields a ``(0, input_dim)`` batch, not an error."""
    reader = _empty_reader(tmp_path)

    batch = _load_replay_batch(reader, _INPUT_DIM, cap=8)

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (0, _INPUT_DIM)


def test_count_replay_records_empty_store_is_zero(tmp_path: Path) -> None:
    """An empty replay store counts zero new records."""
    reader = _empty_reader(tmp_path)

    assert _count_replay_records(reader, cap=8) == 0


def test_build_coordinator_returns_coordinator_when_enabled(tmp_path: Path) -> None:
    """An enabled on-device block wires a non-None coordinator."""
    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "experience": {"path": str(tmp_path / "root"), "map_size_gb": 0.01},
            "on_device_learning": {"enabled": True, "trigger_min_new_records": 5},
        }
    )

    coordinator = build_on_device_coordinator(cfg)

    assert coordinator is not None
