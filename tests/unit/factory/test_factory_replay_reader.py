"""Tests for :func:`mousedroid.factory.build_replay_reader`."""

from __future__ import annotations

from pathlib import Path

from mousedroid.config.loader import load_settings
from mousedroid.factory import build_replay_reader
from mousedroid.training.replay import ReplayReaderProtocol


def test_build_replay_reader_returns_protocol(tmp_path: Path) -> None:
    """Factory yields a reader satisfying the runtime-checkable protocol."""
    cfg = load_settings()
    # Redirect to an empty tmp path so we never touch the operator's real LMDB.
    cfg = cfg.model_copy(
        update={
            "experience": cfg.experience.model_copy(update={"path": str(tmp_path / "replay.lmdb")})
        }
    )
    reader = build_replay_reader(cfg)
    assert isinstance(reader, ReplayReaderProtocol)
    assert reader.stats == {
        "read_records": 0,
        "skipped_schema_mismatch": 0,
        "chunks_yielded": 0,
    }


def test_build_replay_reader_honors_source_path_override(tmp_path: Path) -> None:
    """``training.replay.source_path`` must beat ``experience.path``."""
    cfg = load_settings()
    override = tmp_path / "override.lmdb"
    cfg = cfg.model_copy(
        update={
            "training": cfg.training.model_copy(
                update={
                    "replay": cfg.training.replay.model_copy(update={"source_path": str(override)})
                }
            )
        }
    )
    reader = build_replay_reader(cfg)
    assert Path(reader.path) == override
