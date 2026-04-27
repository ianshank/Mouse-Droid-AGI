"""Integration tests for the Phase 2 replay CLI (``training.replay_real_episodes``)."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import lmdb
import numpy as np
import yaml
from training.replay_real_episodes import _build_arg_parser, _maybe_apply_overrides, main

from mousedroid.experience.record import MouseDroidExperienceRecord

GB_TO_BYTES = 1024**3


def _populate(path: Path, n: int) -> None:
    map_size = max(1, math.ceil(0.001 * GB_TO_BYTES))
    env = lmdb.open(str(path), map_size=map_size, max_dbs=1)
    try:
        with env.begin(write=True) as txn:
            for i in range(n):
                rec = MouseDroidExperienceRecord(
                    timestamp=float(i) * 0.1,
                    vision_features=np.full(8, float(i), dtype=np.float32),
                    distance_m=0.5,
                    motor_state=np.zeros(4, dtype=np.float32),
                    action=np.zeros(3, dtype=np.float32),
                    reward=float(i),
                    surprise=0.0,
                )
                txn.put(struct.pack(">Q", i), rec.serialize())
    finally:
        env.close()


def _write_minimal_config(tmp_path: Path, lmdb_path: Path) -> Path:
    """Write a minimal valid YAML overlay for the CLI."""
    cfg = {
        "platform": "mouse_droid",
        "mock_hardware": True,
        "experience": {
            "path": str(lmdb_path),
            "map_size_gb": 0.001,
            "flush_every_n": 1,
        },
        "training": {
            "replay": {
                "enabled": True,
                "use_chunked_reader": True,
                "chunk_size": 4,
                "source_path": str(lmdb_path),
            },
        },
    }
    p = tmp_path / "replay.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_dry_run_succeeds_on_valid_db(tmp_path: Path) -> None:
    db = tmp_path / "replay_db"
    _populate(db, n=10)
    cfg_path = _write_minimal_config(tmp_path, db)
    rc = main(["--config", str(cfg_path), "--dry-run"])
    assert rc == 0


def test_dry_run_returns_nonzero_on_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    cfg_path = _write_minimal_config(tmp_path, missing)
    rc = main(["--config", str(cfg_path), "--dry-run"])
    assert rc == 3


def test_dry_run_returns_nonzero_on_missing_config(tmp_path: Path) -> None:
    rc = main(["--config", str(tmp_path / "no_such.yaml"), "--dry-run"])
    assert rc == 4


def test_use_real_replay_override_applies(tmp_path: Path) -> None:
    db = tmp_path / "replay_db"
    _populate(db, n=2)
    # Write a config where replay is DISABLED — CLI flag should override.
    cfg = {
        "platform": "mouse_droid",
        "mock_hardware": True,
        "experience": {"path": str(db), "map_size_gb": 0.001, "flush_every_n": 1},
        "training": {
            "replay": {
                "enabled": False,
                "use_chunked_reader": False,
                "source_path": str(db),
            }
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    from mousedroid.config.loader import load_settings

    settings = load_settings(cfg_path)
    parser = _build_arg_parser()
    args = parser.parse_args(["--config", str(cfg_path), "--use-real-replay", "--seed", "7"])
    overridden = _maybe_apply_overrides(settings, args)

    assert overridden.training.replay.enabled is True
    assert overridden.training.replay.use_chunked_reader is True
    assert overridden.training.replay.seed == 7
    # Original settings unchanged (immutability check).
    assert settings.training.replay.enabled is False


def test_no_overrides_returns_settings_unchanged(tmp_path: Path) -> None:
    db = tmp_path / "replay_db"
    _populate(db, n=1)
    cfg_path = _write_minimal_config(tmp_path, db)
    from mousedroid.config.loader import load_settings

    settings = load_settings(cfg_path)
    parser = _build_arg_parser()
    args = parser.parse_args(["--config", str(cfg_path)])
    out = _maybe_apply_overrides(settings, args)
    # Same Pydantic object (no copy) when no overrides applied.
    assert out is settings


def test_dry_run_reports_schema_mismatch_count(tmp_path: Path) -> None:
    """Inject a record with a future schema_version and confirm the dry-run
    surfaces the mismatch via a non-zero exit code (lenient mode)."""
    import msgpack

    db = tmp_path / "replay_db"
    map_size = max(1, math.ceil(0.001 * GB_TO_BYTES))
    env = lmdb.open(str(db), map_size=map_size, max_dbs=1)
    try:
        with env.begin(write=True) as txn:
            ok = MouseDroidExperienceRecord(
                timestamp=0.0,
                vision_features=np.zeros(8, dtype=np.float32),
                distance_m=0.5,
                motor_state=np.zeros(4, dtype=np.float32),
                action=np.zeros(3, dtype=np.float32),
                reward=0.0,
                surprise=0.0,
            )
            txn.put(struct.pack(">Q", 0), ok.serialize())
            bad = msgpack.packb({"schema_version": 999, "timestamp": 0.0})
            txn.put(struct.pack(">Q", 1), bad)
    finally:
        env.close()
    cfg_path = _write_minimal_config(tmp_path, db)
    rc = main(["--config", str(cfg_path), "--dry-run"])
    assert rc == 1  # success-with-mismatches
