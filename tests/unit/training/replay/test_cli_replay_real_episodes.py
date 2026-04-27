"""Smoke tests for the Phase 2 replay CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from training import replay_real_episodes as cli


def test_dry_run_with_no_replay_returns_zero(tmp_path: Path) -> None:
    """--dry-run path: no LMDB, mixer drains synthetic source only."""
    rc = cli._run(
        config_path=Path("config/default.yaml"),
        dry_run=True,
        use_real_replay=False,
        draws=64,
        chunk_size=16,
        alpha_target=0.0,
        seed=0,
    )
    assert rc == 0


def test_dry_run_overrides_use_real_replay(tmp_path: Path) -> None:
    """--dry-run + --use-real-replay must skip the LMDB open path."""
    rc = cli._run(
        config_path=Path("config/default.yaml"),
        dry_run=True,
        use_real_replay=True,  # would otherwise touch a real LMDB
        draws=32,
        chunk_size=8,
        alpha_target=0.5,
        seed=42,
    )
    assert rc == 0


def test_use_real_replay_against_empty_lmdb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--use-real-replay against an empty path must log + return 0."""
    from mousedroid.config.loader import load_settings

    base_cfg = load_settings()
    redirected = base_cfg.model_copy(
        update={
            "experience": base_cfg.experience.model_copy(
                update={"path": str(tmp_path / "empty.lmdb")}
            )
        }
    )

    def _fake_loader(_path: str) -> object:
        return redirected

    monkeypatch.setattr(cli, "load_settings", _fake_loader)

    rc = cli._run(
        config_path=Path("config/default.yaml"),
        dry_run=False,
        use_real_replay=True,
        draws=16,
        chunk_size=4,
        alpha_target=0.0,
        seed=7,
    )
    assert rc == 0


def test_main_invokes_argparse(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` must wire argv -> _run and return its exit code."""
    captured: dict[str, object] = {}

    def _fake_run(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "_run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_real_episodes",
            "--config",
            "config/default.yaml",
            "--dry-run",
            "--draws",
            "8",
            "--seed",
            "1",
        ],
    )
    rc = cli.main()
    assert rc == 0
    assert captured["dry_run"] is True
    assert captured["draws"] == 8
    assert captured["seed"] == 1
