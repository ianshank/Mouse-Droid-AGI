"""Regression tests for the Phase 2 replay block in ``jetson_production.yaml``.

These tests pin the operator-facing defaults so a future YAML edit cannot
silently flip the rover into a real-replay mode. The block defaults to
inert (`alpha_target=0.0`) — the rover keeps drawing only synthetic data
until a deliberate ramp PR lands.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mousedroid.config.schema import Settings


def _load_jetson_production() -> Settings:
    """Load ``config/jetson_production.yaml`` through the canonical Settings."""
    raw = yaml.safe_load(Path("config/jetson_production.yaml").read_text(encoding="utf-8"))
    return Settings.model_validate(raw)


def test_replay_mixer_block_loads_in_jetson_production() -> None:
    """The new `training.replay_mixer` block must parse without error."""
    cfg = _load_jetson_production()
    assert cfg.training.replay_mixer is not None


def test_replay_mixer_default_inert_alpha() -> None:
    """`alpha_target` must default to 0.0 so legacy training stays byte-identical."""
    cfg = _load_jetson_production()
    assert cfg.training.replay_mixer.alpha_target == 0.0


def test_replay_debug_log_disabled_by_default() -> None:
    """`debug_log_every_n` must default to 0 (off) so the rover doesn't flood logs."""
    cfg = _load_jetson_production()
    assert cfg.training.replay_mixer.debug_log_every_n == 0


def test_replay_mixer_chunk_size_matches_reader_default() -> None:
    """Chunk size 64 keeps the LMDB reader's per-chunk RAM bounded on the 8 GB Orin."""
    cfg = _load_jetson_production()
    assert cfg.training.replay_mixer.chunk_size == 64


def test_replay_mixer_log_cadence_present() -> None:
    """The INFO-level `mixer_ratio_check` cadence must be operator-visible."""
    cfg = _load_jetson_production()
    assert cfg.training.replay_mixer.log_every_n > 0
