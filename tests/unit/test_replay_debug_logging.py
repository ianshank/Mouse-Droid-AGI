"""Unit tests for the throttled DEBUG logging in `RatioMixer` and `LMDBReplayReader`.

Operator-facing live triage knob: `cfg.training.replay_mixer.debug_log_every_n`.
0 disables the DEBUG-level `mixer_draw` and `replay_chunk_decoded` lines so
production logs stay clean; positive N emits one line per N operations.

structlog routes to stdout via the project's :mod:`mousedroid.logging.setup`,
so these tests assert against pytest's `capsys` rather than `caplog`.
"""

from __future__ import annotations

import asyncio
import struct
import time
from collections.abc import Iterator
from pathlib import Path

import lmdb
import numpy as np
import pytest

from mousedroid.config.schema import ExperienceConfig
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.training.replay import LMDBReplayReader, MixerConfig, RealSimMixer
from tests import TEST_EXPERIENCE_MAP_SIZE_GB


def _drain(items: int) -> Iterator[int]:
    """Synthetic sim source — bounded so the mixer eventually stops."""
    yield from range(items)


def _empty_real() -> Iterator[int]:
    """Empty real source — every draw falls back to sim."""
    return iter(())


def _captured_text(capsys: pytest.CaptureFixture[str]) -> str:
    """Combined stdout + stderr the test has captured so far."""
    captured = capsys.readouterr()
    return captured.out + captured.err


# ---------------------------------------------------------------------------
# RatioMixer debug-throttle behaviour
# ---------------------------------------------------------------------------


def test_mixer_debug_log_disabled_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    """`debug_log_every_n` defaults to 0 → no `mixer_draw` lines at any log level."""
    cfg = MixerConfig()
    mixer = RealSimMixer(sim_source=_drain(20), real_source=_empty_real(), cfg=cfg)

    list(mixer)

    assert "mixer_draw" not in _captured_text(capsys)


def test_mixer_debug_log_fires_at_cadence(capsys: pytest.CaptureFixture[str]) -> None:
    """With `debug_log_every_n=2` over 10 successful draws, expect 5 `mixer_draw` lines."""
    cfg = MixerConfig(debug_log_every_n=2)
    mixer = RealSimMixer(sim_source=_drain(10), real_source=_empty_real(), cfg=cfg)

    list(mixer)

    assert _captured_text(capsys).count("mixer_draw") == 5


def test_mixer_debug_log_with_one_per_step(capsys: pytest.CaptureFixture[str]) -> None:
    """`debug_log_every_n=1` emits a `mixer_draw` line on every successful draw."""
    cfg = MixerConfig(debug_log_every_n=1)
    mixer = RealSimMixer(sim_source=_drain(7), real_source=_empty_real(), cfg=cfg)

    list(mixer)

    assert _captured_text(capsys).count("mixer_draw") == 7


def test_mixer_debug_log_independent_of_info_cadence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The DEBUG throttle does not depend on `log_every_n`."""
    # log_every_n=500 (default) means no `mixer_ratio_check` lines fire over
    # 10 draws, but debug_log_every_n=5 should still fire twice.
    cfg = MixerConfig(debug_log_every_n=5)
    mixer = RealSimMixer(sim_source=_drain(10), real_source=_empty_real(), cfg=cfg)

    list(mixer)

    text = _captured_text(capsys)
    assert text.count("mixer_ratio_check") == 0
    assert text.count("mixer_draw") == 2


# ---------------------------------------------------------------------------
# LMDBReplayReader debug-throttle behaviour
# ---------------------------------------------------------------------------


def _populate(path: Path, n: int) -> None:
    env = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    base = time.time()
    with env.begin(write=True) as txn:
        for i in range(n):
            key = struct.pack(">Q", int(base * 1_000_000) + i)
            record = MouseDroidExperienceRecord(
                vision_features=np.zeros(256, dtype=np.float32),
                motor_state=np.zeros(4, dtype=np.float32),
                action=np.zeros(3, dtype=np.float32),
                reward=float(i),
            )
            txn.put(key, record.serialize())
    env.close()


def _cfg(path: Path) -> ExperienceConfig:
    return ExperienceConfig(
        path=str(path),
        map_size_gb=TEST_EXPERIENCE_MAP_SIZE_GB,
        flush_every_n=5,
    )


def _drain_reader(reader: LMDBReplayReader, chunk_size: int) -> int:
    async def _go() -> int:
        n_chunks = 0
        async for _ in reader.stream(chunk_size):
            n_chunks += 1
        return n_chunks

    return asyncio.run(_go())


def test_reader_debug_log_disabled_by_default(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Reader without `debug_log_every_n` produces no `replay_chunk_decoded` lines."""
    _populate(tmp_path, n=64)

    reader = LMDBReplayReader(_cfg(tmp_path))
    n_chunks = _drain_reader(reader, chunk_size=8)

    assert n_chunks == 8
    assert "replay_chunk_decoded" not in _captured_text(capsys)


def test_reader_debug_log_fires_at_cadence(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """`debug_log_every_n=4` over 8 chunks yields 2 `replay_chunk_decoded` lines."""
    _populate(tmp_path, n=64)

    reader = LMDBReplayReader(_cfg(tmp_path), debug_log_every_n=4)
    _drain_reader(reader, chunk_size=8)

    assert _captured_text(capsys).count("replay_chunk_decoded") == 2


def test_reader_negative_debug_log_clamps_to_zero(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A nonsense `-3` value clamps to 0 (off) — never raises, no debug lines."""
    _populate(tmp_path, n=8)
    reader = LMDBReplayReader(_cfg(tmp_path), debug_log_every_n=-3)

    assert _drain_reader(reader, chunk_size=4) == 2
    assert "replay_chunk_decoded" not in _captured_text(capsys)
