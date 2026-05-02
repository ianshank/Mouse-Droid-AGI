"""Phase 2 — Real-episode replay CLI.

Operator entry point that exercises the chunked LMDB replay reader and the
deterministic sim/real mixer end-to-end.

Usage:
    # Smoke / planning: verify mixer ratio without touching real LMDB
    python -m training.replay_real_episodes \
        --config config/local_training.yaml --dry-run --draws 10000

    # Actually mix replay records into a synthetic stream
    python -m training.replay_real_episodes \
        --config config/jetson_production.yaml --use-real-replay --draws 1000

The script intentionally does **not** kick off RSSM or PPO training — it is a
focused validator for the replay pipeline. The auxiliary BC loss in
``train_constitutional_rl.py`` consumes the same ``LMDBReplayReader`` factory
when ``OfflineRLConfig.real_supervised_weight > 0``.

Memory budget: the CLI **streams** real records from the LMDB store one chunk
at a time (chunk size ``--chunk-size``) and feeds them into the mixer through
a synchronous bridge generator. The whole DB is never materialised in RAM —
this matches the 8 GB Jetson Orin Nano budget the rest of the replay path was
designed against.

Wiring: real-replay reader construction goes through
:func:`mousedroid.factory.build_replay_reader` so the CLI honours
``training.replay.source_path`` overrides and stays consistent with
``train_constitutional_rl`` (CLAUDE.md invariant 2 — factory.py is the
single wiring point).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from mousedroid.config.loader import load_settings
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.factory import build_replay_reader
from mousedroid.logging.setup import get_logger
from mousedroid.training.replay import MixerConfig, RealSimMixer

if TYPE_CHECKING:
    from mousedroid.training.replay.lmdb_reader import ReplayReaderProtocol

_log = get_logger(__name__)

# CLI defaults — overridable per-invocation. Kept module-level so they are
# discoverable and surface cleanly in --help.
DEFAULT_DRAWS: int = 10_000
DEFAULT_CHUNK_SIZE: int = 64
_LOG_ALPHA_PRECISION: int = 4


def _synthetic_iter(n: int) -> Iterator[MouseDroidExperienceRecord]:
    """Yield ``n`` zero-filled stub records as the sim source."""
    for _ in range(n):
        yield MouseDroidExperienceRecord()


def _stream_real_records_sync(
    reader: ReplayReaderProtocol,
    chunk_size: int,
) -> Iterator[MouseDroidExperienceRecord]:
    """Bridge the reader's async chunk stream to a synchronous record iterator.

    Pulls one chunk at a time through a private event loop, yielding records
    one-by-one. The whole DB is never held in memory at once — at most
    ``chunk_size`` records (one chunk's worth) are resident.

    Args:
        reader: Reader produced by :func:`build_replay_reader`.
        chunk_size: Maximum records per LMDB read.

    Yields:
        Decoded experience records, in LMDB key order.
    """
    loop = asyncio.new_event_loop()
    try:
        agen = reader.stream(chunk_size).__aiter__()
        while True:
            try:
                chunk = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                return
            yield from chunk
    finally:
        # Best-effort async-generator cleanup before tearing down the loop.
        # `aclose()` is idempotent and safe even if the generator already
        # returned; running it inside the same loop avoids "Event loop is
        # closed" warnings under py3.12+. `agen` may have already completed
        # or never been bound (loop initialisation could race) — both raise
        # but neither indicates a cleanup leak we can act on.
        with contextlib.suppress(StopAsyncIteration, RuntimeError):
            loop.run_until_complete(agen.aclose())
        loop.close()


def _run(
    config_path: Path,
    *,
    dry_run: bool,
    use_real_replay: bool,
    draws: int,
    chunk_size: int,
    alpha_target: float | None,
    seed: int | None,
) -> int:
    """Execute the replay smoke. Returns the process exit code."""
    cfg = load_settings(Path(config_path))

    # Real source: empty in dry-run; bounded LMDB pull otherwise.
    real_iter: Iterator[MouseDroidExperienceRecord] = iter(())
    reader: ReplayReaderProtocol | None = None
    reader_stats: dict[str, int] = {
        "read_records": 0,
        "skipped_schema_mismatch": 0,
        "chunks_yielded": 0,
    }
    if use_real_replay and dry_run:
        _log.info("replay_cli_dry_run_overrides_real", note="--dry-run wins over --use-real-replay")

    if use_real_replay and not dry_run:
        # Route through the factory so cfg.training.replay.source_path is
        # honoured and the concrete reader type stays hidden behind the
        # protocol (CLAUDE.md invariants 1+2).
        reader = build_replay_reader(cfg)
        real_iter = _stream_real_records_sync(reader, chunk_size)

    # Pull defaults from Settings; CLI flags override per-invocation.
    mixer_cfg = MixerConfig.from_settings(cfg.training.replay_mixer)
    if alpha_target is None and use_real_replay and mixer_cfg.alpha_target == 0.0:
        # Legacy fallback for invocations that pre-date replay_mixer.
        alpha_target = float(cfg.training.replay.real_episode_ratio)
    if alpha_target is not None:
        mixer_cfg = mixer_cfg.model_copy(update={"alpha_target": alpha_target})
    if seed is not None:
        mixer_cfg = mixer_cfg.model_copy(update={"seed": seed})

    mixer = RealSimMixer(
        sim_source=_synthetic_iter(draws),
        real_source=real_iter,
        cfg=mixer_cfg,
    )

    consumed = 0
    for _ in mixer:
        consumed += 1
        if consumed >= draws:
            break

    if reader is not None:
        reader_stats = reader.stats

    realized = mixer.stats["realized_alpha"]
    _log.info(
        "replay_cli_summary",
        config=str(config_path),
        dry_run=dry_run,
        use_real_replay=use_real_replay,
        draws=consumed,
        target_alpha=mixer_cfg.alpha_target,
        realized_alpha=round(realized, _LOG_ALPHA_PRECISION),
        real_drawn=int(mixer.stats["real_drawn"]),
        sim_drawn=int(mixer.stats["sim_drawn"]),
        reader=reader_stats,
    )

    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 2 real-episode replay loop probe.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip real LMDB; verify mixer determinism over --draws synthetic samples.",
    )
    parser.add_argument(
        "--use-real-replay",
        action="store_true",
        help="Drain the real LMDB store and feed it into the mixer's real source.",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=DEFAULT_DRAWS,
        help="Number of mixer samples to consume.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="LMDB chunk size for the async reader.",
    )
    parser.add_argument(
        "--alpha-target",
        type=float,
        default=None,
        help="Override MixerConfig.alpha_target.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic RNG seed override.",
    )
    args = parser.parse_args()
    return _run(
        config_path=args.config,
        dry_run=bool(args.dry_run),
        use_real_replay=bool(args.use_real_replay),
        draws=int(args.draws),
        chunk_size=int(args.chunk_size),
        alpha_target=args.alpha_target,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
