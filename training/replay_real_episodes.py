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
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterator
from pathlib import Path

from mousedroid.config.loader import load_settings
from mousedroid.experience.record import MouseDroidExperienceRecord
from mousedroid.logging.setup import get_logger
from mousedroid.training.replay import (
    LMDBReplayReader,
    MixerConfig,
    RealSimMixer,
)

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


async def _drain_reader(
    reader: LMDBReplayReader,
    chunk_size: int,
) -> list[MouseDroidExperienceRecord]:
    """Pull every record out of ``reader.stream`` into a list."""
    records: list[MouseDroidExperienceRecord] = []
    async for chunk in reader.stream(chunk_size):
        records.extend(chunk)
    return records


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
    real_records: list[MouseDroidExperienceRecord] = []
    reader_stats: dict[str, int] = {
        "read_records": 0,
        "skipped_schema_mismatch": 0,
        "chunks_yielded": 0,
    }
    if use_real_replay and dry_run:
        _log.info("replay_cli_dry_run_overrides_real", note="--dry-run wins over --use-real-replay")

    if use_real_replay and not dry_run:
        reader = LMDBReplayReader(cfg.experience)
        real_records = asyncio.run(_drain_reader(reader, chunk_size))
        reader_stats = reader.stats
        if not real_records:
            _log.warning(
                "replay_cli_no_real_records",
                path=str(reader.path),
            )

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
        real_source=iter(real_records),
        cfg=mixer_cfg,
    )

    consumed = 0
    for _ in mixer:
        consumed += 1
        if consumed >= draws:
            break

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
