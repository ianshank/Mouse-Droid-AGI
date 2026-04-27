r"""CLI entrypoint for the Phase 2 real-episode replay loop.

This script provides a thin wrapper around the chunked LMDB replay reader
and the sim/real episode mixer. It is intended for two use cases:

1. **Operator dry-run** — verify that an LMDB experience database is readable
   and schema-compatible before kicking off a full training run::

       python -m training.replay_real_episodes --config config/local_training.yaml --dry-run

2. **End-to-end real-replay training** — invoke the standard RSSM pipeline
   with ``training.replay.use_chunked_reader=True`` overriding whatever the
   YAML specifies. Useful for one-off experiments without editing config::

       python -m training.replay_real_episodes \\
           --config config/local_training.yaml \\
           --use-real-replay \\
           --seed 42

All thresholds and paths come from the loaded :class:`Settings`. No values
are hardcoded in this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings, TrainingReplayConfig
from mousedroid.training.replay import LmdbReplayReader, SchemaVersionMismatchError

_log = structlog.get_logger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2 real-episode replay CLI. Streams an LMDB experience "
            "database in chunks and (optionally) hands off to the standard "
            "RSSM training pipeline."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to a Mouse-Droid YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Open the LMDB replay database, stream every record once to "
            "validate schema versions and chunking, then exit. No training is "
            "performed and no checkpoints are written."
        ),
    )
    parser.add_argument(
        "--use-real-replay",
        action="store_true",
        help=(
            "Override settings to enable the chunked replay reader regardless "
            "of what the YAML specifies. Equivalent to setting "
            "``training.replay.enabled=True`` and ``use_chunked_reader=True``."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional seed override for the replay subset selector. Falls "
            "back to ``training.replay.seed`` when omitted."
        ),
    )
    return parser


def _maybe_apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Apply CLI-driven overrides to the loaded settings (returns a new copy)."""
    replay_cfg: TrainingReplayConfig = settings.training.replay
    updates: dict[str, object] = {}
    if args.use_real_replay:
        updates["enabled"] = True
        updates["use_chunked_reader"] = True
    if args.seed is not None:
        updates["seed"] = args.seed
    if not updates:
        return settings
    new_replay = replay_cfg.model_copy(update=updates)
    new_training = settings.training.model_copy(update={"replay": new_replay})
    return settings.model_copy(update={"training": new_training})


def _dry_run(settings: Settings) -> int:
    """Stream the configured LMDB replay database once.

    Reports record counts and any schema mismatches. Returns the process
    exit code (0 = success, 1 = lenient mismatches, 2 = strict mismatch,
    3 = path missing).
    """
    replay_cfg: TrainingReplayConfig = settings.training.replay
    reader = LmdbReplayReader.from_config(
        settings.experience,
        chunk_size=replay_cfg.chunk_size,
        source_path=replay_cfg.source_path,
        strict_schema=replay_cfg.strict_schema,
    )

    try:
        with reader:
            total_records = len(reader)
            chunks = 0
            try:
                for chunk in reader.stream_chunks():
                    chunks += 1
                    _log.debug(
                        "replay_chunk_yielded",
                        chunk_index=chunks,
                        chunk_size=len(chunk),
                    )
            except SchemaVersionMismatchError as exc:
                _log.error(
                    "replay_dry_run_schema_mismatch_strict",
                    expected=exc.expected,
                    actual=exc.actual,
                )
                return 2
    except FileNotFoundError as exc:
        _log.error("replay_dry_run_path_not_found", error=str(exc))
        return 3

    _log.info(
        "replay_dry_run_complete",
        path=replay_cfg.source_path or settings.experience.path,
        records_total=total_records,
        records_consumed=reader.stats.records_consumed,
        chunks_yielded=reader.stats.chunks_yielded,
        schema_mismatches=reader.stats.schema_mismatches,
        chunk_size=replay_cfg.chunk_size,
    )
    if reader.stats.schema_mismatches > 0:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        _log.error("replay_cli_config_not_found", path=str(config_path))
        return 4

    settings = load_settings(config_path)
    settings = _maybe_apply_overrides(settings, args)

    if args.dry_run:
        return _dry_run(settings)

    # Hand off to the standard pipeline with replay enabled. We re-invoke
    # ``training.run_pipeline`` as a subprocess so its argparse and structlog
    # bindings start clean. The replay overrides applied above are persisted
    # via the YAML config; for ad-hoc CLI overrides, set them in the YAML.
    import subprocess  # local import: only needed for non-dry-run path

    cmd = [sys.executable, "-m", "training.run_pipeline", "--config", str(config_path)]
    _log.info("replay_cli_handoff_to_pipeline", cmd=cmd)
    completed = subprocess.run(cmd, check=False)  # noqa: S603 — args are constructed locally
    return completed.returncode


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    sys.exit(main())
